"""Claude 适配器（Anthropic Messages 协议，``provider_type = anthropic_messages``）。

同一个适配器接 Bedrock（``bedrock-runtime`` / ``bedrock-mantle`` 的 ``/anthropic/v1``）
和 ``api.anthropic.com/v1``：请求发到 ``{base_url}/messages``，Key 放在 ``x-api-key``。

三条与另两个适配器不同的约定（见 docs/模型协议适配器改造方案.md 第 5 节）：

- **默认不发采样参数**。新模型收到 ``temperature``/``top_p`` 直接 400；只有连接显式配置才发。
  ``thinking`` 与 ``effort`` 按模型能力表决定（docs/模型思考能力表方案.md）：能关的按表关闭，
  关不掉的（Opus 5.5 收到 ``thinking: disabled`` 会 400）发最低强度；连接显式配置优先。
- **结构化输出有两种模式**。Bedrock mantle 与第三方兼容端点不支持
  ``output_config.format``，那里只靠指令约束；两种模式都由调用方的本地校验兜底，
  发送前删掉 Claude 不支持的 schema 约束。
- **复现身份记录发出的请求**：没发的采样记 API 名义默认值 ``1``，没发的 thinking
  记 ``null``；effort 写进 ``model_version``（PromptEnvelope 的 provider 是封闭结构）。

提示词与 Chat 适配器逐字相同（``legacy_prompts`` 与 ``core_view``），不引入新文本。
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

import httpx

from backend.app.core.config import settings
from backend.app.services.llm import thinking_capabilities as thinking
from backend.app.services.llm.base import LLMScorer
from backend.app.services.llm.base import validated_envelope_provider
from backend.app.services.llm.core_adapter import _decimal_text
from backend.app.services.llm.core_adapter import validated_core_envelope_provider
from backend.app.services.llm.core_view import build_core_group_request
from backend.app.services.llm.core_view import build_core_request
from backend.app.services.llm.core_view import compression_summary
from backend.app.services.llm.core_view import decode_core_group_response
from backend.app.services.llm.core_view import decode_core_response
from backend.app.services.llm.core_view import preflight_core_request
from backend.app.services.llm.errors import ProviderJSONOutputError
from backend.app.services.llm.legacy_prompts import chat_envelope_instructions
from backend.app.services.llm.legacy_prompts import chat_input_payload
from backend.app.services.llm.legacy_prompts import chat_instructions
from backend.app.services.llm.legacy_prompts import envelope_score_schema
from backend.app.services.llm.legacy_prompts import score_schema
from backend.app.services.llm.legacy_prompts import strip_json_fence
from backend.app.services.llm.transport import post_with_retry
from backend.app.services.llm.usage import UsageMeter
from backend.app.services.llm.usage import observed_usage


ANTHROPIC_VERSION = "2023-06-01"
PROTOCOL_VERSION = "messages-%s" % ANTHROPIC_VERSION
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
THINKING_TYPES = ("adaptive", "disabled")
STRUCTURED_OUTPUT_MODES = ("json_schema", "off")
# 只有核实过支持 output_config.format 的主机默认开启。Bedrock 两种主机都默认关闭：
# mantle 明确不支持；bedrock-runtime 上 Opus 4.7 及以后的模型由同一套设施承载。
_STRUCTURED_OUTPUT_HOSTS = frozenset({"api.anthropic.com"})
# Claude 结构化输出不支持的约束：数值、字符串长度、复杂数组约束。pattern 只部分支持，
# 一并删掉更稳（Core 的 evidence_unit_id 本来就用 enum 约束）。删掉的约束由调用方
# 的本地校验兜底。
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "pattern",
})
_SUBSCHEMA_KEYS = ("items", "additionalProperties", "not", "contains")
_SUBSCHEMA_LIST_KEYS = ("anyOf", "allOf", "oneOf", "prefixItems")
_SUBSCHEMA_MAP_KEYS = ("properties", "$defs", "definitions")


class MessagesJSONOutputError(ProviderJSONOutputError):
    """Safe metadata only; never include model content in this exception."""

    def __init__(self, reason):
        super().__init__(reason, "Anthropic Messages JSON output: " + reason)


def resolve_structured_output(base_url, options=None) -> str:
    """连接显式设置优先；否则只有核实过的主机开启结构化输出。"""

    mode = (options or {}).get("structured_output")
    if mode:
        return str(mode)
    host = (urlsplit(base_url or "").hostname or "").rstrip(".").casefold()
    return "json_schema" if host in _STRUCTURED_OUTPUT_HOSTS else "off"


def request_controls(*, temperature=None, top_p=None, thinking_type=None, effort=None, plan=None) -> dict:
    """调用参数（线上形状）。采样参数只发连接显式配置的。

    ``plan``（``ThinkingPlan``）给出时，思考与强度按它发；否则只发显式配置的 ``thinking_type`` / ``effort``
    （改动前的行为）。
    """

    body = {}
    if temperature is not None:
        body["temperature"] = float(temperature)
    elif top_p is not None:
        body["top_p"] = float(top_p)
    if plan is not None:
        body.update(plan.fields())
        return body
    if thinking_type:
        body["thinking"] = {"type": thinking_type}
    if effort:
        body["output_config"] = {"effort": effort}
    return body


def thinking_plans(base_url, model_name, *, thinking_type=None, effort=None, scope=None):
    """``(辅助计划, 评分计划)``，连接测试与适配器共用，测试与评分发同样的参数。"""

    return thinking.plans(
        thinking.MESSAGES, base_url, model_name, thinking_type=thinking_type, effort=effort, scope=scope,
    )


def sanitize_schema(schema):
    """删掉 Claude 结构化输出不支持的约束；属性名与结构不动。"""

    if isinstance(schema, list):
        return [sanitize_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYWORDS:
            continue
        if key == "minItems" and value not in (0, 1):
            continue
        if key in _SUBSCHEMA_MAP_KEYS and isinstance(value, dict):
            result[key] = {name: sanitize_schema(item) for name, item in value.items()}
        elif key in _SUBSCHEMA_KEYS or key in _SUBSCHEMA_LIST_KEYS:
            result[key] = sanitize_schema(value)
        else:
            result[key] = value
    return result


def usage_from_messages(data):
    usage = (data.get("usage") if isinstance(data, dict) else None) or {}

    def count(name):
        value = usage.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    inputs = [
        count(name)
        for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    ]
    # 文档：总输入 = input_tokens + 写缓存 + 读缓存（input_tokens 只是最后一个缓存断点之后的部分）。
    prompt = None if all(value is None for value in inputs) else sum(value or 0 for value in inputs)
    completion = count("output_tokens")
    total = prompt + completion if prompt is not None and completion is not None else None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def observed_usage_from_messages(data):
    """计量用：计费用量之外，再记有没有思考与结束原因（不含任何内容）。

    Claude 的思考 token 计入 ``output_tokens``，不单独报告；有没有思考看应答里有没有
    ``thinking`` / ``redacted_thinking`` 块。思考过时正式输出按 ``text`` 块的文字估算。
    """

    usage = usage_from_messages(data)
    data = data if isinstance(data, dict) else {}
    content = data.get("content") if isinstance(data.get("content"), list) else []
    stop_reason = data.get("stop_reason")
    return observed_usage(
        prompt=usage["prompt_tokens"],
        completion=usage["completion_tokens"],
        total=usage["total_tokens"],
        thinking_seen=any(
            isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
            for block in content
        ),
        finish_reason=stop_reason,
        truncated=stop_reason == "max_tokens",
        answer_text="\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ),
    )


def parse_messages_json(data):
    if not isinstance(data, dict):
        raise MessagesJSONOutputError("invalid_envelope")
    if data.get("type") == "error" or data.get("error"):
        raise MessagesJSONOutputError("error_envelope")
    stop_reason = data.get("stop_reason")
    if stop_reason == "refusal":
        # HTTP 200，但输出可能不符合 schema；不重试，交给人工复核。
        raise MessagesJSONOutputError("refused")
    if stop_reason == "max_tokens":
        # 思考 token 也计入 max_tokens；不赌「刚好写完」。
        raise MessagesJSONOutputError("output_truncated")
    content = data.get("content")
    if not isinstance(content, list):
        raise MessagesJSONOutputError("invalid_envelope")
    if stop_reason not in (None, "end_turn", "stop_sequence"):
        raise MessagesJSONOutputError("incomplete_output")
    text = "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )
    if not text.strip():
        raise MessagesJSONOutputError("empty_content")
    try:
        return json.loads(strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise MessagesJSONOutputError("invalid_json") from exc


class AnthropicMessagesScorer(LLMScorer):
    provider = "anthropic"

    def __init__(
        self,
        *,
        api_key,
        base_url,
        model_name,
        client=None,
        timeout_seconds=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
        thinking_type=None,
        effort=None,
        structured_output=None,
    ):
        if not api_key:
            raise ValueError("Anthropic Messages connection requires an API key")
        if not base_url:
            raise ValueError("Anthropic Messages connection requires a base URL")
        if not model_name:
            raise ValueError("Anthropic Messages connection requires a model name")
        if temperature is not None and top_p is not None:
            raise ValueError("Claude accepts temperature or top_p, not both")
        thinking_type = (thinking_type or "").strip() or None
        if thinking_type is not None and thinking_type not in THINKING_TYPES:
            raise ValueError("unsupported Claude thinking type")
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ValueError("unsupported Claude effort level")
        if structured_output is not None and structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError("unsupported Claude structured_output mode")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds_explicit = timeout_seconds is not None
        self.timeout_seconds = float(timeout_seconds or settings.ANTHROPIC_TIMEOUT_SECONDS)
        self.max_tokens_explicit = max_tokens is not None
        self.max_tokens = int(max_tokens or settings.ANTHROPIC_MAX_TOKENS)
        self.temperature = None if temperature is None else float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.thinking_type = thinking_type
        self.effort = effort
        self.structured_output = resolve_structured_output(
            self.base_url, {"structured_output": structured_output}
        )
        self.response_format = "json_schema" if self.structured_output == "json_schema" else "none"
        # 按模型能力表决定思考与强度：辅助计划给 AI 任务，评分计划给 score_*（scope 不是 all 时与改动前相同）。
        self.thinking_plan, self.scoring_thinking_plan = thinking_plans(
            self.base_url, self.model_name, thinking_type=thinking_type, effort=effort,
        )
        # 强度与放开后的请求上限写进 model_version（PromptEnvelope 的 provider 是封闭结构）。
        self.model_version = PROTOCOL_VERSION + thinking.model_version_suffix(
            self.scoring_thinking_plan, max_tokens_explicit=self.max_tokens_explicit,
        )
        self._owns_client = client is None
        self.usage_meter = UsageMeter()
        self.client = client or httpx.Client(timeout=self.timeout_seconds)

    def close(self):
        if self._owns_client and self.client is not None:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # -- reproducibility identity ------------------------------------------------

    def provider_controls(self):
        """The executable controls frozen into every PromptEnvelope (V1/V3/V4).

        Records what is sent: unsent sampling is the API's nominal default
        ``1``; thinking is the scoring plan's identity (unsent and off is
        ``null``, unsent but on by default is ``provider-default``).
        ``max_tokens`` is the answer budget; a raised request limit for a
        thinking model is recorded in ``model_version``.
        """

        return {
            "sampling": {
                "temperature": 1 if self.temperature is None else self.temperature,
                "top_p": 1 if self.top_p is None else self.top_p,
                "seed": None,
                "max_tokens": self.max_tokens,
            },
            "thinking": self.scoring_thinking_plan.identity,
            "response_format": self.response_format,
        }

    # -- request construction ----------------------------------------------------

    def _body(self, system, user, *, schema=None, max_tokens=None, plan=None):
        """``plan`` 默认是评分计划；``max_tokens`` 已是实发上限。"""

        plan = plan or self.scoring_thinking_plan
        body = {
            "model": self.model_name,
            "max_tokens": int(max_tokens or self._scoring_max_tokens(self.max_tokens)),
            "system": system,
            "messages": [{"role": "user", "content": user}],
            **request_controls(temperature=self.temperature, top_p=self.top_p, plan=plan),
        }
        if schema is not None and self.structured_output == "json_schema":
            body.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": sanitize_schema(schema),
            }
        return body

    def _send(self, body, **options):
        response = self._post_with_retry(body, **options)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise MessagesJSONOutputError("invalid_envelope") from exc

    # -- 思考计划：上限、超时与日志上下文 ---------------------------------------------------------

    def _json_plan(self, scoring=False):
        return self.scoring_thinking_plan if scoring else self.thinking_plan

    def request_output_tokens(self, answer_budget=None, *, scoring=False):
        """AI 任务的实发上限：``answer_budget`` 是正式输出预算（docs/模型思考能力表方案.md 第 5.2 节）。"""

        return thinking.request_max_tokens(
            self._json_plan(scoring), configured=self.max_tokens, explicit=self.max_tokens_explicit,
            answer_budget=answer_budget,
        )

    def request_timeout_seconds(self, default_timeout_seconds=None, *, scoring=False):
        if default_timeout_seconds is None or self.timeout_seconds_explicit:
            return None
        timeout = max(self.timeout_seconds, default_timeout_seconds)
        if self._json_plan(scoring).raises_output_limit:
            timeout = max(timeout, thinking.THINKING_ON_TIMEOUT_SECONDS)
        return timeout

    def _scoring_max_tokens(self, answer_budget):
        if self.scoring_thinking_plan.raises_output_limit and not self.max_tokens_explicit:
            return self.scoring_thinking_plan.raised_max_tokens()
        return int(answer_budget)

    def _context(self, body, plan, answer_budget):
        return thinking.call_context(plan, answer_budget=answer_budget, max_tokens=body.get("max_tokens"))

    def _post_with_retry(self, payload, *, attempts_limit=None, timeout_seconds=None, rate_limit_retries=None, deadline=None,
                         call_context=None, output_limit_fallback=None):
        return post_with_retry(
            self,
            url="%s/messages" % self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            payload=payload,
            provider_label=self.provider,
            operation="messages",
            model_parameters={
                "temperature": payload.get("temperature"),
                "top_p": payload.get("top_p"),
                "max_tokens": payload.get("max_tokens"),
                "effort": (payload.get("output_config") or {}).get("effort"),
            },
            configured_retries=settings.ANTHROPIC_MAX_RETRIES,
            usage_of=observed_usage_from_messages,
            success_output=lambda data: {"stop_reason": data.get("stop_reason")},
            attempts_limit=attempts_limit,
            timeout_seconds=timeout_seconds,
            rate_limit_retries=rate_limit_retries,
            deadline=deadline,
            call_context=call_context,
            output_limit_fallback=output_limit_fallback,
        )

    def _validate_frozen_controls(self, provider):
        controls = self.provider_controls()
        expected = {
            "sampling": {
                key: (value if key in {"seed", "max_tokens"} else _decimal_text(value))
                for key, value in controls["sampling"].items()
            },
            "thinking": controls["thinking"],
            "response_format": controls["response_format"],
        }
        mismatches = sorted(name for name, value in expected.items() if provider[name] != value)
        if mismatches:
            raise ValueError(
                "PromptEnvelope controls do not match the Claude adapter: %s" % ", ".join(mismatches)
            )

    # -- legacy scoring paths ----------------------------------------------------

    def score_criterion(self, paper, criterion, evidence_candidates, structure_checks, anchors=None):
        body = self._body(
            chat_instructions(criterion),
            chat_input_payload(paper, criterion, evidence_candidates, structure_checks, anchors),
            schema=score_schema(),
        )
        data = self._send(body, call_context=self._context(body, self.scoring_thinking_plan, self.max_tokens))
        output = parse_messages_json(data)
        output.setdefault("criterion_id", criterion.id)
        output.setdefault("criterion_name", criterion.name)
        output.setdefault("max_score", float(criterion.max_score))
        output["provider"] = self.provider
        output["provider_response_id"] = data.get("id")
        output["usage"] = usage_from_messages(data)
        return output

    def score_envelope(self, envelope):
        """Send the exact envelope already used for cache identity."""

        envelope, provider = validated_envelope_provider(self, envelope)
        if provider["response_schema"] != "criterion-score-v2":
            raise ValueError("unsupported PromptEnvelope response_schema")
        self._validate_frozen_controls(provider)
        envelope_payload = envelope.to_mapping()
        criterion = envelope_payload["criterion"]
        body = self._body(
            chat_envelope_instructions(criterion["scoring_mode"]),
            json.dumps(envelope_payload, ensure_ascii=False, separators=(",", ":")),
            schema=envelope_score_schema(criterion["scoring_mode"]),
            max_tokens=self._scoring_max_tokens(provider["sampling"]["max_tokens"]),
        )
        data = self._send(body, call_context=self._context(body, self.scoring_thinking_plan, self.max_tokens))
        output = parse_messages_json(data)
        output.setdefault("criterion_id", criterion["code"])
        output.setdefault("criterion_name", criterion["name"])
        output.setdefault("max_score", float(criterion["max_score"]))
        output["provider"] = self.provider
        output["provider_response_id"] = data.get("id")
        output["usage"] = usage_from_messages(data)
        return output

    # -- AtomicRule Core -----------------------------------------------------------

    def _post_core_request(self, envelope, provider, request):
        """Send exactly the provider view (never the identity envelope)."""

        preflight_core_request(envelope, request)
        body = self._body(
            request.system,
            request.user,
            schema=request.schema,
            # 信封记正式输出预算（Core 证据预算按它扣）；思考开着时实发上限放到封顶值，写在 model_version 里。
            max_tokens=self._scoring_max_tokens(provider["sampling"]["max_tokens"]),
        )
        self.last_view_summary = compression_summary(request)
        return parse_messages_json(
            self._send(body, call_context=self._context(body, self.scoring_thinking_plan, self.max_tokens))
        )

    def score_core_envelope(self, *, envelope):
        """Score one immutable PromptEnvelopeV3/V4 semantic rule via its view."""

        envelope, provider = validated_core_envelope_provider(self, envelope)
        request = build_core_request(envelope)
        raw = self._post_core_request(envelope, provider, request)
        return decode_core_response(envelope, raw, request)

    def score_core_group(self, *, envelopes, group_code):
        """Judge every tier of one mutex group in a single request."""

        validated = [validated_core_envelope_provider(self, envelope) for envelope in envelopes]
        envelopes = [envelope for envelope, _provider in validated]
        request = build_core_group_request(envelopes, group_code=group_code)
        raw = self._post_core_request(envelopes[0], validated[0][1], request)
        return decode_core_group_response(envelopes, raw, request)

    # -- generic JSON completion (drafting, classification, structure, coherence) --

    def complete_json(self, instructions, payload, *, response_schema=None, default_max_tokens=None, attempts_limit=None, default_timeout_seconds=None, rate_limit_retries=None, deadline=None, scoring=False):
        """``default_max_tokens`` 是正式输出预算；``scoring=True`` 用评分计划（语义一致性检查）。"""

        request_options = {}
        if attempts_limit is not None:
            request_options["attempts_limit"] = attempts_limit
        if rate_limit_retries:
            request_options["rate_limit_retries"] = rate_limit_retries
        if deadline is not None:
            request_options["deadline"] = deadline
        timeout = self.request_timeout_seconds(default_timeout_seconds, scoring=scoring)
        if timeout is not None:
            request_options["timeout_seconds"] = timeout
        if response_schema is not None:
            # 起草层已有一次格式修正；传输层再重试会把一次超时放大成多次等待（同 Chat）。
            request_options["attempts_limit"] = 1
        plan = self._json_plan(scoring)
        body = self._body(
            instructions,
            json.dumps(payload, ensure_ascii=False),
            schema=response_schema,
            max_tokens=self.request_output_tokens(default_max_tokens, scoring=scoring),
            plan=plan,
        )
        request_options["call_context"] = self._context(body, plan, default_max_tokens)
        fallback = None if scoring else thinking.output_limit_fallback(
            plan, field="max_tokens", sent=body["max_tokens"], configured=self.max_tokens,
            explicit=self.max_tokens_explicit, answer_budget=default_max_tokens,
        )
        if fallback:
            request_options["output_limit_fallback"] = fallback
        return parse_messages_json(self._send(body, **request_options))


__all__ = [
    "ANTHROPIC_VERSION",
    "AnthropicMessagesScorer",
    "EFFORT_LEVELS",
    "MessagesJSONOutputError",
    "STRUCTURED_OUTPUT_MODES",
    "THINKING_TYPES",
    "observed_usage_from_messages",
    "parse_messages_json",
    "request_controls",
    "resolve_structured_output",
    "thinking_plans",
    "sanitize_schema",
    "usage_from_messages",
]

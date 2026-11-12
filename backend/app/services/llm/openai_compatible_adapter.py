import json

import httpx

from backend.app.core.config import settings
from backend.app.services.llm.base import LLMScorer
from backend.app.services.llm.auth_headers import openai_auth_headers
from backend.app.services.llm.base import validated_envelope_provider
from backend.app.services.llm.core_adapter import validated_core_envelope_provider
from backend.app.services.llm.core_view import build_core_group_request
from backend.app.services.llm.core_view import build_core_request
from backend.app.services.llm.core_view import compression_summary
from backend.app.services.llm.core_view import decode_core_group_response
from backend.app.services.llm.core_view import decode_core_response
from backend.app.services.llm.core_view import with_repair
from backend.app.services.llm.core_view import preflight_core_request
from backend.app.services.llm.errors import ProviderJSONOutputError
from backend.app.services.llm.legacy_prompts import chat_envelope_instructions as _envelope_instructions
from backend.app.services.llm.legacy_prompts import chat_input_payload as _input_payload
from backend.app.services.llm.legacy_prompts import chat_instructions as _instructions
from backend.app.services.llm.legacy_prompts import strip_json_fence as _strip_json_fence
from backend.app.services.llm import thinking_capabilities as thinking
from backend.app.services.llm.hosts import OPENROUTER
from backend.app.services.llm.hosts import host_family
from backend.app.services.llm.thinking_params import thinking_request_fields
from backend.app.services.llm.tool_calls import parse_chat_tool_response
from backend.app.services.llm.transport import post_with_retry
from backend.app.services.llm.usage import UsageMeter
from backend.app.services.llm.usage import observed_usage


class OpenAICompatibleChatScorer(LLMScorer):
    provider = "openai_compatible"
    model_version = "chat-completions"

    def __init__(self, api_key=None, base_url=None, model_name=None, provider_name=None, client=None, timeout_seconds=None, max_tokens=None, temperature=None, top_p=None, response_format_json=None, thinking_type=None, service_tier=None):
        self.api_key = api_key or settings.OPENAI_COMPATIBLE_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_COMPATIBLE_API_KEY is required when LLM_PROVIDER=openai_compatible")
        self.base_url = base_url or settings.OPENAI_COMPATIBLE_BASE_URL
        if not self.base_url:
            raise ValueError("OPENAI_COMPATIBLE_BASE_URL is required when LLM_PROVIDER=openai_compatible")
        self.base_url = self.base_url.rstrip("/")
        self.model_name = model_name or settings.OPENAI_COMPATIBLE_MODEL
        self.provider_name = provider_name or settings.OPENAI_COMPATIBLE_PROVIDER_NAME
        self.timeout_seconds_explicit = timeout_seconds is not None
        self.timeout_seconds = float(timeout_seconds or settings.OPENAI_COMPATIBLE_TIMEOUT_SECONDS)
        self.max_tokens_explicit = max_tokens is not None
        self.max_tokens = int(max_tokens or settings.OPENAI_COMPATIBLE_MAX_TOKENS)
        self.temperature = float(temperature if temperature is not None else settings.OPENAI_COMPATIBLE_TEMPERATURE)
        # Chat endpoints disagree on the default top_p (Zhipu/Qwen are < 1),
        # so the frozen value is always sent; a connection may override it
        # (Bailian kimi-k3 rejects 1.0 but accepts 0.95).
        self.top_p = float(top_p if top_p is not None else 1)
        self.response_format_json = settings.OPENAI_COMPATIBLE_RESPONSE_FORMAT_JSON if response_format_json is None else bool(response_format_json)
        configured_thinking = settings.OPENAI_COMPATIBLE_THINKING_TYPE if thinking_type is None else thinking_type
        self.configured_thinking_type = str(configured_thinking or "").strip()
        # 按模型能力表决定思考（docs/模型思考能力表方案.md）：辅助计划给 AI 任务与评分助手，
        # 评分计划给 score_*（THINKING_CAPABILITY_SCOPE 不是 all 时与改动前逐字节相同）。
        self.thinking_plan, self.scoring_thinking_plan = thinking.plans(
            thinking.CHAT, self.base_url, self.model_name, thinking_type=configured_thinking,
        )
        # 运行身份读的是评分计划：Google 主机认 reasoning_effort 而不是 thinking 字段，记换算后的值。
        self.thinking_type = self.scoring_thinking_plan.identity_type
        self.model_version = type(self).model_version + thinking.model_version_suffix(
            self.scoring_thinking_plan, max_tokens_explicit=self.max_tokens_explicit,
        )
        self.service_tier = (
            settings.OPENAI_COMPATIBLE_SERVICE_TIER
            if service_tier is None
            else service_tier
        )
        if self.service_tier not in {None, "auto", "on_demand", "flex", "performance"}:
            raise ValueError("unsupported OpenAI-compatible service_tier")
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

    def score_criterion(self, paper, criterion, evidence_candidates, structure_checks, anchors=None):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _instructions(criterion)},
                {"role": "user", "content": _input_payload(paper, criterion, evidence_candidates, structure_checks, anchors)},
            ],
            "temperature": self.temperature,
            "max_tokens": self._scoring_max_tokens(self.max_tokens),
        }
        payload.update(self.scoring_thinking_plan.fields())
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if self.service_tier:
            payload["service_tier"] = self.service_tier

        response = self._post_with_retry(payload, call_context=self._scoring_context(payload))
        response.raise_for_status()
        data = response.json()
        output = _parse_chat_json_output(data)
        output.setdefault("criterion_id", criterion.id)
        output.setdefault("criterion_name", criterion.name)
        output.setdefault("max_score", float(criterion.max_score))
        output["provider"] = self.provider_name
        output["provider_response_id"] = data.get("id")
        output["usage"] = _usage_from_chat(data)
        return output

    def score_envelope(self, envelope):
        """Send the provider-ready envelope without reconstructing its inputs."""

        envelope, provider = validated_envelope_provider(self, envelope)
        envelope_payload = envelope.to_mapping()
        criterion = envelope_payload["criterion"]
        if provider["response_schema"] != "criterion-score-v2":
            raise ValueError("unsupported PromptEnvelope response_schema")
        if provider["response_format"] not in {"none", "json_object"}:
            raise ValueError("OpenAI-compatible PromptEnvelope response_format is unsupported")
        sampling = provider["sampling"]
        payload = {
            "model": provider["model"],
            "messages": [
                {
                    "role": "system",
                    "content": _envelope_instructions(criterion["scoring_mode"]),
                },
                {
                    "role": "user",
                    "content": json.dumps(envelope_payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            "max_tokens": self._envelope_max_tokens(provider),
        }
        if sampling["seed"] is not None:
            payload["seed"] = sampling["seed"]
        payload.update(self._envelope_thinking_fields(provider))
        if provider["response_format"] == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.service_tier:
            payload["service_tier"] = self.service_tier

        response = self._post_with_retry(payload, call_context=self._scoring_context(payload))
        response.raise_for_status()
        data = response.json()
        output = _parse_chat_json_output(data)
        output.setdefault("criterion_id", criterion["code"])
        output.setdefault("criterion_name", criterion["name"])
        output.setdefault("max_score", float(criterion["max_score"]))
        output["provider"] = self.provider_name
        output["provider_response_id"] = data.get("id")
        output["usage"] = _usage_from_chat(data)
        return output

    def _validated_core(self, envelope):
        envelope, provider = validated_core_envelope_provider(self, envelope)
        if provider["response_format"] not in {"none", "json_object"}:
            raise ValueError(
                "OpenAI-compatible PromptEnvelopeV3 response_format is unsupported"
            )
        return envelope, provider

    def _post_core_request(self, envelope, provider, request):
        """Send exactly the provider view (never the identity envelope)."""

        preflight_core_request(envelope, request)
        sampling = provider["sampling"]
        payload = {
            "model": provider["model"],
            "messages": [
                # The system message is identical for every rule of a kind, so
                # it stays a cacheable prefix; per-rule content is in the view.
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            # 信封记正式输出预算（Core 证据预算按它扣）；思考开着时实发上限放到封顶值，写在 model_version 里。
            "max_tokens": self._envelope_max_tokens(provider),
        }
        if request.repair_hint:
            # 重问：原来的两条消息一字不改，修正提示另起一条。
            payload["messages"].append({"role": "user", "content": request.repair_hint})
        if sampling["seed"] is not None:
            payload["seed"] = sampling["seed"]
        payload.update(self._envelope_thinking_fields(provider))
        if provider["response_format"] == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.service_tier:
            payload["service_tier"] = self.service_tier

        self.last_view_summary = compression_summary(request)
        response = self._post_with_retry(payload, call_context=self._scoring_context(payload))
        response.raise_for_status()
        return _parse_chat_json_output(response.json())

    def score_core_envelope(self, *, envelope, repair=None):
        """Score one immutable PromptEnvelopeV3/V4 semantic rule via its view."""

        envelope, provider = self._validated_core(envelope)
        request = with_repair(build_core_request(envelope), repair)
        raw = self._post_core_request(envelope, provider, request)
        return decode_core_response(envelope, raw, request)

    def score_core_group(self, *, envelopes, group_code, repair=None):
        """Judge every tier of one mutex group in a single request."""

        validated = [self._validated_core(envelope) for envelope in envelopes]
        envelopes = [envelope for envelope, _provider in validated]
        request = with_repair(build_core_group_request(envelopes, group_code=group_code), repair)
        raw = self._post_core_request(envelopes[0], validated[0][1], request)
        return decode_core_group_response(envelopes, raw, request)

    def complete_json(self, instructions, payload, *, response_schema=None, default_max_tokens=None, attempts_limit=None, default_timeout_seconds=None, rate_limit_retries=None, deadline=None, scoring=False):
        """``default_max_tokens`` 是正式输出预算；``scoring=True`` 用评分计划（语义一致性检查）。"""

        request_options = {"call_context": None}
        if attempts_limit is not None:
            request_options["attempts_limit"] = attempts_limit
        if rate_limit_retries:
            request_options["rate_limit_retries"] = rate_limit_retries
        if deadline is not None:
            request_options["deadline"] = deadline
        timeout = self.request_timeout_seconds(default_timeout_seconds, scoring=scoring)
        if timeout is not None:
            request_options["timeout_seconds"] = timeout
        plan = self._json_plan(scoring)
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.request_output_tokens(default_max_tokens, scoring=scoring),
        }
        body.update(plan.fields())
        if self.response_format_json:
            body["response_format"] = {"type": "json_object"}
        if response_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "rubric_rule_draft", "strict": True, "schema": response_schema,
            }}
        if self.service_tier:
            body["service_tier"] = self.service_tier
        request_options["call_context"] = thinking.call_context(
            plan, answer_budget=default_max_tokens, max_tokens=body["max_tokens"],
        )
        fallback = None if scoring else thinking.output_limit_fallback(
            plan, field="max_tokens", sent=body["max_tokens"], configured=self.max_tokens,
            explicit=self.max_tokens_explicit, answer_budget=default_max_tokens,
        )
        if fallback:
            request_options["output_limit_fallback"] = fallback
        if response_schema is not None:
            # ``reasoning`` is an OpenRouter extension.  Other OpenAI-compatible
            # providers (including Google AI Studio) may reject the same field,
            # even though they support strict ``json_schema`` output.
            # 能力表接管后由 OpenRouter 的族默认记录发出；旧行为（scope=off）保留这个特判。
            thinking_type = (plan.identity_type or "").strip() if plan.legacy else None
            if plan.legacy and host_family(self.base_url) == OPENROUTER and (
                not thinking_type or thinking_type == "disabled"
            ):
                body["reasoning"] = {"enabled": False}
            # The drafting layer already permits one schema repair. Avoid multiplying
            # that by transport retries inside a synchronous serverless request.
            response = self._post_with_retry(body, **{**request_options, "attempts_limit": 1})
        else:
            response = self._post_with_retry(body, **request_options)
        response.raise_for_status()
        return _parse_chat_json_output(response.json())

    def complete_with_tools(self, messages, tools, *, tool_choice="required", max_tokens=None, deadline=None,
                            timeout_seconds=None, rate_limit_retries=0, rate_limit_delay_seconds=None,
                            rate_limit_min_call_seconds=None):
        """原生工具调用一步（评分助手 Agent 回合，方案 §13）。

        请求参数在同一回合的各步之间保持不变（§12.1）：``tool_choice`` 固定，思考设置随连接，
        不按步骤调整。超时与 5xx 只试 1 次；429 可以按 ``rate_limit_retries`` 短等后重试
        （``rate_limit_delay_seconds``；等完之后离截止时间至少还要剩 ``rate_limit_min_call_seconds``，§13.2）。``messages`` 与 ``tools`` 由调用方确定性地生成。
        """

        body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": self.temperature,
            "max_tokens": (
                self.request_output_tokens(max_tokens)
                if self.thinking_plan.raises_output_limit
                else int(max_tokens or self.max_tokens)
            ),
        }
        body.update(self.thinking_plan.fields())
        if self.service_tier:
            body["service_tier"] = self.service_tier
        options = {"attempts_limit": 1, "deadline": deadline, "call_context": thinking.call_context(
            self.thinking_plan, answer_budget=max_tokens, max_tokens=body["max_tokens"],
        )}
        fallback = thinking.output_limit_fallback(
            self.thinking_plan, field="max_tokens", sent=body["max_tokens"],
            configured=int(max_tokens or self.max_tokens), explicit=self.max_tokens_explicit,
        )
        if fallback:
            options["output_limit_fallback"] = fallback
        if rate_limit_retries:
            options["rate_limit_retries"] = rate_limit_retries
            options["rate_limit_delay_seconds"] = rate_limit_delay_seconds
            options["rate_limit_min_call_seconds"] = rate_limit_min_call_seconds
        if timeout_seconds is not None and not self.timeout_seconds_explicit:
            options["timeout_seconds"] = timeout_seconds
        response = self._post_with_retry(body, **options)
        response.raise_for_status()
        return parse_chat_tool_response(response.json())

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
        """AI 任务的单次等待：连接显式超时优先；思考开着时不少于 THINKING_ON_TIMEOUT_SECONDS。"""

        if default_timeout_seconds is None or self.timeout_seconds_explicit:
            return None
        timeout = max(self.timeout_seconds, default_timeout_seconds)
        if self._json_plan(scoring).raises_output_limit:
            timeout = max(timeout, thinking.THINKING_ON_TIMEOUT_SECONDS)
        return timeout

    def _scoring_max_tokens(self, answer_budget):
        """评分请求的实发上限：信封与运行身份里记的是正式输出预算。"""

        if self.scoring_thinking_plan.raises_output_limit and not self.max_tokens_explicit:
            return self.scoring_thinking_plan.raised_max_tokens()
        return int(answer_budget)

    def _envelope_matches_plan(self, provider):
        return provider["thinking"] == self.scoring_thinking_plan.identity

    def _envelope_thinking_fields(self, provider):
        """信封冻结的思考设置为准：与评分计划一致时发计划的写法（含强度）；不一致的旧信封按改动前的换算发。"""

        if self._envelope_matches_plan(provider):
            return self.scoring_thinking_plan.fields()
        return thinking_request_fields(self.base_url, self.model_name, provider["thinking"]["type"])

    def _envelope_max_tokens(self, provider):
        answer_budget = provider["sampling"]["max_tokens"]
        if self._envelope_matches_plan(provider):
            return self._scoring_max_tokens(answer_budget)
        return int(answer_budget)

    def _scoring_context(self, payload):
        return thinking.call_context(
            self.scoring_thinking_plan, answer_budget=self.max_tokens, max_tokens=payload.get("max_tokens"),
        )

    def _post_with_retry(self, payload, *, attempts_limit=None, timeout_seconds=None, rate_limit_retries=None, deadline=None,
                         rate_limit_delay_seconds=None, rate_limit_min_call_seconds=None, call_context=None,
                         output_limit_fallback=None):
        return post_with_retry(
            self,
            url="%s/chat/completions" % self.base_url,
            headers={
                **openai_auth_headers(self.base_url, self.api_key),
                "Content-Type": "application/json",
            },
            payload=payload,
            provider_label=self.provider_name,
            operation="chat",
            model_parameters={
                "temperature": payload.get("temperature"),
                "top_p": payload.get("top_p"),
                "max_tokens": payload.get("max_tokens"),
            },
            configured_retries=settings.OPENAI_COMPATIBLE_MAX_RETRIES,
            usage_of=_observed_usage_from_chat,
            success_output=lambda data: {"service_tier": data.get("service_tier")},
            attempts_limit=attempts_limit,
            timeout_seconds=timeout_seconds,
            rate_limit_retries=rate_limit_retries,
            deadline=deadline,
            rate_limit_delay_seconds=rate_limit_delay_seconds,
            rate_limit_min_call_seconds=rate_limit_min_call_seconds,
            call_context=call_context,
            output_limit_fallback=output_limit_fallback,
        )


def _usage_from_chat(data):
    usage = data.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _observed_usage_from_chat(data):
    """计量用：计费用量之外，再记推理 token 与结束原因（不含任何内容）。

    推理 token 的位置各家不同：OpenAI、DeepSeek、OpenRouter 报在
    ``usage.completion_tokens_details.reasoning_tokens``（Vertex 也报，2026-10-19 核实）；不报的从合计推算；
    智谱、DeepSeek 还会在 ``message.reasoning_content`` 里给出思考文本。
    """

    data = data if isinstance(data, dict) else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details")
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    finish_reason = choice.get("finish_reason")
    return observed_usage(
        prompt=usage.get("prompt_tokens"),
        completion=usage.get("completion_tokens"),
        total=usage.get("total_tokens"),
        reported_reasoning=details.get("reasoning_tokens") if isinstance(details, dict) else None,
        thinking_seen=bool(
            str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
        ),
        finish_reason=finish_reason,
        truncated=finish_reason == "length",
        # 智谱只给思考文本、不报推理计数：正式输出按可见文字估算。
        answer_text=_message_content_to_text(message.get("content")),
    )


class ChatJSONOutputError(ProviderJSONOutputError):
    """Safe metadata only; never include provider content in this exception."""

    def __init__(self, reason):
        super().__init__(reason, "OpenAI-compatible JSON output: " + reason)


def _parse_chat_json_output(data):
    if not isinstance(data, dict):
        raise ChatJSONOutputError("invalid_envelope")
    if data.get("error"):
        raise ChatJSONOutputError("error_envelope")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ChatJSONOutputError("missing_choices")
    choice = choices[0]
    truncated = choice.get("finish_reason") == "length"
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ChatJSONOutputError("missing_message")
    text = _message_content_to_text(message.get("content"))
    if not text.strip():
        raise ChatJSONOutputError("output_truncated" if truncated else "empty_content")
    try:
        return json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ChatJSONOutputError("output_truncated" if truncated else "invalid_json") from exc


def _message_content_to_text(content):
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _empty_content_detail(choice):
    message = choice.get("message") or {}
    reasoning = str(message.get("reasoning_content") or "")
    return json.dumps(
        {
            "finish_reason": choice.get("finish_reason"),
            "choice_keys": sorted(choice.keys()),
            "message_keys": sorted(message.keys()),
            "reasoning_content_chars": len(reasoning),
            "diagnosis": (
                "模型返回了 reasoning_content 但没有最终 content，通常是 thinking 模式消耗了输出预算。"
                if reasoning
                else "模型响应没有最终 content。请检查模型名、thinking 参数、max_tokens 和提示词。"
            ),
        },
        ensure_ascii=False,
    )

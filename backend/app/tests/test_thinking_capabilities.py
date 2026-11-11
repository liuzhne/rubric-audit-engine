"""按模型能力表决定思考（docs/模型思考能力表方案.md）。

2026-11-10 生产上 glm-5.3 起草失败：它关不掉思考，连接没配置时什么都不发，模型按默认 max 强度思考，
把 6144 的输出上限用光（finish_reason=length）。规则：能关的关，关不掉的发最低强度、只限制正式输出。

全部用 httpx.MockTransport 截获请求，不调用真实模型。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

import httpx
import pytest

from backend.app.core.config import settings
from backend.app.services.llm import hosts
from backend.app.services.llm import thinking_capabilities as thinking
from backend.app.services.llm.anthropic_messages_adapter import AnthropicMessagesScorer
from backend.app.services.llm.core_adapter import core_runtime_provider_contract
from backend.app.services.llm.openai_adapter import OpenAIResponsesScorer
from backend.app.services.llm.openai_compatible_adapter import OpenAICompatibleChatScorer
from backend.app.services.llm.usage import ThinkingConsistencyGuard
from backend.app.services.llm.usage import ThinkingNotDisabledError
from backend.app.services.llm.usage import metered_call
from backend.app.services.llm.usage import observed_usage
from backend.app.services.scoring.core.contracts import _normalize_provider


VERTEX = "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/openai"
ZHIPU = "https://open.bigmodel.cn/api/paas/v4"
OPENROUTER = "https://openrouter.ai/api/v1"
ANTHROPIC = "https://api.anthropic.com/v1"
BEDROCK = "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1"
OPENAI = "https://api.openai.com/v1"
DASHSCOPE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MOONSHOT = "https://api.moonshot.cn/v1"
DEEPSEEK = "https://api.deepseek.com"
UNKNOWN = "https://llm.example.com/v1"


@pytest.fixture(autouse=True)
def _default_scope(monkeypatch):
    monkeypatch.setattr(settings, "THINKING_CAPABILITY_SCOPE", "auxiliary")


# ---------------------------------------------------------------- 主机族、规整与匹配


@pytest.mark.parametrize(
    "base_url, family",
    [
        (VERTEX, hosts.VERTEX),
        ("https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi", hosts.VERTEX),
        (GEMINI_API, hosts.GEMINI_API),
        (ZHIPU, hosts.ZHIPU),
        ("https://api.z.ai/api/paas/v4", hosts.ZHIPU),
        (OPENROUTER, hosts.OPENROUTER),
        (DASHSCOPE, hosts.DASHSCOPE),
        (DEEPSEEK, hosts.DEEPSEEK),
        (MOONSHOT, hosts.MOONSHOT),
        (OPENAI, hosts.OPENAI),
        (ANTHROPIC, hosts.ANTHROPIC),
        (BEDROCK, hosts.BEDROCK),
        ("https://bedrock-runtime.us-west-2.amazonaws.com", hosts.BEDROCK),
        ("https://generativelanguage.googleapis.com.example.com/v1", hosts.OTHER),
        ("https://asia-southeast1-aiplatform.googleapis.com.example.com/v1", hosts.OTHER),
        ("https://-aiplatform.googleapis.com/v1", hosts.OTHER),
        ("https://example.com/us-central1-aiplatform.googleapis.com", hosts.OTHER),
        (UNKNOWN, hosts.OTHER),
        ("", hosts.OTHER),
    ],
)
def test_host_family(base_url, family):
    assert hosts.host_family(base_url) == family


@pytest.mark.parametrize(
    "raw, normalized",
    [
        ("  Google/Gemini-2.5-Flash ", "gemini-2.5-flash"),
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
        ("z-ai/GLM-5.3", "glm-5.3"),
        ("global.anthropic.claude-opus-5-5", "claude-opus-5-5"),
        ("us.anthropic.claude-sonnet-4-6-v1:0", "claude-sonnet-4-6-v1:0"),
        ("anthropic.claude-opus-5-5", "claude-opus-5-5"),
        (None, ""),
    ],
)
def test_model_names_are_normalized_before_matching(raw, normalized):
    assert thinking.normalize_model_name(raw) == normalized


@pytest.mark.parametrize(
    "protocol, base_url, model, entry",
    [
        # 最长前缀优先；变体与日期快照后缀算命中。
        (thinking.CHAT, VERTEX, "gemini-2.5-flash-lite", "chat:vertex:gemini-2.5-flash"),
        (thinking.CHAT, GEMINI_API, "models/gemini-2.5-flash-preview-09-2025", "chat:gemini_api:gemini-2.5-flash"),
        (thinking.CHAT, ZHIPU, "glm-5.3-flash", "chat:zhipu:glm-5.3"),
        (thinking.MESSAGES, ANTHROPIC, "claude-haiku-4-5-20251001", "messages:anthropic+bedrock:claude-haiku-4-5"),
        (thinking.MESSAGES, ANTHROPIC, "claude-opus-4-5@20251101", None),
        (thinking.RESPONSES, OPENAI, "gpt-5-2025-08-07", "responses:openai:gpt-5"),
        (thinking.RESPONSES, OPENAI, "gpt-5.1-2025-11-13", "responses:openai:gpt-5.1"),
        (thinking.CHAT, DASHSCOPE, "qwen3.8-max-0902", "chat+responses:dashscope:qwen3.8-max"),
        (thinking.CHAT, DASHSCOPE, "qwen3.7-max-2026-05-17", "chat+responses:dashscope:qwen3.7-max-2026-05-17"),
        # 版本号的延续不算命中：新出的型号不继承旧型号的记录，落到未知模型策略。
        (thinking.MESSAGES, ANTHROPIC, "claude-opus-5-6", None),
        (thinking.RESPONSES, OPENAI, "gpt-5.2", None),
        (thinking.CHAT, MOONSHOT, "kimi-k3.2", None),
        (thinking.CHAT, ZHIPU, "glm-5.21", None),
        # 族默认记录（空前缀）。
        (thinking.CHAT, OPENROUTER, "anthropic/claude-sonnet-4.6", "chat:openrouter:*"),
        # 同一模型换个主机族或协议就不是同一条记录。
        (thinking.CHAT, UNKNOWN, "glm-5.3", None),
        (thinking.RESPONSES, ZHIPU, "glm-5.3", None),
    ],
)
def test_lookup_matches_family_and_longest_prefix_on_a_version_boundary(protocol, base_url, model, entry):
    capability = thinking.lookup(protocol, base_url, model)
    assert (capability.key if capability else None) == entry


# ---------------------------------------------------------------- 表的自检与写法快照


def test_every_record_is_complete_and_sourced():
    seen = set()
    for capability in thinking.CAPABILITIES:
        assert capability.protocols and set(capability.protocols) <= set(thinking.PROTOCOLS), capability.key
        assert capability.families, capability.key
        assert capability.sources and all(str(source).strip() for source in capability.sources), capability.key
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", capability.verified_on), capability.key
        if capability.can_disable:
            assert capability.off_request is not None, capability.key
            assert capability.effort_request is None and capability.lowest_effort is None, capability.key
        else:
            assert capability.off_request is None, capability.key
            # 关不掉：有强度参数就一定有强度名，反之亦然（没有强度参数的不发参数）。
            assert (capability.effort_request is None) == (capability.lowest_effort is None), capability.key
        assert capability.max_output_tokens is None or capability.max_output_tokens > 0, capability.key
        assert capability.model_prefix == thinking.normalize_model_name(capability.model_prefix), capability.key
        for protocol in capability.protocols:
            for family in capability.families:
                key = (protocol, family, capability.model_prefix)
                assert key not in seen, key
                seen.add(key)


#: 能力表请求写法的快照。**改了某条记录的写法，必须同时改它的身份值（换强度名）或升
#: CORE_PROVIDER_PROMPT_VERSION**：运行身份只记开关与强度，不记写法（方案第 6.2 节），否则账本与缓存会把
#: 写法不同的旧判定当作同一请求复用。确认之后再更新这里的哈希；新增记录同样要更新。
WIRE_SNAPSHOT = "78446fda86f186d2"


def _wire_fingerprint():
    rows = sorted(
        [
            capability.key,
            capability.can_disable,
            dict(capability.off_request) if capability.off_request is not None else None,
            capability.lowest_effort,
            dict(capability.effort_request) if capability.effort_request is not None else None,
        ]
        for capability in thinking.CAPABILITIES
    )
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def test_wire_forms_are_frozen_with_their_identity():
    assert _wire_fingerprint() == WIRE_SNAPSHOT, (
        "能力表的请求写法变了：改写法必须同时换身份值（强度名）或升 CORE_PROVIDER_PROMPT_VERSION，"
        "确认后再更新 WIRE_SNAPSHOT（当前为 %s）" % _wire_fingerprint()
    )


# ---------------------------------------------------------------- 解析顺序


def test_chat_resolution_order():
    # 显式配置原样发出。
    explicit = thinking.resolve_chat(ZHIPU, "glm-4.6", "enabled")
    assert (explicit.thinking_on, explicit.identity_type, explicit.fields()) == (True, "enabled", {"thinking": {"type": "enabled"}})
    # 能力表：能关的按表关闭。
    glm52 = thinking.resolve_chat(ZHIPU, "glm-5.2", "")
    assert (glm52.thinking_on, glm52.identity_type, glm52.fields(), glm52.source) == (
        False, "disabled", {"thinking": {"type": "disabled"}}, thinking.SOURCE_TABLE)
    # 关不掉的发最低强度，身份如实记开启。
    glm53 = thinking.resolve_chat(ZHIPU, "glm-5.3", "")
    assert (glm53.thinking_on, glm53.identity_type, glm53.effort, glm53.fields()) == (
        True, thinking.PROVIDER_DEFAULT, "low", {"reasoning_effort": "low"})
    # 族默认记录。
    routed = thinking.resolve_chat(OPENROUTER, "z-ai/glm-5.3", "")
    assert (routed.thinking_on, routed.fields(), routed.source) == (
        False, {"reasoning": {"enabled": False}}, thinking.SOURCE_FAMILY_DEFAULT)
    # 未知模型（Q1）：不发参数，按可能在思考处理。
    unknown = thinking.resolve_chat(UNKNOWN, "house-model", "")
    assert (unknown.thinking_on, unknown.identity_type, unknown.fields(), unknown.source) == (
        True, thinking.PROVIDER_DEFAULT, {}, thinking.SOURCE_UNKNOWN)


def test_explicit_disable_on_a_model_that_cannot_disable_is_sent_as_the_lowest_effort(caplog):
    caplog.set_level(logging.WARNING, logger="backend.app.services.llm.thinking_capabilities")
    plan = thinking.resolve_chat(ZHIPU, "glm-5.3", "disabled")
    # Q4：照发 disabled 厂商一定拒绝；改发最低强度，身份如实记开启。
    assert (plan.thinking_on, plan.fields(), plan.source) == (
        True, {"reasoning_effort": "low"}, thinking.SOURCE_ADJUSTED)
    assert "thinking_explicit_adjusted" in caplog.text and "entry=chat:zhipu:glm-5.3" in caplog.text

    gemini_pro = thinking.resolve_chat(VERTEX, "gemini-2.5-pro", "disabled")
    assert (gemini_pro.thinking_on, gemini_pro.fields()) == (True, {"reasoning_effort": "low"})


def test_explicit_disable_on_google_keeps_the_existing_identity_and_wire():
    plan = thinking.resolve_chat(VERTEX, "google/gemini-2.5-flash", "disabled")
    assert (plan.thinking_on, plan.identity_type) == (False, "none")
    assert plan.fields() == {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}
    # Google 不认 thinking 字段；表外模型不发参数，如实记按模型默认（与改动前相同）。
    unknown = thinking.resolve_chat(GEMINI_API, "gemini-9-experimental", "disabled")
    assert (unknown.thinking_on, unknown.identity_type, unknown.fields()) == (True, thinking.PROVIDER_DEFAULT, {})


@pytest.mark.parametrize(
    "model, thinking_type, effort, expected",
    [
        ("claude-opus-5-5", None, None, (True, thinking.PROVIDER_DEFAULT, "low", {"output_config": {"effort": "low"}})),
        ("claude-sonnet-5-5", None, None, (False, "disabled", None, {"thinking": {"type": "between_tools"}})),
        ("claude-opus-5", None, None, (False, "disabled", None, {"thinking": {"type": "disabled"}})),
        ("claude-opus-4-8", None, None, (False, None, None, {})),
        # 显式配置原样发出；只配了强度时，省略 thinking 即不思考的型号仍记关闭。
        ("claude-opus-5-5", "adaptive", "high", (True, "adaptive", "high", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}})),
        ("claude-opus-4-8", None, "medium", (False, None, "medium", {"output_config": {"effort": "medium"}})),
        ("claude-opus-5-5", None, "medium", (True, thinking.PROVIDER_DEFAULT, "medium", {"output_config": {"effort": "medium"}})),
        # Q4：Opus 5.5 收到 disabled 会 400。
        ("claude-opus-5-5", "disabled", None, (True, thinking.PROVIDER_DEFAULT, "low", {"output_config": {"effort": "low"}})),
        ("claude-3-7-sonnet", None, None, (True, thinking.PROVIDER_DEFAULT, None, {})),
    ],
)
def test_claude_resolution(model, thinking_type, effort, expected):
    plan = thinking.resolve_messages(ANTHROPIC, model, thinking_type, effort)
    assert (plan.thinking_on, plan.identity_type, plan.effort, plan.fields()) == expected


def test_claude_on_bedrock_does_not_send_an_unverified_effort():
    plan = thinking.resolve_messages(BEDROCK, "global.anthropic.claude-opus-5-5")
    assert (plan.thinking_on, plan.effort, plan.fields()) == (True, None, {})


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-5", (True, thinking.PROVIDER_DEFAULT, "minimal", {"reasoning": {"effort": "minimal"}})),
        ("gpt-5.1", (False, "disabled", None, {"reasoning": {"effort": "none"}})),
        # 非推理模型：省略即不思考，身份与改动前相同。
        ("gpt-4.1-mini", (False, None, None, {})),
        # 官方页没写是不是推理模型：按未知模型处理（Q1）。
        ("gpt-4o", (True, thinking.PROVIDER_DEFAULT, None, {})),
    ],
)
def test_responses_resolution(model, expected):
    plan = thinking.resolve_responses(OPENAI, model)
    assert (plan.thinking_on, plan.identity_type, plan.effort, plan.fields()) == expected


@pytest.mark.parametrize(
    "scope, auxiliary_source, scoring_source",
    [
        ("off", thinking.SOURCE_LEGACY, thinking.SOURCE_LEGACY),
        ("auxiliary", thinking.SOURCE_TABLE, thinking.SOURCE_LEGACY),
        ("all", thinking.SOURCE_TABLE, thinking.SOURCE_TABLE),
    ],
)
def test_scope_decides_which_paths_use_the_table(scope, auxiliary_source, scoring_source):
    auxiliary, scoring = thinking.plans(thinking.CHAT, ZHIPU, "glm-5.3", thinking_type="", scope=scope)
    assert (auxiliary.source, scoring.source) == (auxiliary_source, scoring_source)


def test_plan_fields_are_copies():
    plan = thinking.resolve_chat(VERTEX, "gemini-2.5-flash", "")
    plan.fields()["extra_body"]["google"]["thinking_config"]["thinking_budget"] = 99
    again = thinking.resolve_chat(VERTEX, "gemini-2.5-flash", "")
    assert again.fields()["extra_body"]["google"]["thinking_config"]["thinking_budget"] == 0


# ---------------------------------------------------------------- 请求体（AI 任务）


class _Capture:
    def __init__(self, body):
        self.body = body
        self.requests = []

    def client(self, timeout=5.0):
        def respond(request):
            self.requests.append(request)
            return httpx.Response(200, json=self.body)

        return httpx.Client(transport=httpx.MockTransport(respond), timeout=timeout)

    @property
    def payload(self):
        return json.loads(self.requests[-1].content)

    @property
    def read_timeout(self):
        return self.requests[-1].extensions["timeout"]["read"]


def _chat_body(content='{"ok": true}', *, finish_reason="stop", usage=None, message=None):
    return {
        "id": "chat-1",
        "choices": [{"finish_reason": finish_reason, "message": {"content": content, **(message or {})}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _chat(base_url, model, capture, **kwargs):
    kwargs.setdefault("thinking_type", "")
    return OpenAICompatibleChatScorer(
        api_key="test-key", base_url=base_url, model_name=model, client=capture.client(), **kwargs
    )


def test_glm_53_drafting_sends_the_lowest_effort_and_only_limits_the_answer():
    capture = _Capture(_chat_body())
    scorer = _chat(ZHIPU, "glm-5.3", capture)

    scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144, default_timeout_seconds=120, attempts_limit=1)

    payload = capture.payload
    assert payload["reasoning_effort"] == "low"
    assert "thinking" not in payload
    # 6144 是正式输出预算：思考计入上限，所以请求上限放到 min(模型最大输出, 32768)。
    assert payload["max_tokens"] == thinking.THINKING_ON_MAX_TOKENS_CAP
    # 真正的上限变成了时间：思考开着时缺省等待 240 秒。
    assert capture.read_timeout == thinking.THINKING_ON_TIMEOUT_SECONDS


def test_explicit_connection_limits_still_win_for_a_thinking_model():
    capture = _Capture(_chat_body())
    # 连接显式超时：适配器不再按次覆盖，沿用按连接超时建的客户端（生产里由适配器自己建）。
    scorer = OpenAICompatibleChatScorer(api_key="k", base_url=ZHIPU, model_name="glm-5.3", thinking_type="",
                                        max_tokens=4000, timeout_seconds=90, client=capture.client(timeout=90))

    scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144, default_timeout_seconds=120)

    assert capture.payload["max_tokens"] == 4000
    assert capture.read_timeout == 90


def test_glm_52_is_disabled_and_keeps_the_existing_budget():
    capture = _Capture(_chat_body())
    scorer = _chat(ZHIPU, "glm-5.2", capture)

    scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144, default_timeout_seconds=120)

    assert capture.payload["thinking"] == {"type": "disabled"}
    assert capture.payload["max_tokens"] == 6144
    assert capture.read_timeout == 120


def test_vertex_gemini_flash_request_is_unchanged():
    capture = _Capture(_chat_body())
    scorer = _chat(VERTEX, "google/gemini-2.5-flash", capture)

    scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144)

    payload = capture.payload
    assert payload["extra_body"] == {"google": {"thinking_config": {"thinking_budget": 0}}}
    assert "reasoning_effort" not in payload and "thinking" not in payload
    assert payload["max_tokens"] == 6144
    assert scorer.model_version == "chat-completions"


def test_openrouter_disables_reasoning_for_every_auxiliary_call():
    capture = _Capture(_chat_body())
    scorer = _chat(OPENROUTER, "google/gemini-2.5-flash", capture)

    scorer.complete_json("归类", {"x": 1})

    assert capture.payload["reasoning"] == {"enabled": False}


def test_assistant_tool_step_raises_the_limit_only_when_thinking():
    body = {"id": "t", "choices": [{"finish_reason": "stop", "message": {"content": "好的"}}], "usage": {}}
    capture = _Capture(body)
    scorer = _chat(ZHIPU, "glm-5.3", capture)
    scorer.complete_with_tools([{"role": "user", "content": "hi"}], [], max_tokens=1200, deadline=None)
    assert capture.payload["max_tokens"] == thinking.THINKING_ON_MAX_TOKENS_CAP
    assert capture.payload["reasoning_effort"] == "low"

    capture = _Capture(body)
    scorer = _chat(ZHIPU, "glm-5.2", capture)
    scorer.complete_with_tools([{"role": "user", "content": "hi"}], [], max_tokens=1200, deadline=None)
    assert capture.payload["max_tokens"] == 1200


def _message_body(text='{"ok": true}', *, content=None):
    return {
        "id": "msg-1",
        "type": "message",
        "stop_reason": "end_turn",
        "content": content or [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-opus-5-5", {"output_config": {"effort": "low"}}),
        ("claude-sonnet-5-5", {"thinking": {"type": "between_tools"}}),
        ("claude-opus-4-8", {}),
    ],
)
def test_claude_requests_follow_the_table(model, expected):
    capture = _Capture(_message_body())
    scorer = AnthropicMessagesScorer(api_key="k", base_url=ANTHROPIC, model_name=model, client=capture.client(),
                                     structured_output="off")

    scorer.complete_json("指令", {"x": 1}, default_max_tokens=6144)

    payload = capture.payload
    assert {key: payload[key] for key in ("thinking", "output_config") if key in payload} == expected
    expected_max = thinking.THINKING_ON_MAX_TOKENS_CAP if model == "claude-opus-5-5" else max(
        settings.ANTHROPIC_MAX_TOKENS, 6144)
    assert payload["max_tokens"] == expected_max


def test_responses_reasoning_model_sends_the_lowest_effort():
    capture = _Capture({"id": "r", "status": "completed", "output_text": '{"ok": true}', "usage": {}})
    scorer = OpenAIResponsesScorer(api_key="k", base_url=OPENAI, model_name="gpt-5", client=capture.client())

    scorer.complete_json("指令", {"x": 1}, default_max_tokens=6144)

    assert capture.payload["reasoning"] == {"effort": "minimal"}
    assert capture.payload["max_output_tokens"] == thinking.THINKING_ON_MAX_TOKENS_CAP


def test_scope_off_restores_the_previous_auxiliary_request(monkeypatch):
    monkeypatch.setattr(settings, "THINKING_CAPABILITY_SCOPE", "off")
    capture = _Capture(_chat_body())
    scorer = _chat(ZHIPU, "glm-5.3", capture)

    scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144, default_timeout_seconds=120)

    payload = capture.payload
    assert "reasoning_effort" not in payload and "thinking" not in payload
    assert payload["max_tokens"] == 6144
    assert capture.read_timeout == 120


# ---------------------------------------------------------------- 猜的上限被拒时退回


class _Sequence(_Capture):
    """按顺序给出应答：``(状态码, 应答体)``。"""

    def __init__(self, *responses):
        super().__init__(None)
        self.responses = list(responses)

    def client(self, timeout=5.0):
        def respond(request):
            self.requests.append(request)
            status, body = self.responses.pop(0)
            return httpx.Response(status, json=body)

        return httpx.Client(transport=httpx.MockTransport(respond), timeout=timeout)

    def sent(self, field):
        return [json.loads(request.content)[field] for request in self.requests]


_OPENAI_TOO_LARGE = {"error": {
    "message": "max_tokens is too large: 32768. This model supports at most 16384 completion tokens, "
               "whereas you provided 32768.",
    "type": "invalid_request_error", "param": "max_tokens", "code": None,
}}
_VERTEX_OUT_OF_RANGE = [{"error": {
    "code": 400, "status": "INVALID_ARGUMENT",
    "message": "Unable to submit request because it has a maxOutputTokens value of 32768 but the supported "
               "range is from 1 (inclusive) to 8193 (exclusive). Update the value and try again.",
}}]
_CLAUDE_TOO_LARGE = {"type": "error", "error": {
    "type": "invalid_request_error",
    "message": "max_tokens: 32768 > 8192, which is the maximum allowed number of output tokens for "
               "claude-3-5-haiku-20241022",
}}
_RESPONSES_TOO_LARGE = {"error": {
    "message": "Invalid 'max_output_tokens': integer above maximum value. Expected a value <= 16384, "
               "but got 32768 instead.",
    "type": "invalid_request_error", "param": "max_output_tokens", "code": "integer_above_max_value",
}}


@pytest.mark.parametrize("base_url, model, error", [
    (OPENAI, "gpt-4o-mini", _OPENAI_TOO_LARGE),
    (VERTEX, "google/gemini-2.0-flash", _VERTEX_OUT_OF_RANGE),
])
def test_unknown_model_rejecting_the_raised_limit_is_resent_once_with_the_previous_limit(
    caplog, base_url, model, error,
):
    capture = _Sequence((400, error), (200, _chat_body()))
    scorer = _chat(base_url, model, capture)
    assert scorer.thinking_plan.source == thinking.SOURCE_UNKNOWN

    with caplog.at_level(logging.INFO, logger="paper_grading.llm.calls"):
        result = scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144, default_timeout_seconds=120,
                                      attempts_limit=1)

    assert result == {"ok": True}
    # 放开前的取法：max(连接上限, 正式输出预算)，与改动前发的相同；退回不占重试次数。
    assert capture.sent("max_tokens") == [thinking.THINKING_ON_MAX_TOKENS_CAP, max(scorer.max_tokens, 6144)]
    fallback = [record.getMessage() for record in caplog.records
                if record.getMessage().startswith("thinking_output_limit_fallback")]
    assert fallback == [
        "thinking_output_limit_fallback provider=%s model=%s field=max_tokens sent=32768 fallback=6144 "
        "thinking_source=unknown entry=-" % (scorer.provider_name, model)
    ]
    succeeded = [record.getMessage() for record in caplog.records if record.getMessage().startswith("llm_call ")]
    assert succeeded and "max_tokens=6144" in succeeded[-1]


def test_assistant_tool_step_falls_back_to_its_step_budget():
    body = {"id": "t", "choices": [{"finish_reason": "stop", "message": {"content": "好的"}}], "usage": {}}
    capture = _Sequence((400, _OPENAI_TOO_LARGE), (200, body))
    scorer = _chat(OPENAI, "gpt-4o-mini", capture)

    scorer.complete_with_tools([{"role": "user", "content": "hi"}], [], max_tokens=1200, deadline=None)

    assert capture.sent("max_tokens") == [thinking.THINKING_ON_MAX_TOKENS_CAP, 1200]


def test_claude_and_responses_fall_back_on_their_own_limit_field():
    capture = _Sequence((400, _CLAUDE_TOO_LARGE), (200, _message_body()))
    scorer = AnthropicMessagesScorer(api_key="k", base_url=ANTHROPIC, model_name="claude-3-5-haiku-20241022",
                                     client=capture.client(), structured_output="off")
    scorer.complete_json("指令", {"x": 1}, default_max_tokens=6144)
    assert capture.sent("max_tokens") == [thinking.THINKING_ON_MAX_TOKENS_CAP, max(scorer.max_tokens, 6144)]

    capture = _Sequence(
        (400, _RESPONSES_TOO_LARGE),
        (200, {"id": "r", "status": "completed", "output_text": '{"ok": true}', "usage": {}}),
    )
    scorer = OpenAIResponsesScorer(api_key="k", base_url=OPENAI, model_name="gpt-4o-mini", client=capture.client())
    scorer.complete_json("指令", {"x": 1}, default_max_tokens=6144)
    assert capture.sent("max_output_tokens") == [
        thinking.THINKING_ON_MAX_TOKENS_CAP, max(scorer.max_output_tokens, 6144),
    ]


@pytest.mark.parametrize("case", ["known_maximum", "explicit_limit", "other_400", "second_rejection", "scoring"])
def test_no_fallback_outside_a_guessed_limit_rejected_once(monkeypatch, case):
    error = {"error": {"message": "response_format is invalid", "type": "invalid_request_error"}} \
        if case == "other_400" else _OPENAI_TOO_LARGE
    responses = [(400, error), (400, error)] if case == "second_rejection" else [(400, error)]
    capture = _Sequence(*responses)
    kwargs = {"max_tokens": 4000} if case == "explicit_limit" else {}
    model = "glm-5.3" if case == "known_maximum" else "gpt-4o-mini"
    base_url = ZHIPU if case == "known_maximum" else OPENAI
    if case == "scoring":
        monkeypatch.setattr(settings, "THINKING_CAPABILITY_SCOPE", "all")
    scorer = _chat(base_url, model, capture, **kwargs)

    with pytest.raises(Exception):
        scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144, attempts_limit=1,
                             scoring=case == "scoring")

    expected_requests = 2 if case == "second_rejection" else 1
    assert len(capture.requests) == expected_requests


# ---------------------------------------------------------------- 运行身份


@pytest.mark.parametrize(
    "factory, thinking_identity, model_version",
    [
        # 评分路径默认旧行为（auxiliary）：与改动前逐字节相同。
        (lambda capture: _chat(ZHIPU, "glm-5.3", capture), {"enabled": False, "type": None}, "chat-completions"),
        (lambda capture: _chat(VERTEX, "gemini-2.5-pro", capture),
         {"enabled": True, "type": thinking.PROVIDER_DEFAULT}, "chat-completions"),
        (lambda capture: _chat(VERTEX, "gemini-2.5-flash", capture), {"enabled": False, "type": "none"}, "chat-completions"),
        (lambda capture: AnthropicMessagesScorer(api_key="k", base_url=ANTHROPIC, model_name="claude-opus-5-5",
                                                 client=capture.client()),
         {"enabled": False, "type": None}, "messages-2023-06-01"),
        (lambda capture: OpenAIResponsesScorer(api_key="k", base_url=OPENAI, model_name="gpt-5",
                                               client=capture.client()),
         {"enabled": False, "type": None}, "responses-api"),
    ],
)
def test_scoring_identity_is_unchanged_until_scope_all(factory, thinking_identity, model_version):
    scorer = factory(_Capture({}))
    contract = core_runtime_provider_contract(scorer, artifact_hash="a" * 64)
    assert contract["thinking"] == thinking_identity
    assert contract["model_version"] == model_version


@pytest.mark.parametrize(
    "factory, thinking_identity, model_version, max_tokens",
    [
        (lambda capture: _chat(ZHIPU, "glm-5.3", capture), {"enabled": True, "type": thinking.PROVIDER_DEFAULT},
         "chat-completions;effort=low;max_tokens=32768", 2048),
        (lambda capture: _chat(ZHIPU, "glm-5.2", capture), {"enabled": False, "type": "disabled"},
         "chat-completions", 2048),
        (lambda capture: _chat(VERTEX, "gemini-2.5-flash", capture), {"enabled": False, "type": "none"},
         "chat-completions", 2048),
        (lambda capture: _chat(ZHIPU, "glm-5.3", capture, max_tokens=4000),
         {"enabled": True, "type": thinking.PROVIDER_DEFAULT}, "chat-completions;effort=low", 4000),
        (lambda capture: AnthropicMessagesScorer(api_key="k", base_url=ANTHROPIC, model_name="claude-opus-5-5",
                                                 client=capture.client()),
         {"enabled": True, "type": thinking.PROVIDER_DEFAULT}, "messages-2023-06-01;effort=low;max_tokens=32768",
         settings.ANTHROPIC_MAX_TOKENS),
        (lambda capture: AnthropicMessagesScorer(api_key="k", base_url=ANTHROPIC, model_name="claude-sonnet-5-5",
                                                 client=capture.client()),
         {"enabled": False, "type": "disabled"}, "messages-2023-06-01", settings.ANTHROPIC_MAX_TOKENS),
        (lambda capture: OpenAIResponsesScorer(api_key="k", base_url=OPENAI, model_name="gpt-5",
                                               client=capture.client()),
         {"enabled": True, "type": thinking.PROVIDER_DEFAULT}, "responses-api;effort=minimal;max_tokens=32768",
         settings.OPENAI_MAX_OUTPUT_TOKENS),
    ],
)
def test_scope_all_identity_records_switch_effort_and_raised_limit(monkeypatch, factory, thinking_identity,
                                                                   model_version, max_tokens):
    monkeypatch.setattr(settings, "THINKING_CAPABILITY_SCOPE", "all")
    scorer = factory(_Capture({}))
    contract = core_runtime_provider_contract(scorer, artifact_hash="a" * 64)
    assert contract["thinking"] == thinking_identity
    assert contract["model_version"] == model_version
    # 信封记正式输出预算：Core 证据预算按它扣，不会因为放开的请求上限变成负数。
    assert contract["sampling"]["max_tokens"] == max_tokens
    # 封闭契约照样接受（enabled 与 type 一致）。
    _normalize_provider({key: value for key, value in contract.items() if key != "artifact_hash"})


def test_scope_all_core_request_sends_the_plan_and_the_raised_limit(monkeypatch):
    from backend.app.tests.test_core_llm_adapters import _core_envelope
    from backend.app.tests.test_core_llm_adapters import _provider_response

    monkeypatch.setattr(settings, "THINKING_CAPABILITY_SCOPE", "all")
    capture = _Capture({})
    scorer = _chat(ZHIPU, "glm-5.3", capture)
    envelope = _core_envelope(scorer)
    capture.body = _chat_body(json.dumps(_provider_response(envelope), ensure_ascii=False))

    scorer.score_core_envelope(envelope=envelope)

    payload = capture.payload
    assert payload["reasoning_effort"] == "low"
    assert payload["max_tokens"] == thinking.THINKING_ON_MAX_TOKENS_CAP
    assert envelope.to_mapping()["runtime_identity"]["provider"]["sampling"]["max_tokens"] == 2048


def test_scope_all_lets_responses_reasoning_models_into_core(monkeypatch):
    from backend.app.tests.test_core_llm_adapters import _core_envelope
    from backend.app.tests.test_core_llm_adapters import _provider_response

    monkeypatch.setattr(settings, "THINKING_CAPABILITY_SCOPE", "all")
    capture = _Capture({})
    scorer = OpenAIResponsesScorer(api_key="k", base_url=OPENAI, model_name="gpt-5", client=capture.client())
    envelope = _core_envelope(scorer)
    capture.body = {"id": "r", "status": "completed", "usage": {},
                    "output_text": json.dumps(_provider_response(envelope), ensure_ascii=False)}

    scorer.score_core_envelope(envelope=envelope)

    assert capture.payload["reasoning"] == {"effort": "minimal"}
    assert capture.payload["max_output_tokens"] == thinking.THINKING_ON_MAX_TOKENS_CAP


def test_legacy_cache_key_includes_scoring_thinking_and_answer_budget():
    from backend.app.services.cache.llm_cache import build_request

    class Criterion:
        id = code = name = "C1"
        max_score = 10

    scorer = _chat(ZHIPU, "glm-5.2", _Capture({}))
    request = build_request(scorer, Criterion(), [], {}, "rubric@1")
    assert request["thinking"] == {"enabled": False, "type": None, "request_fields": {}}
    assert request["max_tokens"] == 2048


def test_legacy_cache_key_changes_when_only_the_wire_form_changes(monkeypatch):
    """只改能力表的写法、身份不变时，旧兼容路径的 L0 缓存键也要变（不靠记得升版本号）。"""

    from backend.app.services.cache.llm_cache import build_request

    class Criterion:
        id = code = name = "C1"
        max_score = 10

    monkeypatch.setattr(settings, "THINKING_CAPABILITY_SCOPE", "all")
    scorer = _chat(ZHIPU, "glm-5.2", _Capture({}))
    before = build_request(scorer, Criterion(), [], {}, "rubric@1")
    assert before["thinking"]["request_fields"] == {"thinking": {"type": "disabled"}}

    plan = scorer.scoring_thinking_plan
    scorer.scoring_thinking_plan = thinking.ThinkingPlan(
        plan.thinking_on, plan.identity_type, plan.effort, {"reasoning_effort": "none"}, plan.source, plan.entry,
        plan.model_max_output_tokens,
    )
    after = build_request(scorer, Criterion(), [], {}, "rubric@1")
    assert after["thinking"]["enabled"] == before["thinking"]["enabled"]
    assert after["thinking"]["type"] == before["thinking"]["type"]
    assert after["model_version"] == before["model_version"]
    assert after != before


# ---------------------------------------------------------------- 观测：正式输出与误标


@pytest.mark.parametrize(
    "kwargs, answer",
    [
        # OpenAI / OpenRouter：推理算在输出里。
        ({"prompt": 100, "completion": 300, "total": 400, "reported_reasoning": 250}, (50, "reported")),
        # Vertex：推理不在输出里（合计 > 输入 + 输出）。
        ({"prompt": 100, "completion": 60, "total": 410, "reported_reasoning": 250}, (60, "reported")),
        # 没有思考迹象。
        ({"prompt": 100, "completion": 60, "total": 160}, (60, "reported")),
        # 智谱 / Claude：只有思考文本或思考块，正式输出按可见文字估算（UTF-8 字节 ÷ 3）。
        ({"prompt": 100, "completion": 900, "total": 1000, "thinking_seen": True, "answer_text": "评分规则" * 3},
         (12, "estimated")),
        ({"prompt": 100, "completion": None, "total": None, "thinking_seen": True}, (None, None)),
    ],
)
def test_answer_tokens_are_recorded_after_the_fact(kwargs, answer):
    usage = observed_usage(**kwargs)
    assert (usage["answer_tokens"], usage["answer_source"]) == answer


def test_call_log_records_the_plan_and_the_answer(caplog):
    caplog.set_level(logging.INFO, logger="paper_grading.llm.calls")
    usage = {"prompt_tokens": 50, "completion_tokens": 900, "total_tokens": 950}
    capture = _Capture(_chat_body(usage=usage, message={"reasoning_content": "想了很多"}))
    scorer = _chat(ZHIPU, "glm-5.3", capture)

    scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144)

    line = next(record.getMessage() for record in caplog.records if record.getMessage().startswith("llm_call "))
    assert "thinking=on effort=low thinking_source=table entry=chat:zhipu:glm-5.3" in line
    assert "answer_budget=6144 max_tokens=32768" in line
    assert "answer=estimated" in line
    assert "想了很多" not in line


def test_table_mislabel_is_logged_for_auxiliary_calls_and_stopped_by_the_guard(caplog):
    caplog.set_level(logging.WARNING, logger="paper_grading.llm.calls")
    usage = {"prompt_tokens": 50, "completion_tokens": 300, "total_tokens": 350,
             "completion_tokens_details": {"reasoning_tokens": 250}}
    capture = _Capture(_chat_body(usage=usage))
    scorer = _chat(ZHIPU, "glm-5.2", capture)

    scorer.complete_json("起草", {"x": 1}, default_max_tokens=6144)
    assert "thinking_capability_mismatch" in caplog.text
    assert "entry=chat:zhipu:glm-5.2" in caplog.text

    # 评分路径：能力表把模型误标成可关闭时，身份写关闭、模型在思考，一致性守卫兜住。
    guard = ThinkingConsistencyGuard()
    with pytest.raises(ThinkingNotDisabledError):
        metered_call(scorer, lambda: scorer.complete_json("x", {}), guard=guard,
                     thinking={"enabled": False, "type": "disabled"})


def test_explicit_or_unknown_plans_do_not_log_a_table_mismatch(caplog):
    caplog.set_level(logging.WARNING, logger="paper_grading.llm.calls")
    usage = {"prompt_tokens": 50, "completion_tokens": 300, "total_tokens": 350,
             "completion_tokens_details": {"reasoning_tokens": 250}}
    capture = _Capture(_chat_body(usage=usage))
    scorer = _chat(UNKNOWN, "house-model", capture, thinking_type="disabled")

    scorer.complete_json("起草", {"x": 1})

    assert "thinking_capability_mismatch" not in caplog.text


# ---------------------------------------------------------------- 起草器


def _draft_scorer(capture, model="glm-5.3", **kwargs):
    return _chat(ZHIPU, model, capture, **kwargs)


def _draft_inputs():
    from backend.app.services.rubric_import.compiler import analyze_rule_input

    criterion = {"code": "T02", "name": "需求分析", "max_score": 10, "description": "需求完整",
                 "evidence_hints": ["需求规格说明"]}
    return criterion, analyze_rule_input(["需求边界表达不清"], criterion_code="T02")


def _draft_output():
    return {"rule_groups": [{"group_code": "G1", "issue": "需求边界不清", "mutex_group": "G1-S", "cap_points": 4,
                             "rules": [{"severity": "minor", "trigger": "边界不清", "points": 2, "reason": "r",
                                        "repeat_policy": "once", "source": "ai_interpreted_user_text",
                                        "source_refs": ["/criteria/T02/deduction_rules/0"]}]}]}


def test_drafter_asks_the_adapter_for_the_request_limit(caplog):
    from backend.app.services.rubric_import.ai_rule_drafter import draft_deduction_rules

    caplog.set_level(logging.INFO, logger="backend.app.services.rubric_import.ai_rule_drafter")
    capture = _Capture(_chat_body(json.dumps(_draft_output(), ensure_ascii=False)))
    criterion, analysis = _draft_inputs()

    draft = draft_deduction_rules(criterion=criterion, input_analysis=analysis, scorer=_draft_scorer(capture),
                                  business_profile_key="thesis")

    assert draft["rule_groups"]
    payload = capture.payload
    assert payload["reasoning_effort"] == "low"
    assert payload["max_tokens"] == thinking.THINKING_ON_MAX_TOKENS_CAP
    assert "answer_budget=6144 thinking=on effort=low" in caplog.text


def test_drafter_truncation_and_timeout_explain_a_model_that_cannot_stop_thinking():
    from backend.app.services.rubric_import.ai_rule_drafter import AIRuleDraftValidationError
    from backend.app.services.rubric_import.ai_rule_drafter import draft_deduction_rules

    criterion, analysis = _draft_inputs()
    truncated = _Capture(_chat_body("", finish_reason="length"))
    with pytest.raises(AIRuleDraftValidationError) as exc:
        draft_deduction_rules(criterion=criterion, input_analysis=analysis, scorer=_draft_scorer(truncated),
                              business_profile_key="thesis")
    assert exc.value.code == "AI_DRAFT_OUTPUT_TRUNCATED"
    assert "关不掉思考" in exc.value.user_action and "low" in exc.value.user_action

    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    # 换一个同样命中 glm-5.3 记录的型号名：超时会计入这条连接的熔断，不影响其它测试。
    scorer = OpenAICompatibleChatScorer(api_key="k", base_url=ZHIPU, model_name="glm-5.3-flashx-timeout",
                                        thinking_type="", client=httpx.Client(transport=httpx.MockTransport(timeout)))
    with pytest.raises(AIRuleDraftValidationError) as exc:
        draft_deduction_rules(criterion=criterion, input_analysis=analysis, scorer=scorer,
                              business_profile_key="thesis")
    assert "240 秒" in exc.value.message
    assert "关不掉思考" in exc.value.user_action


def test_semantic_coherence_uses_the_scoring_plan(monkeypatch):
    from backend.app.services.coherence.semantic import analyze_semantic_coherence

    monkeypatch.setattr(settings, "COHERENCE_SEMANTIC_ENABLED", True)
    capture = _Capture(_chat_body('{"research_questions": [], "conclusion_claims": []}'))
    scorer = _chat(ZHIPU, "glm-5.3", capture)
    parsed = {"sections": [{"title": "绪论", "paragraphs": [{"text": "研究问题"}]},
                           {"title": "结论", "paragraphs": [{"text": "结论"}]}],
              "full_text": "正文"}
    analyze_semantic_coherence(parsed, scorer)
    # 评分路径在 scope=auxiliary 时是旧行为：不发强度、上限不放开。
    assert "reasoning_effort" not in capture.payload
    assert capture.payload["max_tokens"] == 2048

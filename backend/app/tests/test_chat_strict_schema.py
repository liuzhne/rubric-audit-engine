"""Chat 协议的 Core 评分请求带严格 Schema（docs/Chat评分请求严格Schema方案.md）。

全部用假的 HTTP 客户端，不调用真实模型。
"""

from copy import deepcopy

import pytest

from backend.app.core.config import settings
from backend.app.services import ai_connections
from backend.app.services.ai_connections import ConnectionRuntime
from backend.app.services.ai_connections import validate_provider_options_for
from backend.app.services.llm import schema_dialects
from backend.app.services.llm.core_view import build_core_request
from backend.app.services.llm.openai_compatible_adapter import OpenAICompatibleChatScorer
from backend.app.services.scoring.core.failures import project_rule_execution_failure
from backend.app.tests import test_core_llm_adapters as adapter_tests


VERTEX = "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"
OTHER = "https://provider.example/v1"


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_CHAT_STRICT_SCHEMA_ENABLED", True)


class _Client(adapter_tests._CompatibleCoreClient):
    """与共用的假客户端相同，但不限定鉴权头（Vertex 用 x-goog-api-key）。"""

    def post(self, url, headers, json):
        headers = {**headers, "Authorization": "Bearer test-key"}
        return super().post(url, headers, json)


class _Rejected:
    status_code = 400
    headers = {"content-type": "application/json"}
    text = '{"error":{"type":"invalid_request"}}'

    def json(self):
        return {"error": {"type": "invalid_request"}}

    def raise_for_status(self):
        import httpx

        request = httpx.Request("POST", OTHER + "/chat/completions")
        raise httpx.HTTPStatusError("400", request=request, response=httpx.Response(400, request=request))


def _scorer(base_url=OTHER, **options):
    client = _Client()
    options.setdefault("response_format_json", False)
    scorer = OpenAICompatibleChatScorer(
        api_key="test-key", base_url=base_url, model_name="schema-test",
        client=client, temperature=0, max_tokens=700, thinking_type="", **options,
    )
    return scorer, client


def _score(scorer, client, **kwargs):
    envelope = adapter_tests._core_envelope(scorer)
    client.output = adapter_tests._provider_response(envelope)
    scorer.score_core_envelope(envelope=envelope, **kwargs)
    return envelope, deepcopy(client.payload)


# ---------------------------------------------------------------- 开关关闭：什么都不变


@pytest.mark.parametrize("legacy_json", [False, True])
def test_with_the_switch_off_the_option_is_stored_but_changes_nothing(legacy_json):
    plain, plain_client = _scorer(response_format_json=legacy_json)
    opted, opted_client = _scorer(response_format_json=legacy_json, structured_output="json_schema")

    plain_envelope, plain_payload = _score(plain, plain_client)
    opted_envelope, opted_payload = _score(opted, opted_client)

    assert opted_payload == plain_payload
    assert opted.model_version == plain.model_version == "chat-completions"
    assert opted_envelope.to_mapping() == plain_envelope.to_mapping()
    expected = {"type": "json_object"} if legacy_json else None
    assert plain_payload.get("response_format") == expected


# ---------------------------------------------------------------- 输出约束的解析


@pytest.mark.parametrize(
    "structured_output, legacy_json, expected",
    [
        ("json_schema", False, "json_schema"),
        ("json_schema", True, "json_schema"),
        ("json_object", False, "json_object"),
        ("off", True, "none"),
        (None, True, "json_object"),
        (None, False, "none"),
    ],
)
def test_an_explicit_option_wins_then_the_old_flag(enabled, structured_output, legacy_json, expected):
    assert schema_dialects.resolve_chat_response_format(
        OTHER, structured_output=structured_output, response_format_json=legacy_json,
    ) == expected


def test_no_host_family_sends_a_schema_by_default_until_it_is_verified(enabled, monkeypatch):
    assert schema_dialects.VERIFIED_STRICT_SCHEMA_FAMILIES == frozenset()
    assert schema_dialects.resolve_chat_response_format(VERTEX) == "none"
    monkeypatch.setattr(schema_dialects, "VERIFIED_STRICT_SCHEMA_FAMILIES", frozenset({"vertex"}))
    assert schema_dialects.resolve_chat_response_format(VERTEX) == "json_schema"
    assert schema_dialects.resolve_chat_response_format(VERTEX, structured_output="off") == "none"
    assert schema_dialects.resolve_chat_response_format(OTHER) == "none"


# ---------------------------------------------------------------- 请求与身份


def test_the_rule_schema_is_sent_and_the_identity_says_so(enabled):
    scorer, client = _scorer(structured_output="json_schema")
    envelope, payload = _score(scorer, client)

    schema = build_core_request(envelope).schema
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "semantic_rule_response", "strict": True, "schema": schema},
    }
    provider = envelope.to_mapping()["runtime_identity"]["provider"]
    assert provider["response_format"] == "json_schema"
    assert provider["model_version"] == "chat-completions;schema=plain@1"


def test_the_prompt_is_identical_with_and_without_the_schema(enabled):
    plain, plain_client = _scorer()
    strict, strict_client = _scorer(structured_output="json_schema")
    _envelope, without = _score(plain, plain_client)
    _envelope, with_schema = _score(strict, strict_client)

    assert with_schema["messages"] == without["messages"]
    assert {k: v for k, v in with_schema.items() if k != "response_format"} == without


def test_a_repair_request_carries_the_same_schema(enabled):
    scorer, client = _scorer(structured_output="json_schema")
    _envelope, first = _score(scorer, client)
    _envelope, second = _score(scorer, client, repair="fields")

    assert second["response_format"] == first["response_format"]
    assert len(second["messages"]) == 3


def test_an_envelope_made_without_the_schema_is_refused_by_a_strict_scorer(enabled):
    plain, _client = _scorer()
    strict, strict_client = _scorer(structured_output="json_schema")
    envelope = adapter_tests._core_envelope(plain)

    with pytest.raises(ValueError):
        strict.score_core_envelope(envelope=envelope)
    assert strict_client.payload is None


def test_a_rejected_schema_is_reported_and_not_resent_without_it(enabled):
    scorer, client = _scorer(structured_output="json_schema")
    envelope = adapter_tests._core_envelope(scorer)
    calls = []

    def reject(url, **kwargs):
        calls.append(kwargs["json"])
        return _Rejected()

    client.post = reject
    with pytest.raises(Exception) as caught:
        scorer.score_core_envelope(envelope=envelope)

    assert len(calls) == 1 and calls[0]["response_format"]["type"] == "json_schema"
    assert project_rule_execution_failure(caught.value)[0] == "PROVIDER_INVALID_REQUEST"


# ---------------------------------------------------------------- 方言


def test_the_google_dialect_rewrites_what_vertex_does_not_accept(enabled, monkeypatch):
    monkeypatch.setattr(settings, "SCORING_CHAT_SCHEMA_ENUM_LIMIT", 3)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["level_code", "quote", "ref", "none"],
        "properties": {
            "level_code": {"type": ["string", "null"], "enum": ["A", "B", None]},
            "quote": {"type": "string", "minLength": 1},
            "ref": {"type": "string", "enum": ["e1", "e2", "e3", "e4"]},
            "small": {"type": "string", "enum": ["e1", "e2", "e3"]},
            "none": {
                "type": "array", "maxItems": 0,
                "items": {"type": "object", "additionalProperties": False, "properties": {}, "required": []},
            },
            "nested": {"type": "array", "items": {"type": "object", "properties": {
                "quote": {"type": "string", "minLength": 1},
            }}},
        },
    }
    original = deepcopy(schema)

    rewritten = schema_dialects.apply(schema_dialects.GOOGLE, schema)

    assert schema == original
    properties = rewritten["properties"]
    assert properties["level_code"] == {"type": "string", "nullable": True, "enum": ["A", "B"]}
    assert properties["quote"] == {"type": "string"}
    assert properties["ref"] == {"type": "string"}
    assert properties["small"] == {"type": "string", "enum": ["e1", "e2", "e3"]}
    assert properties["none"] == {"type": "array", "maxItems": 0, "items": {"type": "string"}}
    assert properties["nested"]["items"]["properties"]["quote"] == {"type": "string"}
    assert rewritten["additionalProperties"] is False and rewritten["required"] == original["required"]


def test_a_nullable_field_whose_only_choice_is_null_loses_its_enum():
    rewritten = schema_dialects.apply(
        schema_dialects.GOOGLE, {"type": ["string", "null"], "enum": [None]},
    )
    assert rewritten == {"type": "string", "nullable": True}


def test_other_hosts_get_the_schema_unchanged_but_copied():
    schema = {"type": ["string", "null"], "enum": ["A", None], "minLength": 1}
    rewritten = schema_dialects.apply(schema_dialects.PLAIN, schema)
    assert rewritten == schema and rewritten is not schema


def test_a_google_host_sends_the_rewritten_schema_and_names_the_dialect(enabled):
    scorer, client = _scorer(VERTEX, structured_output="json_schema")
    envelope, payload = _score(scorer, client)

    sent = payload["response_format"]["json_schema"]["schema"]
    assert sent == schema_dialects.apply(schema_dialects.GOOGLE, build_core_request(envelope).schema)
    assert sent["properties"]["level_code"].get("nullable") is True
    assert "minLength" not in str(sent) and "null" not in str(sent["properties"]["level_code"].get("type"))
    assert ";schema=google@1/e%d" % settings.SCORING_CHAT_SCHEMA_ENUM_LIMIT in scorer.model_version


def test_the_enum_limit_is_part_of_the_identity(enabled, monkeypatch):
    first, _client = _scorer(VERTEX, structured_output="json_schema")
    monkeypatch.setattr(settings, "SCORING_CHAT_SCHEMA_ENUM_LIMIT", 8)
    second, _client = _scorer(VERTEX, structured_output="json_schema")
    assert first.model_version != second.model_version


def test_the_local_checks_still_accept_a_valid_answer_under_the_rewritten_schema(enabled):
    scorer, client = _scorer(VERTEX, structured_output="json_schema")
    envelope = adapter_tests._core_envelope(scorer)
    client.output = adapter_tests._provider_response(envelope)

    assert scorer.score_core_envelope(envelope=envelope) is not None


# ---------------------------------------------------------------- 连接选项与测试连接


def test_connection_options_accept_the_mode_per_protocol():
    for mode in ("json_schema", "json_object", "off"):
        assert validate_provider_options_for(
            "openai_compatible", {"structured_output": mode},
        )["structured_output"] == mode
    assert validate_provider_options_for("anthropic_messages", {"structured_output": "off"})
    with pytest.raises(ValueError, match="Claude"):
        validate_provider_options_for("anthropic_messages", {"structured_output": "json_object"})
    with pytest.raises(ValueError, match="Responses"):
        validate_provider_options_for("openai_responses", {"structured_output": "json_schema"})
    with pytest.raises(ValueError):
        validate_provider_options_for("openai_compatible", {"structured_output": "strict"})
    with pytest.raises(ValueError, match="Claude"):
        validate_provider_options_for("openai_compatible", {"effort": "low"})


def _runtime(base_url=OTHER, **options):
    return ConnectionRuntime(
        connection_id="draft", key_version=0, organization_id="org", provider_type="openai_compatible",
        base_url=base_url, model_name="schema-test", provider_options=options, api_key="test-key",
    )


def test_the_connection_test_is_unchanged_unless_scoring_would_send_a_schema(enabled):
    assert ai_connections._chat_schema_probe_fields(_runtime(), OTHER) == {}
    assert ai_connections._chat_schema_probe_fields(_runtime(response_format_json=True), OTHER) == {}


def test_the_connection_test_is_unchanged_with_the_switch_off():
    runtime = _runtime(structured_output="json_schema")
    assert ai_connections._chat_schema_probe_fields(runtime, OTHER) == {}


def test_the_connection_test_sends_a_small_schema_in_the_hosts_dialect(enabled):
    fields = ai_connections._chat_schema_probe_fields(_runtime(VERTEX, structured_output="json_schema"), VERTEX)

    schema = fields["response_format"]["json_schema"]["schema"]
    assert fields["response_format"]["json_schema"]["strict"] is True
    assert schema["properties"]["level_code"] == {"type": "string", "nullable": True, "enum": ["A"]}
    assert fields["max_tokens"] >= 32 and len(fields["messages"]) == 1


# ---------------------------------------------------------------- 实测脚本（不联网的部分）


def test_the_probe_script_is_dry_by_default_and_stays_within_its_call_cap(monkeypatch, capsys):
    from backend.app.scripts import probe_chat_schema as probe

    monkeypatch.setenv("CHAT_SCHEMA_PROBE_BASE_URL", VERTEX)
    monkeypatch.setenv("CHAT_SCHEMA_PROBE_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setattr(probe.httpx, "Client", lambda *args, **kwargs: pytest.fail("must not connect"))
    monkeypatch.setattr(settings, "SCORING_CHAT_SCHEMA_ENUM_LIMIT", 64)  # 脚本会改它，测试后还原

    assert probe.main([]) == 0
    assert "不发送" in capsys.readouterr().out
    assert probe.main(["--sizes", "1,2,3,4,5,6,7,8,9", "--send"]) == 2


def test_the_probe_schema_has_the_shape_of_a_rule_schema_and_judges_answers():
    from backend.app.scripts import probe_chat_schema as probe

    schema = probe.rule_schema(4)
    assert schema["properties"]["level_code"]["type"] == ["string", "null"]
    assert len(schema["properties"]["occurrences"]["items"]["properties"]["evidence"]["items"]
               ["properties"]["evidence_ref"]["enum"]) == 4
    good = {"schema_version": "semantic-rule-response@2", "rule_code": "probe.v1",
            "status": "not_triggered", "level_code": None, "occurrences": []}
    assert probe.conforms(good)
    assert not probe.conforms({**good, "status": "maybe"})
    assert not probe.conforms({**good, "note": "x"})

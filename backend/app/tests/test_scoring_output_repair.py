"""模型回答不合格时的重问与报错（docs/评分模型回答不合格的重问与报错方案.md）。

全部使用本地夹具与假运行时，不调用任何真实模型。
"""

from copy import deepcopy
import json
import logging
import re
from types import SimpleNamespace

import pytest

from backend.app.core.config import settings
from backend.app.services.assistant import views as assistant_views
from backend.app.services.batch_scoring import jobs
from backend.app.services.llm.core_adapter import OUTPUT_REPAIR_HINTS
from backend.app.services.llm.core_adapter import output_repair_hint
from backend.app.services.llm.core_view import build_core_request
from backend.app.services.llm.errors import CoreResponseInvalidError
from backend.app.services.llm.errors import ProviderJSONOutputError
from backend.app.services.llm.openai_adapter import OpenAIResponsesScorer
from backend.app.services.llm.openai_compatible_adapter import OpenAICompatibleChatScorer
from backend.app.services.scoring.core.failures import OUTPUT_INVALID_CHECKS
from backend.app.services.scoring.core.failures import failure_site
from backend.app.services.scoring.core.failures import project_rule_execution_failure
from backend.app.services.scoring.core.rule_executor import execute_rule_plan
from backend.app.services.scoring.decision_ledger import RuleCallJournal
from backend.app.services.scoring.profiles.technical_proposal import TechnicalProposalLLMRuntime
from backend.app.services.scoring.profiles.thesis import ThesisLLMRuntime
from backend.app.services.llm.mock import MockLLMScorer
from backend.app.tests import test_anthropic_messages_adapter as messages_tests
from backend.app.tests import test_core_llm_adapters as adapter_tests
from backend.app.tests import test_m4_rule_executor as m4
from backend.app.tests import test_token_compression_pipeline as groups

MARKER = "学生原文-不得回显-7Q2Z"


# ---------------------------------------------------------------- 夹具


class _ScriptedRuntime:
    """按脚本逐次作答的假运行时：脚本项是回答，或要抛出的异常。"""

    output_repair_limit = 5

    def __init__(self, script, *, limit=5):
        self.script = {code: list(items) for code, items in script.items()}
        self.output_repair_limit = limit
        self.calls = []
        self.last_call_usage = None

    def _next(self, code, repair):
        self.calls.append((code, repair))
        self.last_call_usage = {"request_count": 1, "prompt_tokens": 100, "completion_tokens": 5}
        item = self.script[code].pop(0)
        if isinstance(item, Exception):
            raise item
        return deepcopy(item)

    def score(self, *, envelope, repair=None):
        return self._next(envelope.to_mapping()["atomic_rule_snapshot"]["rule_code"], repair)


class _GroupScriptedRuntime(_ScriptedRuntime):
    supports_group_calls = True

    def score_group(self, *, envelopes, group_code, repair=None):
        return self._next(group_code, repair)


def _request(*codes):
    criterion = m4._criterion()
    return m4._request_for(*((criterion, m4._deduct_rule(code=code)) for code in codes))


def _execute(request, runtime, *, ledger=None):
    journal = RuleCallJournal()
    result = m4._result_mapping(
        execute_rule_plan(
            request=deepcopy(request),
            checker_registry=m4._CheckerRegistry(),
            llm_runtime=runtime,
            profile=m4._TechnicalProposalProfile(),
            decision_ledger=ledger,
            execution_journal=journal,
        )
    )
    return result, journal


def _issue_codes(result):
    return {
        issue["rule_code"]: issue["code"]
        for issue in result["review_issues"]
        if issue.get("rule_code") and issue["severity"] == "block"
    }


def _ok(code):
    return m4._semantic_response(code, status="not_triggered")


def _invalid(check="fields"):
    return CoreResponseInvalidError(check, "Core provider response has invalid fields " + MARKER)


# ---------------------------------------------------------------- 单条规则


def test_an_invalid_answer_is_asked_again_once_and_the_repaired_answer_counts():
    request = _request("R1")
    runtime = _ScriptedRuntime({"R1": [_invalid("fields"), _ok("R1")]})
    ledger = groups._MemoryLedger()

    result, journal = _execute(request, runtime, ledger=ledger)

    assert runtime.calls == [("R1", None), ("R1", "fields")]
    assert m4._decision(result, "R1")["status"] == "not_triggered"
    assert _issue_codes(result) == {}
    assert journal.provider_attempts == {"R1": 2}
    assert len(ledger.rows) == 1


def test_two_invalid_answers_fail_the_rule_as_a_model_output_problem(caplog):
    request = _request("R1")
    runtime = _ScriptedRuntime({"R1": [_invalid("fields"), _invalid("status")]})

    with caplog.at_level(logging.WARNING):
        result, journal = _execute(request, runtime, ledger=groups._MemoryLedger())

    assert runtime.calls == [("R1", None), ("R1", "fields")]
    assert m4._decision(result, "R1")["status"] == "invalid"
    issue = next(item for item in result["review_issues"] if item.get("rule_code") == "R1")
    assert issue["code"] == "PROVIDER_OUTPUT_INVALID"
    assert issue["message"] == "semantic provider output failed validation (check=status)"
    assert journal.provider_attempts == {"R1": 2}
    assert "rule_output_repair rule_code=R1 check=fields outcome=rejected" in caplog.text
    assert "error_code=PROVIDER_OUTPUT_INVALID" in caplog.text and "check=status" in caplog.text
    # 异常消息（可能拼进了内容）不进日志，也不进复核说明。
    assert MARKER not in caplog.text and MARKER not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize(
    "reason, check",
    [("invalid_json", "json"), ("empty_content", "json")],
)
def test_an_unusable_json_answer_is_also_asked_again(reason, check):
    request = _request("R1")
    runtime = _ScriptedRuntime({"R1": [ProviderJSONOutputError(reason), _ok("R1")]})

    result, _journal = _execute(request, runtime)

    assert runtime.calls == [("R1", None), ("R1", check)]
    assert _issue_codes(result) == {}


def test_answers_rejected_by_the_executor_are_asked_again_and_keep_their_code():
    request = _request("R1")
    unknown_finding = m4._semantic_response(
        "R1", occurrences=[m4._quote_occurrence(request, finding_code="NOT_PUBLISHED")]
    )
    repaired = _ScriptedRuntime({"R1": [unknown_finding, _ok("R1")]})

    result, journal = _execute(request, repaired)

    assert repaired.calls == [("R1", None), ("R1", "finding_code")]
    assert _issue_codes(result) == {} and journal.provider_attempts == {"R1": 2}

    still_bad = _ScriptedRuntime({"R1": [unknown_finding, unknown_finding]})
    result, _journal = _execute(request, still_bad)

    assert len(still_bad.calls) == 2
    assert _issue_codes(result) == {"R1": "FINDING_CODE_NOT_AUTHORIZED"}


def test_every_executor_code_maps_to_a_published_hint():
    for check in OUTPUT_INVALID_CHECKS.values():
        assert output_repair_hint(check)
    assert output_repair_hint("no-such-check") is None


class _Coded(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@pytest.mark.parametrize(
    "failure, code",
    [
        (ProviderJSONOutputError("output_truncated"), "PROVIDER_OUTPUT_TRUNCATED"),
        (ProviderJSONOutputError("refused"), "PROVIDER_OUTPUT_REFUSED"),
        (_Coded("PROVIDER_CIRCUIT_OPEN"), "PROVIDER_CIRCUIT_OPEN"),
        (_Coded("TOKEN_BUDGET_EXCEEDED"), "TOKEN_BUDGET_EXCEEDED"),
        (KeyError("bug"), "RULE_EXECUTION_FAILED"),
    ],
)
def test_other_failures_are_not_asked_again(failure, code):
    request = _request("R1")
    runtime = _ScriptedRuntime({"R1": [failure]})

    result, _journal = _execute(request, runtime)

    assert runtime.calls == [("R1", None)]
    assert _issue_codes(result) == {"R1": code}


def test_the_per_paper_cap_stops_further_repairs(caplog):
    request = _request("R1", "R2")
    runtime = _ScriptedRuntime(
        {"R1": [_invalid(), _ok("R1")], "R2": [_invalid("status")]}, limit=1
    )

    with caplog.at_level(logging.WARNING):
        result, _journal = _execute(request, runtime)

    assert runtime.calls == [("R1", None), ("R1", "fields"), ("R2", None)]
    assert _issue_codes(result) == {"R2": "PROVIDER_OUTPUT_INVALID"}
    assert "rule_output_repair rule_code=R2 check=status outcome=skipped_cap" in caplog.text


def test_a_zero_limit_or_a_runtime_without_one_never_asks_again():
    request = _request("R1")
    off = _ScriptedRuntime({"R1": [_invalid()]}, limit=0)
    result, _journal = _execute(request, off)
    assert off.calls == [("R1", None)]
    assert _issue_codes(result) == {"R1": "PROVIDER_OUTPUT_INVALID"}

    class _Legacy:
        def score(self, *, envelope):
            raise _invalid()

    result, _journal = _execute(request, _Legacy())
    assert _issue_codes(result) == {"R1": "PROVIDER_OUTPUT_INVALID"}


def test_a_repair_that_could_not_be_sent_reports_the_original_problem():
    request = _request("R1")
    runtime = _ScriptedRuntime({"R1": [_invalid("level"), _Coded("TOKEN_BUDGET_EXCEEDED")]})

    result, _journal = _execute(request, runtime)

    issue = next(item for item in result["review_issues"] if item.get("rule_code") == "R1")
    assert issue["code"] == "PROVIDER_OUTPUT_INVALID" and "check=level" in issue["message"]


# ---------------------------------------------------------------- 互斥组


def _group_request():
    return groups._request(*groups._tiers())


def _group_answer(selected=None):
    return {
        code: m4._semantic_response(code, status="not_triggered") for code in ("T.1", "T.2", "T.3")
    }


def test_a_group_is_asked_again_once_as_a_group():
    request = _group_request()
    runtime = _GroupScriptedRuntime({"G": [_invalid("status"), _group_answer()]})

    result, journal = _execute(request, runtime)

    assert runtime.calls == [("G", None), ("G", "status")]
    assert _issue_codes(result) == {}
    assert journal.provider_attempts == {"T.1": 2, "T.2": 2, "T.3": 2}


def test_a_group_that_stays_invalid_fails_every_tier_without_splitting():
    request = _group_request()
    runtime = _GroupScriptedRuntime({"G": [_invalid("status"), _invalid("fields")]})

    result, _journal = _execute(request, runtime)

    # 只有两次整组调用，没有退回逐条判定。
    assert runtime.calls == [("G", None), ("G", "status")]
    assert _issue_codes(result) == {code: "PROVIDER_OUTPUT_INVALID" for code in ("T.1", "T.2", "T.3")}


def test_a_group_answer_rejected_by_the_executor_is_asked_again():
    request = _group_request()
    bad = _group_answer()
    bad["T.2"] = m4._semantic_response(
        "T.2", occurrences=[m4._quote_occurrence(request, finding_code="NOT_PUBLISHED")]
    )
    runtime = _GroupScriptedRuntime({"G": [bad, _group_answer()]})

    result, _journal = _execute(request, runtime)

    assert runtime.calls == [("G", None), ("G", "finding_code")]
    assert _issue_codes(result) == {}


# ---------------------------------------------------------------- 运行时


class _RepairableScorer:
    provider = "fake"

    def score_core_envelope(self, *, envelope, repair=None):
        return {"repair": repair}


class _OldScorer:
    provider = "fake"

    def score_core_envelope(self, *, envelope):
        return {}


@pytest.mark.parametrize("runtime_type", [ThesisLLMRuntime, TechnicalProposalLLMRuntime])
def test_runtimes_offer_repairs_only_for_scorers_that_accept_them(runtime_type, monkeypatch):
    monkeypatch.setattr(settings, "SCORING_OUTPUT_REPAIR_MAX_PER_PAPER", 3)
    assert runtime_type(_RepairableScorer()).output_repair_limit == 3
    assert runtime_type(_OldScorer()).output_repair_limit == 0
    assert runtime_type(MockLLMScorer()).output_repair_limit == 0

    monkeypatch.setattr(settings, "SCORING_OUTPUT_REPAIR_MAX_PER_PAPER", 0)
    assert runtime_type(_RepairableScorer()).output_repair_limit == 0


# ---------------------------------------------------------------- 适配器的请求形状


def _hint(check="fields"):
    hint = output_repair_hint(check)
    assert hint and MARKER not in hint
    return hint


def test_chat_repair_appends_one_fixed_message_and_changes_nothing_else():
    client = adapter_tests._CompatibleCoreClient()
    scorer = OpenAICompatibleChatScorer(
        api_key="test-key", base_url="https://provider.example/v1", model_name="repair-test",
        client=client, temperature=0, max_tokens=700, response_format_json=True, thinking_type="",
    )
    envelope = adapter_tests._core_envelope(scorer)
    client.output = adapter_tests._provider_response(envelope)

    scorer.score_core_envelope(envelope=envelope)
    first = deepcopy(client.payload)
    scorer.score_core_envelope(envelope=envelope, repair="fields")
    second = client.payload

    assert second["messages"][:2] == first["messages"]
    assert second["messages"][2] == {"role": "user", "content": _hint()}
    assert {k: v for k, v in second.items() if k != "messages"} == {
        k: v for k, v in first.items() if k != "messages"
    }


def test_responses_repair_appends_one_fixed_message():
    client = adapter_tests._OpenAICoreClient()
    scorer = OpenAIResponsesScorer(
        api_key="test-key", model_name="gpt-core-test", client=client, temperature=0,
        max_output_tokens=900,
    )
    envelope = adapter_tests._core_envelope(scorer)
    client.output = adapter_tests._provider_response(envelope)

    scorer.score_core_envelope(envelope=envelope, repair="evidence")

    user = build_core_request(envelope).user
    assert client.payload["input"] == [
        {"role": "user", "content": user},
        {"role": "user", "content": _hint("evidence")},
    ]


def test_messages_repair_adds_a_second_text_block_to_the_same_user_turn():
    capture = messages_tests.Capture(messages_tests._message("{}"))
    scorer = capture.scorer()
    envelope = adapter_tests._core_envelope(scorer)
    capture.bodies = [
        messages_tests._message(json.dumps(adapter_tests._provider_response(envelope), ensure_ascii=False))
    ]

    scorer.score_core_envelope(envelope=envelope, repair="status")

    assert capture.payload["messages"] == [
        {"role": "user", "content": [
            {"type": "text", "text": build_core_request(envelope).user},
            {"type": "text", "text": _hint("status")},
        ]}
    ]


def test_an_adapter_rejects_an_answer_with_a_named_check():
    client = adapter_tests._CompatibleCoreClient()
    scorer = OpenAICompatibleChatScorer(
        api_key="test-key", base_url="https://provider.example/v1", model_name="repair-test",
        client=client, temperature=0, max_tokens=700, response_format_json=True, thinking_type="",
    )
    envelope = adapter_tests._core_envelope(scorer)
    answer = adapter_tests._provider_response(envelope)
    del answer["occurrences"]
    client.output = answer

    with pytest.raises(CoreResponseInvalidError) as caught:
        scorer.score_core_envelope(envelope=envelope)

    assert (caught.value.reason, caught.value.check) == ("output_invalid", "fields")
    # 仍是 ValueError 的子类：既有的 except (TypeError, ValueError) 不受影响。
    assert isinstance(caught.value, ValueError)
    assert set(OUTPUT_REPAIR_HINTS) >= {caught.value.check}


# ---------------------------------------------------------------- 报错


def test_code_failures_report_where_they_happened_without_the_message():
    def boom():
        raise KeyError(MARKER)

    try:
        boom()
    except KeyError as exc:
        code, message = project_rule_execution_failure(exc)
        site = failure_site(exc)

    assert code == "RULE_EXECUTION_FAILED"
    assert re.fullmatch(r"tests/test_scoring_output_repair:\d+", site)
    assert message == "rule execution failed (exception_type=KeyError, where=%s)" % site
    assert MARKER not in message
    assert failure_site(ValueError("never raised")) == "-"


def test_an_unsafe_check_is_not_copied_into_the_message():
    error = CoreResponseInvalidError("fields " + MARKER)
    assert project_rule_execution_failure(error) == (
        "PROVIDER_OUTPUT_INVALID",
        "semantic provider output failed validation (check=fields)",
    )


@pytest.mark.parametrize("code", ["PROVIDER_OUTPUT_INVALID", *OUTPUT_INVALID_CHECKS])
def test_the_paper_level_error_names_the_model_output(code):
    error = jobs._incomplete_scoring_error(
        [SimpleNamespace(status="failed_exhausted", provider_error={"code": code})]
    )

    assert error.code == "PROVIDER_OUTPUT_INVALID"
    assert "没有通过格式校验" in str(error) and "重试失败项" in str(error)
    # 说明不声称重问过：上限为 0 或名额用完时并没有重问。
    assert "已自动重问" not in str(error)
    assert jobs._classify_failure(error) == ("PROVIDER_OUTPUT_INVALID", "llm")
    title, _text = assistant_views.DIAGNOSES[assistant_views.normalize_error_code(error.code)]
    assert title == "模型回答不合格"


def test_a_code_failure_keeps_the_system_error_wording():
    error = jobs._incomplete_scoring_error(
        [SimpleNamespace(status="failed_exhausted", provider_error=None)]
    )
    assert error.code == "RULE_EXECUTION_FAILED"


# ---------------------------------------------------------------- 一篇论文（落库）

from backend.app.services.scoring import engine as scoring_engine  # noqa: E402
from backend.app.tests import test_provider_output_observation as observed  # noqa: E402
from backend.app.tests.test_provider_output_observation import thinking_paper  # noqa: E402,F401


class _FirstRuleScorer(observed._ThinkingScorer):
    """第一条被问到的规则先答一次不合格；``mode`` 决定重问后怎么答。"""

    reasoning = {}
    mode = "repaired"
    target = None
    repairs = []

    def score_core_envelope(self, *, envelope, repair=None):
        cls = type(self)
        answer = super().score_core_envelope(envelope=envelope)
        code = answer["rule_code"]
        if cls.target is None:
            cls.target = code
        if code != cls.target:
            return answer
        if repair is not None:
            cls.repairs.append(repair)
        if repair is None or cls.mode == "adapter":
            raise CoreResponseInvalidError("fields")
        if cls.mode == "executor":
            return {**answer, "rule_code": "someone-else"}
        return answer


@pytest.fixture
def first_rule_scorer(monkeypatch, thinking_paper):  # noqa: F811
    monkeypatch.setattr(_FirstRuleScorer, "target", None)
    monkeypatch.setattr(_FirstRuleScorer, "repairs", [])
    monkeypatch.setattr(
        scoring_engine, "get_llm_scorer", lambda *args, **kwargs: _FirstRuleScorer()
    )
    return thinking_paper


def _target_task(db, run):
    tasks = observed._semantic_tasks(db, run.id)
    return next(task for task in tasks if task.rule_code == _FirstRuleScorer.target), tasks


def test_a_repaired_rule_is_stored_as_two_attempts_and_nothing_fails(
    client, first_rule_scorer, monkeypatch
):
    monkeypatch.setattr(_FirstRuleScorer, "mode", "repaired")
    with client.session_factory() as db:
        run = scoring_engine.score_paper(db, first_rule_scorer)
        task, tasks = _target_task(db, run)

        assert _FirstRuleScorer.repairs == ["fields"]
        assert (task.status, task.attempt_count, task.max_attempts) == ("succeeded", 2, 2)
        assert task.provider_error is None
        assert all(other.max_attempts == 1 for other in tasks if other is not task)
        # 这套夹具的档位规则一律答 not_triggered，本来就不出总分；这里只看没有规则因此失败。
        assert not [other for other in tasks if other.status == "failed_exhausted"]


@pytest.mark.parametrize(
    "mode, stored_code", [("adapter", "PROVIDER_OUTPUT_INVALID"), ("executor", "UNKNOWN_RULE")]
)
def test_a_rule_that_stays_invalid_records_the_model_output_as_the_cause(
    client, first_rule_scorer, monkeypatch, mode, stored_code
):
    monkeypatch.setattr(_FirstRuleScorer, "mode", mode)
    with client.session_factory() as db:
        run = scoring_engine.score_paper(db, first_rule_scorer)
        task, tasks = _target_task(db, run)

        assert (task.status, task.attempt_count, task.max_attempts) == ("failed_exhausted", 2, 2)
        assert task.provider_error["code"] == stored_code
        assert run.final_total_score is None
        error = jobs._incomplete_scoring_error(tasks)
        assert (error.code, jobs._classify_failure(error)[1]) == ("PROVIDER_OUTPUT_INVALID", "llm")


# ---------------------------------------------------------------- #103 复审


def test_the_core_and_the_adapter_agree_on_the_checks():
    from backend.app.services.scoring.core.failures import OUTPUT_REPAIR_CHECKS

    assert set(OUTPUT_REPAIR_HINTS) == set(OUTPUT_REPAIR_CHECKS)
    assert set(OUTPUT_INVALID_CHECKS.values()) <= set(OUTPUT_REPAIR_CHECKS)


def test_an_unregistered_check_is_neither_logged_nor_used_for_the_retry(caplog):
    request = _request("R1")
    # 外来评分器给了一个格式合法、但没有登记的标识，以及一段任意文字。
    for foreign in ("no_such_check", MARKER):
        runtime = _ScriptedRuntime(
            {"R1": [CoreResponseInvalidError(foreign), CoreResponseInvalidError(foreign)]}
        )
        with caplog.at_level(logging.WARNING):
            result, _journal = _execute(request, runtime)

        assert runtime.calls == [("R1", None), ("R1", "fields")]
        issue = next(item for item in result["review_issues"] if item.get("rule_code") == "R1")
        assert issue["message"].endswith("(check=fields)")
    assert "no_such_check" not in caplog.text and MARKER not in caplog.text


def _chat_scorer():
    client = adapter_tests._CompatibleCoreClient()
    scorer = OpenAICompatibleChatScorer(
        api_key="test-key", base_url="https://provider.example/v1", model_name="repair-test",
        client=client, temperature=0, max_tokens=700, response_format_json=True, thinking_type="",
    )
    return scorer, client


@pytest.mark.parametrize(
    "change, check",
    [
        ({"occurrences": 7}, "fields"),
        ({"occurrences": "none"}, "fields"),
        ({"status": ["triggered"]}, "status"),
        ({"level_code": []}, "level"),
    ],
)
def test_wrongly_typed_fields_are_invalid_answers_not_crashes(change, check):
    scorer, client = _chat_scorer()
    envelope = adapter_tests._core_envelope(scorer)
    client.output = {**adapter_tests._provider_response(envelope), **change}

    with pytest.raises(CoreResponseInvalidError) as caught:
        scorer.score_core_envelope(envelope=envelope)

    assert caught.value.check == check


@pytest.mark.parametrize(
    "path, value, check",
    [
        (("finding_code",), ["GAP"], "finding_code"),
        (("evidence",), "E1", "triggered_without_evidence"),
        (("evidence", 0, "evidence_unit_id"), ["E1"], "evidence"),
    ],
)
def test_wrongly_typed_occurrence_fields_are_invalid_answers(path, value, check):
    scorer, client = _chat_scorer()
    envelope = adapter_tests._core_envelope(scorer)
    answer = adapter_tests._provider_response(envelope)
    target = answer["occurrences"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    client.output = answer

    with pytest.raises(CoreResponseInvalidError) as caught:
        scorer.score_core_envelope(envelope=envelope)

    assert caught.value.check == check


@pytest.mark.parametrize(
    "body, reason",
    [
        ({"id": "r", "output_text": "{not json"}, "invalid_json"),
        ({"id": "r", "output": []}, "empty_content"),
    ],
)
def test_responses_parse_failures_carry_a_reason_so_they_can_be_asked_again(body, reason):
    from backend.app.services.llm.openai_adapter import _parse_json_output
    from backend.app.services.scoring.core.rule_executor import _output_invalid_check

    with pytest.raises(ProviderJSONOutputError) as caught:
        _parse_json_output(body)

    assert caught.value.reason == reason and isinstance(caught.value, ValueError)
    assert _output_invalid_check(caught.value) == "json"
    assert project_rule_execution_failure(caught.value)[0] == "PROVIDER_OUTPUT_INVALID"


def test_the_repair_hint_is_counted_in_the_input_estimate():
    from backend.app.services.llm.core_view import request_input_estimate
    from backend.app.services.llm.core_view import with_repair
    from backend.app.services.scoring.retrieval.selection import conservative_token_estimate

    scorer, _client = _chat_scorer()
    request = build_core_request(adapter_tests._core_envelope(scorer))
    repaired = with_repair(request, "evidence")

    extra = request_input_estimate(repaired) - request_input_estimate(request)
    assert extra > conservative_token_estimate(_hint("evidence"))
    assert request_input_estimate(with_repair(request, None)) == request_input_estimate(request)


def test_a_group_is_not_asked_again_when_the_runtime_cannot_repair_groups():
    request = _group_request()
    runtime = _GroupScriptedRuntime({"G": [_invalid("status")]})
    runtime.supports_group_repair = False

    result, _journal = _execute(request, runtime)

    assert runtime.calls == [("G", None)]
    assert _issue_codes(result) == {code: "PROVIDER_OUTPUT_INVALID" for code in ("T.1", "T.2", "T.3")}


def test_the_paper_runtime_checks_the_group_entry_separately():
    class _GroupWithoutRepair(_RepairableScorer):
        def score_core_group(self, *, envelopes, group_code):
            return {}

    class _GroupWithRepair(_RepairableScorer):
        def score_core_group(self, *, envelopes, group_code, repair=None):
            return {}

    assert ThesisLLMRuntime(_GroupWithoutRepair()).supports_group_repair is False
    assert ThesisLLMRuntime(_GroupWithRepair()).supports_group_repair is True
    assert ThesisLLMRuntime(_RepairableScorer()).supports_group_repair is False
    # 单条规则的判断不受整组入口影响。
    assert ThesisLLMRuntime(_GroupWithoutRepair()).output_repair_limit > 0


@pytest.mark.parametrize(
    "body",
    [
        {"id": "r", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": MARKER}]}]},
        {"id": "r", "output_text": "", "output": [{"content": [{"type": "refusal", "refusal": MARKER}]}]},
        # 拒答标记与正文同时出现：正文是合法 JSON 也好、不是也好，都不采用。
        {"id": "r", "output_text": "{\"ok\": true}", "output": [{"content": [{"type": "refusal", "refusal": MARKER}]}]},
        {"id": "r", "output": [{"content": [
            {"type": "output_text", "text": "{not json"}, {"type": "refusal", "refusal": MARKER},
        ]}]},
    ],
)
def test_a_responses_refusal_is_a_refusal_not_an_empty_answer(body):
    from backend.app.services.llm.openai_adapter import _parse_json_output
    from backend.app.services.scoring.core.rule_executor import _output_invalid_check

    with pytest.raises(ProviderJSONOutputError) as caught:
        _parse_json_output(body)

    assert caught.value.reason == "refused" and MARKER not in str(caught.value)
    # 拒答不重问，照旧报成拒答。
    assert _output_invalid_check(caught.value) is None
    assert project_rule_execution_failure(caught.value)[0] == "PROVIDER_OUTPUT_REFUSED"


@pytest.mark.parametrize(
    "choice, reason",
    [
        ({"message": {"role": "assistant", "content": None, "refusal": MARKER}}, "refused"),
        ({"finish_reason": "content_filter", "message": {"role": "assistant", "content": ""}}, "refused"),
        ({"finish_reason": "length", "message": {"role": "assistant", "content": "", "refusal": "x"}}, "output_truncated"),
        # 拒答标记与正文同时出现：不采用正文。
        ({"message": {"role": "assistant", "content": "{\"ok\": true}", "refusal": MARKER}}, "refused"),
        ({"finish_reason": "content_filter", "message": {"role": "assistant", "content": "{not json"}}, "refused"),
        # 截断 + 拒答标记 + 合法 JSON 正文：报截断，正文同样不采用。
        ({"finish_reason": "length", "message": {"role": "assistant", "content": "{\"ok\": true}", "refusal": MARKER}}, "output_truncated"),
        ({"message": {"role": "assistant", "content": "", "refusal": None}}, "empty_content"),
        ({"message": {"role": "assistant", "content": ""}}, "empty_content"),
    ],
)
def test_a_chat_refusal_is_told_apart_from_an_empty_answer(choice, reason):
    from backend.app.services.llm.openai_compatible_adapter import _parse_chat_json_output

    with pytest.raises(ProviderJSONOutputError) as caught:
        _parse_chat_json_output({"id": "c", "choices": [choice]})

    assert caught.value.reason == reason and MARKER not in str(caught.value)


@pytest.mark.parametrize(
    "body, reason",
    [
        ({"id": "r", "output": [{"content": [{"type": "refusal", "refusal": MARKER}]}]}, "refused"),
        ({"id": "r", "output": []}, "empty_content"),
        ({"id": "r", "output_text": "{not json"}, "invalid_json"),
    ],
)
def test_responses_complete_json_keeps_the_parsed_classification(body, reason):
    class _Client:
        def post(self, url, headers, json):
            return adapter_tests._Response(body)

    scorer = OpenAIResponsesScorer(
        api_key="test-key", model_name="gpt-core-test", client=_Client(), temperature=0,
        max_output_tokens=900,
    )

    with pytest.raises(ProviderJSONOutputError) as caught:
        scorer.complete_json("只输出 JSON。", {"q": 1})

    # 拒答不能被改报成格式错误，否则归类等流程会把它当格式问题再问一次。
    assert caught.value.reason == reason and MARKER not in str(caught.value)


def test_a_truncated_chat_answer_without_a_refusal_marker_still_uses_valid_json():
    from backend.app.services.llm.openai_compatible_adapter import _parse_chat_json_output

    choice = {"finish_reason": "length", "message": {"role": "assistant", "content": "{\"ok\": true}"}}
    assert _parse_chat_json_output({"id": "c", "choices": [choice]}) == {"ok": True}

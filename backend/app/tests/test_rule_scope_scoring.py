"""评分时应用适用范围：缺章不调用模型、多章对照的引用检查、章节角色归类（方案第 8 节）。

全部用合成数据与假运行时，不调用真实模型。
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.app.services.scoring.core.rule_executor import SCOPE_CITATION_INCOMPLETE_CODE
from backend.app.services.scoring.core.rule_executor import SCOPE_SECTION_MISSING_CODE
from backend.app.services.scoring.core.rule_executor import _same_scope_rules
from backend.app.services.scoring.profiles.thesis import ThesisLLMRuntime
from backend.app.services.scoring.retrieval.scope_policy import RULE_SCOPE_POLICY
from backend.app.services.scoring.retrieval.section_roles import SECTION_ROLE_MAP_VERSION
from backend.app.services.scoring.retrieval.selection import build_v4_envelope
from backend.app.services.scoring.review_reasons import from_core_issues
from backend.app.tests import test_m4_rule_executor as m4
from backend.app.tests.test_m4_rule_executor import _criterion
from backend.app.tests.test_m4_rule_executor import _criterion_outcome
from backend.app.tests.test_m4_rule_executor import _decision
from backend.app.tests.test_m4_rule_executor import _deduct_rule
from backend.app.tests.test_m4_rule_executor import _quote_occurrence
from backend.app.tests.test_m4_rule_executor import _request_for
from backend.app.tests.test_m4_rule_executor import _semantic_response
from backend.app.tests.test_scoring_stage2 import BUDGET


class _ScopedProfile(m4._TechnicalProposalProfile):
    """测试用 Profile：按范围取证据（相当于打开 SCORING_JUDGE_SCOPE_ENABLED）。"""

    rule_scope_policy = RULE_SCOPE_POLICY

    def build_provider_envelope(self, *, base_envelope, selection_rules=None, criterion_rules=None):
        return build_v4_envelope(base_envelope, top_k=12, judge_scope=True,
                                 selection_rules=selection_rules, **BUDGET)


def _scoped_rule(scope_type, sections):
    return {
        **_deduct_rule(),
        "schema_version": "atomic-rule-snapshot@5",
        "name": "需求与方案对应", "rule_text": "需求理解中的目标在总体方案里没有对应",
        "positive_example": None, "negative_example": None, "boundary_example": None, "context_needs": [],
        "scope": {"type": scope_type, "sections": list(sections)},
    }


def _run(rule, responses, *, runtime=None):
    request = _request_for((_criterion(), rule))
    runtime = runtime or m4._SemanticRuntime(responses)
    result = m4._require_executor()(
        request=request, checker_registry=m4._CheckerRegistry(), llm_runtime=runtime, profile=_ScopedProfile(),
    )
    return m4._result_mapping(result), runtime, request


def _scope_issues(result):
    return [item for item in result["review_issues"] if item["code"].startswith("RULE_SCOPE_")]


def test_missing_target_section_skips_the_model_and_routes_to_review():
    rule = _scoped_rule("section", ["经费预算"])
    result, runtime, _request = _run(rule, {})

    assert runtime.envelopes == []  # 没有调用模型
    assert _decision(result)["status"] == "not_triggered"
    assert [item["code"] for item in _scope_issues(result)] == [SCOPE_SECTION_MISSING_CODE]
    assert "经费预算" in _scope_issues(result)[0]["message"]
    assert _criterion_outcome(result)["status"] == "review_required"


def test_cross_section_triggered_without_two_sections_is_not_adopted():
    rule = _scoped_rule("cross_section", ["需求分析", "总体方案"])
    request = _request_for((_criterion(), rule))
    response = _semantic_response(rule["rule_code"], occurrences=[_quote_occurrence(request, 1)])
    result, runtime, _ = _run(rule, {rule["rule_code"]: response})

    assert len(runtime.envelopes) == 1
    assert runtime.envelopes[0]["coverage_mode"] == "cross_section"
    decision = _decision(result)
    assert decision["status"] == "not_triggered" and decision["occurrences"] == []
    assert [item["code"] for item in _scope_issues(result)] == [SCOPE_CITATION_INCOMPLETE_CODE]
    assert _criterion_outcome(result)["status"] == "review_required"


def test_cross_section_citing_both_sections_is_adopted():
    rule = _scoped_rule("cross_section", ["需求分析", "总体方案"])
    request = _request_for((_criterion(), rule))
    response = _semantic_response(rule["rule_code"], occurrences=[
        _quote_occurrence(request, 0, occurrence_id="a"), _quote_occurrence(request, 1, occurrence_id="b")])
    result, _runtime, _ = _run(rule, {rule["rule_code"]: response})

    assert _decision(result)["status"] == "triggered"
    assert _scope_issues(result) == []


def test_document_scope_rule_is_judged_as_before():
    rule = _scoped_rule("document", [])
    response = _semantic_response(rule["rule_code"], status="not_triggered")
    result, runtime, _ = _run(rule, {rule["rule_code"]: response})

    assert len(runtime.envelopes) == 1 and runtime.envelopes[0]["coverage_mode"] != "section"
    assert _decision(result)["status"] == "not_triggered" and _scope_issues(result) == []
    # 没有规则要看章节：信封里不加章节角色归类。
    assert "section_roles" not in runtime.envelopes[0]["profile_prompt_extensions"]


def test_scoped_rule_envelope_carries_the_section_role_map():
    rule = _scoped_rule("section", ["总体方案"])
    response = _semantic_response(rule["rule_code"], status="not_triggered")
    result, runtime, _ = _run(rule, {rule["rule_code"]: response})

    roles = runtime.envelopes[0]["profile_prompt_extensions"]["section_roles"]
    assert roles == {"version": SECTION_ROLE_MAP_VERSION, "headings": {}}  # 标题都在词表里，没有调用归类
    assert {unit["section_path"][0] for unit in runtime.envelopes[0]["evidence_units"]} == {"总体方案"}


def _role_value(headings):
    units = [{"section_path": [heading, "小节"]} for heading in headings]
    return {"runtime_identity": {"profile_key": "technical_proposal"}, "document": {"evidence_units": units}}


def test_section_role_map_classifies_only_unmatched_headings_once():
    calls = []

    class Runtime:
        def classify_section_headings(self, *, headings, profile_key):
            calls.append((list(headings), profile_key))
            return {"背景说明": "项目概述"}

    value = _role_value(["背景说明", "总体方案", "背景说明"])
    rules = [{"judge_type": "semantic", "scope": {"type": "section", "sections": ["项目概述"]}}]
    mapped = RULE_SCOPE_POLICY.section_role_map(value=value, rules=rules, llm_runtime=Runtime())
    assert calls == [(["背景说明"], "technical_proposal")]
    assert mapped == {"version": SECTION_ROLE_MAP_VERSION, "headings": {"背景说明": "项目概述"}}

    # 没有规则要看章节：不归类。
    assert RULE_SCOPE_POLICY.section_role_map(
        value=value, rules=[{"judge_type": "semantic"}], llm_runtime=Runtime()) is None
    assert len(calls) == 1


def test_section_role_classification_failure_falls_back_to_the_vocabulary():
    class Broken:
        def classify_section_headings(self, *, headings, profile_key):
            raise RuntimeError("provider down")

    rules = [{"judge_type": "semantic", "scope": {"type": "section", "sections": ["项目概述"]}}]
    mapped = RULE_SCOPE_POLICY.section_role_map(value=_role_value(["背景说明"]), rules=rules, llm_runtime=Broken())
    assert mapped["headings"] == {}


def test_runtime_classifies_headings_with_one_metered_call():
    class Scorer:
        provider = "openai_compatible"
        model_name = "fake"

        def __init__(self):
            self.calls = []

        def complete_json(self, instructions, payload, response_schema=None, default_max_tokens=None,
                          attempts_limit=None, scoring=False):
            self.calls.append({"payload": payload, "schema": response_schema, "scoring": scoring})
            return {"headings": [{"heading": "背景说明", "role": "项目概述"},
                                 {"heading": "其它", "role": "不存在的角色"},
                                 {"heading": "伪造的标题", "role": "附录"}]}

    scorer = Scorer()
    runtime = ThesisLLMRuntime(scorer)
    result = runtime.classify_section_headings(headings=["背景说明", "其它"], profile_key="technical_proposal")
    assert result == {"背景说明": "项目概述"}
    assert len(scorer.calls) == 1 and scorer.calls[0]["scoring"] is True
    assert scorer.calls[0]["payload"]["headings"] == ["背景说明", "其它"]
    assert scorer.calls[0]["schema"]["properties"]["headings"]["items"]["properties"]["heading"]["enum"] == [
        "背景说明", "其它"]

    assert ThesisLLMRuntime(SimpleNamespace(provider="mock")).classify_section_headings(
        headings=["背景说明"], profile_key="technical_proposal") == {}


def test_shared_selection_is_limited_to_rules_with_the_same_scope():
    section = {"rule_code": "a", "scope": {"type": "section", "sections": ["绪论"]}}
    other = {"rule_code": "b", "scope": {"type": "section", "sections": ["结论"]}}
    same = {"rule_code": "c", "scope": {"type": "section", "sections": ["绪论"]}}
    assert _same_scope_rules([section, other, same], section) == [section, same]
    assert _same_scope_rules(None, section) is None
    plain = [{"rule_code": "x"}, {"rule_code": "y"}]
    assert _same_scope_rules(plain, plain[0]) == plain  # 没开范围时照旧共用


def test_scope_review_issues_become_structured_reasons():
    reasons = from_core_issues([
        {"code": SCOPE_SECTION_MISSING_CODE, "severity": "review", "criterion_code": "T01", "rule_code": "R1", "message": "x"},
        {"code": SCOPE_CITATION_INCOMPLETE_CODE, "severity": "review", "criterion_code": "T01", "rule_code": "R2", "message": "x"},
    ], rule_names={"R1": "预算", "R2": "对应"})
    assert [entry["code"] for entry in reasons] == ["scope_section_missing", "scope_citation_incomplete"]
    assert "没有找到" in reasons[0]["message"] and "两章" in reasons[1]["message"]

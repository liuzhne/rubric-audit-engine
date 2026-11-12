"""第二阶段改造：预算量实际视图、判定范围、证据不足（docs/论文评分第二阶段改造方案.md）。

全部用合成数据与假运行时，不调用真实模型。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from backend.app.core.config import settings
from backend.app.services.llm import core_view
from backend.app.services.scoring.core.contracts import PromptEnvelopeV3
from backend.app.services.scoring.core.contracts import derive_evidence_unit_id
from backend.app.services.scoring.core.rule_executor import EVIDENCE_INSUFFICIENT_CODE
from backend.app.services.scoring.core.rule_executor import _prompt_envelope
from backend.app.services.scoring.retrieval import judge_scope
from backend.app.services.scoring.retrieval import section_roles
from backend.app.services.scoring.retrieval.selection import TOKEN_POLICY_VERSION
from backend.app.services.scoring.retrieval.selection import VIEW_TOKEN_POLICY_VERSION
from backend.app.services.scoring.retrieval.selection import TokenBudgetError
from backend.app.services.scoring.retrieval.selection import build_v4_envelope
from backend.app.tests.m3_contract_fixtures import SEMANTIC_RULE_CODE
from backend.app.tests.m3_contract_fixtures import scoring_request_payload
from backend.app.tests.test_m4_rule_executor import _band_rule
from backend.app.tests.test_m4_rule_executor import _criterion
from backend.app.tests.test_m4_rule_executor import _criterion_outcome
from backend.app.tests.test_m4_rule_executor import _decision
from backend.app.tests.test_m4_rule_executor import _deduct_rule
from backend.app.tests.test_m4_rule_executor import _execute
from backend.app.tests.test_m4_rule_executor import _request_for
from backend.app.tests.test_m4_rule_executor import _semantic_response


BUDGET = {"context_window_tokens": 9040, "reserved_output_tokens": 2048, "safety_margin_tokens": 1024}
CROSS_RULE_TEXT = {
    "name": "研究问题与结论呼应",
    "rule_text": "绪论提出的研究问题在结论中没有得到回应",
}


class _Profile:
    profile_key = "technical_proposal"
    profile_version = "technical-proposal-profile@1"
    prompt_version = "technical-proposal-prompt@1"

    def build_prompt_extensions(self, *, submission_snapshot, document_snapshot):
        return {"metadata": {}, "profile_extensions": {}}


def _base_mapping():
    request = scoring_request_payload()
    node = next(
        item for item in request["plan"]["nodes"] if item["rule_code"] == SEMANTIC_RULE_CODE
    )
    return _prompt_envelope(request=request, node=node, profile=_Profile()).to_mapping()


def _unit(content_hash, section_path, section_ordinal, unit_ordinal, text):
    normalized = {
        "normalized_content_hash": content_hash,
        "section_path": list(section_path),
        "section_ordinal": section_ordinal,
        "unit_ordinal": unit_ordinal,
        "normalized_text": text,
    }
    unit_id = derive_evidence_unit_id(normalized)
    return {
        **normalized,
        "evidence_unit_id": unit_id,
        "unit_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "locator": {"kind": "text_span", "evidence_unit_id": unit_id, "start": 0, "end": len(text)},
    }


def _paper_units(content_hash):
    sections = [
        (["第1章 绪论"], ["本文的研究问题是如何提升校园互助的响应效率。", "研究目标是设计一个互助平台。"]),
        (
            ["第3章 系统设计"],
            [
                "系统设计围绕研究问题展开，结论部分会再讨论。" * 3,
                "数据库设计采用关系模型，结论在第五章。" * 3,
                "接口设计说明了研究问题对应的模块划分。" * 3,
            ],
        ),
        (["第5章 结论", "5.1 总结"], ["本文完成了平台实现，回应了绪论中的研究问题。"]),
    ]
    units = []
    for section_ordinal, (path, texts) in enumerate(sections):
        for unit_ordinal, text in enumerate(texts):
            units.append(_unit(content_hash, path, section_ordinal, unit_ordinal, text))
    return units


def _envelope(*, units=None, rule=None, criterion=None):
    value = _base_mapping()
    if units is not None:
        value["evidence_units"] = units
    if rule is not None:
        value["atomic_rule_snapshot"] = {
            **value["atomic_rule_snapshot"],
            "schema_version": "atomic-rule-snapshot@3",
            "positive_example": None,
            "negative_example": None,
            "boundary_example": None,
            "context_needs": [],
            **rule,
        }
    if criterion is not None:
        value["criterion_snapshot"] = {**value["criterion_snapshot"], **criterion}
    return PromptEnvelopeV3.from_mapping(value)


def _paper_envelope(rule=None):
    content_hash = _base_mapping()["evidence_units"][0]["normalized_content_hash"]
    return _envelope(units=_paper_units(content_hash), rule=rule or CROSS_RULE_TEXT)


# ── 分期 A：预算量实际视图 ──────────────────────────────────────────


def test_view_budget_fits_where_envelope_budget_fails():
    content_hash = _base_mapping()["evidence_units"][0]["normalized_content_hash"]
    units = [
        _unit(content_hash, ["第%d章" % (index + 1)], index, 0, "实验数据与分析过程说明。" * 30)
        for index in range(40)
    ]
    # 评分项名带“缺失”→ 旧逻辑要求整篇放下（exhaustive）。
    envelope = _envelope(units=units, criterion={"name": "必备内容缺失"})

    with pytest.raises(TokenBudgetError):
        build_v4_envelope(envelope, top_k=12, **BUDGET)

    value = build_v4_envelope(envelope, top_k=12, budget_target="provider_view", **BUDGET).to_mapping()
    budget = value["token_budget_identity"]
    assert value["coverage_mode"] == "exhaustive"
    assert budget["policy_version"] == VIEW_TOKEN_POLICY_VERSION
    assert budget["within_budget"] is True


def test_view_budget_records_the_estimate_of_the_request_actually_sent():
    envelope = build_v4_envelope(_paper_envelope(), top_k=12, budget_target="provider_view", **BUDGET)
    request = core_view.build_core_request(envelope)

    budget = envelope.to_mapping()["token_budget_identity"]
    assert budget["estimated_input_tokens"] == core_view.request_input_estimate(request)
    assert budget["estimated_input_tokens"] < budget["context_window_tokens"]


def test_envelope_target_keeps_the_previous_identity():
    envelope = _paper_envelope()
    default = build_v4_envelope(envelope, top_k=12, **BUDGET).to_mapping()
    explicit = build_v4_envelope(envelope, top_k=12, budget_target="envelope", **BUDGET).to_mapping()

    assert default == explicit
    assert default["token_budget_identity"]["policy_version"] == TOKEN_POLICY_VERSION


def test_unknown_budget_target_is_rejected():
    with pytest.raises(ValueError, match="budget_target"):
        build_v4_envelope(_paper_envelope(), top_k=3, budget_target="whole", **BUDGET)


# ── 分期 B：判定范围（docs/评分细则适用范围方案.md 第 8 节） ─────────────


def _scoped_rule(scope_type, sections, **text):
    """@5 规则：范围来自细则自带的 scope，不再从 applies_to 或规则文字推断。"""

    return {"schema_version": "atomic-rule-snapshot@5", **(text or CROSS_RULE_TEXT),
            "scope": {"type": scope_type, "sections": list(sections)}}


def _thesis_envelope(rule, *, roles=None):
    """毕业论文 Profile 的信封：章节角色按毕业论文词表对应。"""

    content_hash = _base_mapping()["evidence_units"][0]["normalized_content_hash"]
    value = _envelope(units=_paper_units(content_hash), rule=rule).to_mapping()
    value["runtime_identity"]["profile_key"] = "thesis"
    if roles is not None:
        value["profile_prompt_extensions"]["section_roles"] = {"version": "section-role-map@1", "headings": roles}
    return PromptEnvelopeV3.from_mapping(value)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ({"judge_type": "semantic", "scope": {"type": "cross_section", "sections": ["绪论", "结论"]}},
         ("cross_section", ["绪论", "结论"])),
        ({"judge_type": "semantic", "scope": {"type": "section", "sections": ["系统设计"]}}, ("section", ["系统设计"])),
        ({"judge_type": "semantic", "scope": {"type": "document", "sections": []}}, ("document", [])),
        # 不再按 applies_to 或规则文字推断：没有 scope 一律全文。
        ({"judge_type": "semantic", "applies_to": "系统设计", "name": "", "rule_text": ""}, ("document", [])),
        ({"judge_type": "semantic", **CROSS_RULE_TEXT}, ("document", [])),
        ({"judge_type": "deterministic", "scope": {"type": "section", "sections": ["绪论"]}}, ("document", [])),
    ],
)
def test_rule_scope_comes_only_from_the_snapshot(rule, expected):
    scope = judge_scope.rule_scope(rule)
    assert (scope["scope"], scope["targets"]) == expected


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("第1章 绪论", "绪论"),
        ("1 引言", "绪论"),
        ("4.2 总体设计", "系统设计"),
        ("第5章 总结与展望", "结论"),
        ("系统设计与实现", "系统设计"),  # 取命中最长的同义词
        ("致谢", "致谢"),
        ("第3章 校园互助", None),
        ("", None),
    ],
)
def test_heading_role_strips_numbering_and_uses_the_vocabulary(heading, expected):
    assert section_roles.heading_role(heading, "thesis") == expected


def test_cross_section_selection_takes_evidence_from_each_target_first():
    value = build_v4_envelope(_thesis_envelope(_scoped_rule("cross_section", ["绪论", "结论"])),
                              top_k=3, judge_scope=True, **BUDGET).to_mapping()

    assert value["coverage_mode"] == "cross_section"
    assert value["evidence_selection_identity"]["selector_version"].endswith("+judge-scope@2")
    first_two = [unit["section_path"][0] for unit in value["evidence_units"][:2]]
    assert first_two == ["第1章 绪论", "第5章 结论"]
    assert all("系统设计" not in unit["section_path"][0] for unit in value["evidence_units"])
    # 开了范围时，未选的标题路径也列出，用来判断整章与缺章。
    paths = {tuple(item["section_path"]) for item in value["evidence_selection_identity"]["section_coverage"]}
    assert ("第3章 系统设计",) in paths


def test_section_scope_gives_the_whole_chapter_when_it_fits():
    value = build_v4_envelope(_thesis_envelope(_scoped_rule("section", ["系统设计"])),
                              top_k=1, judge_scope=True, **BUDGET).to_mapping()
    assert value["coverage_mode"] == "section"
    assert len(value["evidence_units"]) == 3
    assert {unit["section_path"][0] for unit in value["evidence_units"]} == {"第3章 系统设计"}


def test_scope_switch_off_matches_the_unscoped_identity():
    envelope = _paper_envelope()
    off = build_v4_envelope(envelope, top_k=3, **BUDGET).to_mapping()

    assert off["coverage_mode"] != "cross_section"
    # BM25 alone is pulled toward the long design paragraphs that repeat the words.
    assert any("系统设计" in unit["section_path"][0] for unit in off["evidence_units"])


def test_document_scope_rule_is_unchanged_when_the_switch_is_on():
    envelope = _paper_envelope(rule={"name": "语言表达规范", "rule_text": "存在大量口语化表达"})
    off = build_v4_envelope(envelope, top_k=3, **BUDGET).to_mapping()
    on = build_v4_envelope(envelope, top_k=3, judge_scope=True, **BUDGET).to_mapping()

    assert on == off


def test_missing_target_is_reported_and_heading_overrides_are_used():
    missing = build_v4_envelope(_thesis_envelope(_scoped_rule("section", ["参考文献"])),
                                top_k=3, judge_scope=True, **BUDGET).to_mapping()
    assert judge_scope.missing_targets(missing) == ["参考文献"]

    # 第 3 章标题词表认不出时，本篇的模型归类结果说它是“需求分析”。
    rule = _scoped_rule("section", ["需求分析"])
    plain = build_v4_envelope(_thesis_envelope(rule), top_k=3, judge_scope=True, **BUDGET).to_mapping()
    assert judge_scope.missing_targets(plain) == ["需求分析"]
    classified = build_v4_envelope(_thesis_envelope(rule, roles={"第3章 系统设计": "需求分析"}),
                                   top_k=3, judge_scope=True, **BUDGET).to_mapping()
    assert judge_scope.missing_targets(classified) == []
    assert {unit["section_path"][0] for unit in classified["evidence_units"]} == {"第3章 系统设计"}


def test_view_reports_scope_coverage_and_section_roles(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    envelope = build_v4_envelope(_thesis_envelope(_scoped_rule("cross_section", ["绪论", "结论"])),
                                 top_k=3, judge_scope=True, **BUDGET)
    request = core_view.build_core_request(envelope)
    view = json.loads(request.user)

    assert view["scope"] == {"type": "cross_section", "sections": [
        {"role": "绪论", "headings": ["第1章 绪论"], "coverage": "complete"},
        {"role": "结论", "headings": ["第5章 结论"], "coverage": "complete"},
    ]}
    sections = {item["section"] for item in view["evidence"]}
    assert "第5章 结论 › 5.1 总结" in sections
    assert {item["section_role"] for item in view["evidence"]} == {"绪论", "结论"}
    assert "coverage=complete" in request.system and "不能判缺失" in request.system
    # 开了范围，扣分规则就允许“证据不足”（分档规则仍必须选档）。
    deduct = build_v4_envelope(
        _thesis_envelope(_scoped_rule("cross_section", ["绪论", "结论"], **{**CROSS_RULE_TEXT, **DEDUCT_RULE})),
        top_k=3, judge_scope=True, **BUDGET)
    assert "insufficient_evidence" in core_view.build_core_request(deduct).schema["properties"]["status"]["enum"]


def test_view_marks_a_partial_chapter(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    envelope = _thesis_envelope(_scoped_rule("section", ["系统设计"]))
    value = build_v4_envelope(envelope, top_k=1, judge_scope=True, **BUDGET).to_mapping()
    value["evidence_units"] = value["evidence_units"][:1]
    value["evidence_selection_identity"]["section_coverage"] = [
        {**item, "selected_unit_count": 1, "complete": False} if item["section_path"] == ["第3章 系统设计"] else item
        for item in value["evidence_selection_identity"]["section_coverage"]
    ]
    assert judge_scope.scope_view([value["atomic_rule_snapshot"]], value)["sections"][0]["coverage"] == "partial"


def test_view_is_unchanged_while_the_scope_switch_is_off():
    envelope = build_v4_envelope(_paper_envelope(), top_k=3, **BUDGET)
    request = core_view.build_core_request(envelope)
    view = json.loads(request.user)

    assert "scope" not in view
    assert all("›" not in item["section"] and "section_role" not in item for item in view["evidence"])
    assert request.system == core_view._SINGLE_INSTRUCTIONS
    assert core_view.provider_prompt_suffix() == ""


# ── 分期 C：证据不足 ────────────────────────────────────────────────


def _insufficient_response():
    return {
        "schema_version": "semantic-rule-response@2",
        "rule_code": SEMANTIC_RULE_CODE,
        "status": "insufficient_evidence",
        "level_code": None,
        "occurrences": [],
    }


def test_insufficient_evidence_is_rejected_while_the_mode_is_off():
    envelope = build_v4_envelope(_envelope(), top_k=3, **BUDGET)
    request = core_view.build_core_request(envelope)

    assert request.schema["properties"]["status"]["enum"] == [
        "triggered",
        "not_triggered",
        "not_applicable",
    ]
    with pytest.raises(ValueError, match="status is invalid"):
        core_view.decode_core_response(envelope, _insufficient_response(), request)


DEDUCT_RULE = {
    "direction": "deduct",
    "levels": [],
    "max_points": "2",
    "repeat_policy": "once",
    "name": "研究方法描述不清",
    "rule_text": "没有说明数据来源与分析步骤",
}


def test_review_mode_offers_and_decodes_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_INSUFFICIENT_EVIDENCE_MODE", "review")
    envelope = build_v4_envelope(_envelope(rule=DEDUCT_RULE), top_k=3, **BUDGET)
    request = core_view.build_core_request(envelope)

    assert "insufficient_evidence" in request.schema["properties"]["status"]["enum"]
    assert "insufficient_evidence" in request.system
    assert "或现有证据不足以证明缺陷存在" not in request.system
    assert "是否已说明原因、限定范围或给出替代做法" in request.system
    assert core_view.provider_prompt_suffix() == "+insufficient@1"
    decoded = core_view.decode_core_response(envelope, _insufficient_response(), request)
    assert decoded["status"] == "insufficient_evidence"
    assert decoded["occurrences"] == []


def test_insufficient_evidence_scores_zero_and_routes_the_criterion_to_review():
    criterion = _criterion()
    rule = _deduct_rule()
    request = _request_for((criterion, rule))
    result, _registry, _runtime = _execute(
        request,
        semantic_responses={rule["rule_code"]: _semantic_response(rule["rule_code"], status="insufficient_evidence")},
    )

    decision = _decision(result)
    assert decision["status"] == "not_triggered"
    outcome = _criterion_outcome(result)
    assert outcome["status"] == "review_required"
    assert outcome["auto_score"] == "100"
    assert outcome["final_score"] is None
    assert result["status"] == "review_required"
    issues = [item for item in result["review_issues"] if item["code"] == EVIDENCE_INSUFFICIENT_CODE]
    assert issues == [
        {
            "code": EVIDENCE_INSUFFICIENT_CODE,
            "severity": "review",
            "criterion_code": criterion["criterion_code"],
            "rule_code": rule["rule_code"],
            "message": "evidence does not cover what the rule checks",
        }
    ]


def test_band_rules_must_choose_a_level_even_in_review_mode(monkeypatch):
    # A band criterion without an automatic score is stored blocked, which a
    # person cannot override; so band requests never offer the status.
    monkeypatch.setattr(settings, "SCORING_INSUFFICIENT_EVIDENCE_MODE", "review")
    envelope = build_v4_envelope(_envelope(), top_k=3, **BUDGET)
    request = core_view.build_core_request(envelope)
    assert "insufficient_evidence" not in request.schema["properties"]["status"]["enum"]
    with pytest.raises(ValueError, match="status is invalid"):
        core_view.decode_core_response(envelope, _insufficient_response(), request)

    criterion = _criterion(mode="band")
    rule = _band_rule()
    executed, _registry, _runtime = _execute(
        _request_for((criterion, rule)),
        semantic_responses={rule["rule_code"]: _semantic_response(rule["rule_code"], status="insufficient_evidence")},
    )
    assert _decision(executed)["status"] == "invalid"
    assert _criterion_outcome(executed)["status"] == "invalid"


def test_insufficient_evidence_with_a_finding_is_invalid():
    criterion = _criterion()
    rule = _deduct_rule()
    request = _request_for((criterion, rule))
    response = _semantic_response(rule["rule_code"], status="insufficient_evidence")
    response["occurrences"] = [{"finding_code": "GAP", "evidence": [], "locator": {}}]
    result, _registry, _runtime = _execute(
        request, semantic_responses={rule["rule_code"]: response}
    )

    assert _decision(result)["status"] == "invalid"
    assert not [item for item in result["review_issues"] if item["code"] == EVIDENCE_INSUFFICIENT_CODE]


def test_prompt_suffix_covers_both_switches(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    monkeypatch.setattr(settings, "SCORING_INSUFFICIENT_EVIDENCE_MODE", "review")
    assert core_view.provider_prompt_suffix() == "+rule-scope@1+insufficient@1"
    group = core_view._group_instructions()
    assert group.endswith(core_view._GROUP_INSTRUCTIONS[len(core_view._COMMON_INSTRUCTIONS):])
    assert deepcopy(core_view._statuses())[-1] == "insufficient_evidence"


# ── 分期 D：规则体检报表 ────────────────────────────────────────────


def _task(rule_code, decision, *, item="item", status="succeeded", criterion="T01"):
    from types import SimpleNamespace

    return SimpleNamespace(
        rule_code=rule_code,
        criterion_code=criterion,
        status=status,
        score_item_id=item,
        result_snapshot=None if decision is None else {"status": decision},
    )


def test_rule_health_flags_rules_that_never_discriminate():
    from backend.app.services.calibration.analytics import summarize_rule_health

    runs = [
        [_task("R1", "triggered", item="i%d" % index), _task("R2", "triggered" if index % 2 else "not_triggered", item="j%d" % index, criterion="T02")]
        for index in range(5)
    ]
    report = summarize_rule_health(runs)
    rows = {row["rule_code"]: row for row in report["rules"]}

    assert report["papers"] == 5
    assert report["zero_discrimination"] == ["R1"]
    assert rows["R1"]["decided"] == 5 and rows["R1"]["triggered"] == 5
    assert rows["R2"]["zero_discrimination"] is False


def test_rule_health_needs_enough_papers_before_judging():
    from backend.app.services.calibration.analytics import summarize_rule_health

    runs = [[_task("R1", "triggered", item="i%d" % index)] for index in range(4)]
    assert summarize_rule_health(runs)["zero_discrimination"] == []


def test_rule_health_counts_insufficient_evidence_and_failures_separately():
    from backend.app.services.calibration.analytics import summarize_rule_health

    runs = [
        [_task("R1", "not_triggered", status="review_required")],
        [_task("R1", "not_triggered")],
        [_task("R1", None, status="failed_exhausted")],
        [_task("R1", "skipped", status="skipped")],
    ]
    row = summarize_rule_health(runs)["rules"][0]

    assert (row["insufficient"], row["not_triggered"], row["invalid"], row["skipped"]) == (1, 1, 1, 1)
    assert row["insufficient_rate"] == 0.25
    assert row["decided"] == 1


def test_rule_health_lists_rules_that_trigger_together():
    from backend.app.services.calibration.analytics import summarize_rule_health

    runs = []
    for index in range(4):
        runs.append(
            [
                _task("A", "triggered", item="a%d" % index),
                _task("B", "triggered", item="b%d" % index, criterion="T02"),
                _task("C", "triggered" if index == 0 else "not_triggered", item="c%d" % index),
            ]
        )
    report = summarize_rule_health(runs)

    assert report["co_triggered"] == [
        {"rule_codes": ["A", "B"], "same_criterion": False, "both": 4, "ratio": 1.0}
    ]


def test_rule_health_counts_raised_scores_only_when_one_rule_triggered():
    from backend.app.services.calibration.analytics import summarize_rule_health

    runs = [
        [_task("A", "triggered", item="x1")],
        [_task("A", "triggered", item="x2"), _task("B", "triggered", item="x2")],
    ]
    rows = {row["rule_code"]: row for row in summarize_rule_health(runs, {"x1", "x2"})["rules"]}

    assert rows["A"]["triggered_alone"] == 1
    assert rows["A"]["raised_after_alone"] == 1
    assert rows["B"]["triggered_alone"] == 0


def test_rule_health_report_script_rejects_missing_arguments(capsys):
    from backend.app.scripts.rule_health_report import main

    assert main([]) == 2
    assert "用法" in capsys.readouterr().err


def test_rule_health_reports_unknown_batch(client):
    from backend.app.db.session import get_db
    from backend.app.main import app
    from backend.app.services.calibration.analytics import rule_health

    db = next(app.dependency_overrides[get_db]())
    try:
        with pytest.raises(ValueError, match="batch not found"):
            rule_health(db, "missing")
    finally:
        db.close()

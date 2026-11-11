"""第二阶段第二轮：快照 @4、人工指定判定范围、复核原因（docs/论文评分第二阶段改造方案.md 第 13 节）。

全部用合成数据、内存 SQLite 与假运行时，不调用真实模型。
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.db import models
from backend.app.services.rubrics.review_workspace import content_token
from backend.app.services.scoring.adapters.persistence import CoreRunPersistence
from backend.app.services.scoring.adapters.rubric_snapshot import CompiledRubricSnapshotLoader
from backend.app.services.scoring.core import engine as core_engine
from backend.app.services.scoring.core.contracts import AtomicRuleSnapshot
from backend.app.services.scoring.retrieval import judge_scope
from backend.app.services.scoring.review_reasons import from_core_issues
from backend.app.tests.test_m4_execution_persistence import _DocumentSnapshotStore
from backend.app.tests.test_m4_execution_persistence import _database_graph
from backend.app.tests.test_m4_rule_executor import _CheckerRegistry
from backend.app.tests.test_m4_rule_executor import _SemanticRuntime
from backend.app.tests.test_m4_rule_executor import _TechnicalProposalProfile
from backend.app.tests.test_m4_rule_executor import _criterion
from backend.app.tests.test_m4_rule_executor import _deduct_rule
from backend.app.tests.test_m4_rule_executor import _none_rule
from backend.app.tests.test_m4_rule_executor import _plain
from backend.app.tests.test_m4_rule_executor import _quote_occurrence
from backend.app.tests.test_m4_rule_executor import _request_for
from backend.app.tests.test_m4_rule_executor import _semantic_response


def _create(client):
    response = client.post("/api/rubrics", json={
        "name": "合成判定范围", "version": "v1", "total_score": 20,
        "criteria": [
            {"code": "T01", "name": "设计", "max_score": 10, "scoring_mode": "deductive",
             "deduction_rules_structured": [
                 # 手动录入的规则：名称是“评分项名 #序号”，条款正文取 reason。
                 {"match": "合成条款一", "points": 2, "reason": "数据库设计缺少说明"},
                 {"match": "合成条款二", "points": 3, "reason": "研究问题在结论中没有回应"},
             ]},
            {"code": "T02", "name": "规范", "max_score": 10, "scoring_mode": "deductive",
             "deduction_rules_structured": [
                 {"match": "合成条款三", "points": 1, "reason": "引用格式不统一"},
             ]},
        ],
    })
    assert response.status_code == 200, response.text
    return response.json()


def _workspace(client, rubric_id):
    response = client.get(f"/api/rubrics/{rubric_id}/review-workspace")
    assert response.status_code == 200, response.text
    return response.json()


def _publish(client, rubric_id):
    view = _workspace(client, rubric_id)
    for rule in view["rules"]:
        if rule["status"] in {"draft", "review"}:
            response = client.post(
                f"/api/rubrics/{rubric_id}/rules/{rule['rule_code']}/confirm",
                json={"compilation_id": view["compilation_id"], "rule_id": rule["id"],
                      "content_token": rule["content_token"], "reason": "核对合成条款"},
            )
            assert response.status_code == 200, response.text
    assert client.post(f"/api/rubrics/{rubric_id}/submit-review").status_code == 200
    published = client.post(f"/api/rubrics/{rubric_id}/publish", json={
        "compilation_id": view["compilation_id"], "reason": "合成发布"})
    assert published.status_code == 200, published.text


def _first_rule(session, rubric_id):
    return session.scalars(
        select(models.AtomicRule)
        .join(models.RubricVersion, models.AtomicRule.rubric_version_id == models.RubricVersion.id)
        .where(models.RubricVersion.rubric_id == rubric_id)
        .order_by(models.AtomicRule.rule_code)
    ).first()


def _load(client, rubric_id):
    with client.session_factory() as session:
        version = session.scalar(
            select(models.RubricVersion).where(models.RubricVersion.rubric_id == rubric_id)
        )
        return CompiledRubricSnapshotLoader().load_from_session(
            session=session,
            rubric_version_id=version.id,
            expected_profile_key=version.business_profile_key,
        ).to_mapping()


# ── 快照 @5（docs/评分细则适用范围方案.md 第 8 节） ──────────────────


def test_snapshot_carries_scope_only_while_the_switch_is_on(client, monkeypatch):
    rubric = _create(client)
    with client.session_factory() as session:
        rule = _first_rule(session, rubric["id"])
        # 旧的人工字段不再进快照：只认模型给出的 scope。
        rule.applies_to = "第3章 系统设计"
        rule.evidence_policy = {**(rule.evidence_policy or {}), "judge_scope": "section"}
        rule.scope = {"type": "section", "sections": ["系统设计"], "source": "ai_draft", "vocabulary": "section-roles@1"}
        code = rule.rule_code
        session.commit()
    _publish(client, rubric["id"])

    off = {item["rule_code"]: item for item in _load(client, rubric["id"])["atomic_rules"]}
    assert off[code]["schema_version"] == "atomic-rule-snapshot@3"
    assert "scope" not in off[code] and "applies_to" not in off[code]
    assert "judge_scope" not in off[code]["evidence_policy"]

    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    on_snapshot = _load(client, rubric["id"])
    on = {item["rule_code"]: item for item in on_snapshot["atomic_rules"]}
    assert on[code]["schema_version"] == "atomic-rule-snapshot@5"
    assert on[code]["scope"] == {"type": "section", "sections": ["系统设计"]}
    assert "judge_scope" not in on[code]["evidence_policy"] and "applies_to" not in on[code]
    # 没有范围的规则在身份里明确记为全文（方案 Q5）。
    assert {item["scope"]["type"] for code_, item in on.items() if code_ != code} == {"document"}
    assert judge_scope.rule_scope(on[code]) == {"scope": "section", "targets": ["系统设计"]}
    assert on_snapshot["version_hash"] == _load(client, rubric["id"])["version_hash"]
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", False)
    assert _load(client, rubric["id"])["rubric_snapshot_hash"] != on_snapshot["rubric_snapshot_hash"]


def _snapshot_value(**overrides):
    value = {
        "schema_version": "atomic-rule-snapshot@5",
        "rule_code": "R1", "criterion_code": "T01", "direction": "deduct", "effect_type": "score",
        "judge_type": "semantic", "checker_key": None, "checker_version": None, "checker_params": {},
        "evidence_policy": {"mode": "source_quote", "requirement": "required", "minimum_coverage": "1",
                            "allowed_finding_codes": ["R1"]},
        "max_points": "2", "repeat_policy": "once", "cap_points": None, "depends_on_rule_codes": [],
        "mutex_group": None, "levels": [], "name": "规则", "rule_text": "规则", "positive_example": None,
        "negative_example": None, "boundary_example": None, "context_needs": [],
        "scope": {"type": "cross_section", "sections": ["绪论", "结论"]},
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize("scope", [
    {"type": "chapter", "sections": []},
    {"type": "section", "sections": []},
    {"type": "document", "sections": ["绪论"]},
    {"type": "cross_section", "sections": ["绪论", "绪论"]},
    {"type": "section", "sections": ["绪论"], "source": "ai_draft"},
])
def test_snapshot_contract_checks_the_scope_shape(scope):
    with pytest.raises(ValueError):
        AtomicRuleSnapshot.from_mapping(_snapshot_value(scope=scope))


def test_snapshot_contract_accepts_scope_only_on_at5():
    assert AtomicRuleSnapshot.from_mapping(_snapshot_value()).to_mapping()["scope"]["sections"] == ["绪论", "结论"]
    with pytest.raises(ValueError):
        AtomicRuleSnapshot.from_mapping(_snapshot_value(schema_version="atomic-rule-snapshot@3"))
    with pytest.raises(ValueError):
        AtomicRuleSnapshot.from_mapping(_snapshot_value(schema_version="atomic-rule-snapshot@4"))
    legacy = _snapshot_value()
    legacy["evidence_policy"]["judge_scope"] = "section"
    with pytest.raises(ValueError):
        AtomicRuleSnapshot.from_mapping(legacy)


# ── 页面 ────────────────────────────────────────────────────────────


def test_review_workspace_shows_stored_scope_without_changing_the_content_token(client):
    # 适用范围改由模型给出（docs/评分细则适用范围方案.md）：工作区只读显示库里的范围，空即全文。
    rubric = _create(client)
    rules = _workspace(client, rubric["id"])["rules"]
    assert {rule["scope_label"] for rule in rules} == {"全文"}
    assert all(rule["scope"] is None and "judge_scope" not in rule for rule in rules)
    with client.session_factory() as session:
        row = session.scalar(select(models.AtomicRule).where(models.AtomicRule.id == rules[0]["id"]))
        row.scope = {"type": "section", "sections": ["绪论"], "source": "ai_identified", "vocabulary": "section-roles@1"}
        session.commit()
    shown = next(rule for rule in _workspace(client, rubric["id"])["rules"] if rule["id"] == rules[0]["id"])
    assert shown["scope"]["sections"] == ["绪论"] and shown["scope_label"] == "章节内：绪论"
    with client.session_factory() as session:
        row = session.scalar(select(models.AtomicRule).where(models.AtomicRule.id == rules[0]["id"]))
        assert shown["content_token"] == content_token(row)


def test_rule_edit_cannot_set_the_scope(client):
    rubric = _create(client)
    rule = _workspace(client, rubric["id"])["rules"][0]
    policy = dict(rule["evidence_policy"] or {})

    rejected = client.patch(f"/api/rubrics/{rubric['id']}/rules/{rule['rule_code']}",
                            json={"changes": {"applies_to": "绪论"}, "reason": "合成修改"})
    assert rejected.status_code == 400 and "applies_to" in rejected.json()["detail"], rejected.text

    ok = client.patch(f"/api/rubrics/{rubric['id']}/rules/{rule['rule_code']}",
                      json={"changes": {"evidence_policy": {**policy, "judge_scope": "section"}}, "reason": "合成修改"})
    assert ok.status_code == 200, ok.text
    edited = next(item for item in _workspace(client, rubric["id"])["rules"] if item["rule_code"] == rule["rule_code"])
    assert "judge_scope" not in edited["evidence_policy"]
    assert edited["scope"] is None


# ── 复核原因 ────────────────────────────────────────────────────────


def test_core_issues_become_structured_review_reasons():
    reasons = from_core_issues(
        [
            {"code": "REVIEW_REQUIRED", "severity": "review", "criterion_code": "T01", "rule_code": "R2", "message": "x"},
            {"code": "RULE_EVIDENCE_INSUFFICIENT", "severity": "review", "criterion_code": "T01", "rule_code": "R1", "message": "x"},
            {"code": "REPORT_ONLY", "severity": "info", "criterion_code": "T01", "rule_code": "R3", "message": "x"},
        ],
        rule_names={"R1": "研究方法描述不清"},
    )
    assert [(item["code"], item["rule_code"]) for item in reasons] == [
        ("evidence_insufficient", "R1"),
        ("rule_requires_review", "R2"),
    ]
    assert "研究方法描述不清" in reasons[0]["message"]
    assert "R2" in reasons[1]["message"]


def test_core_persistence_keeps_the_auto_score_and_explains_insufficient_evidence():
    criterion_snapshot = _criterion()
    rule = _deduct_rule()
    request = _request_for((criterion_snapshot, rule))
    outcome = core_engine.score_submission(
        request=deepcopy(request),
        checker_registry=_CheckerRegistry(),
        llm_runtime=_SemanticRuntime(
            {rule["rule_code"]: _semantic_response(rule["rule_code"], status="insufficient_evidence")}
        ),
        profile=_TechnicalProposalProfile(),
    )
    assert _plain(outcome)["status"] == "review_required"

    engine, db, rubric, criterion, paper = _database_graph(request)
    try:
        store = _DocumentSnapshotStore()
        run = CoreRunPersistence(db, document_snapshot_store=store).persist(
            request=deepcopy(request),
            outcome=outcome,
            paper_id=paper.id,
            rubric_id=rubric.id,
            criterion_id_by_code={criterion.code: criterion.id},
            workflow_profile="template_driven",
            document_snapshot_ref=store.put(request["document"]),
        )
        item = run.items[0]
        # 自动分保留、最终分留空：普通改分可以处理，不会变成改不了分的 blocked。
        assert item.auto_score_status == "calculated"
        assert item.ai_score is not None and item.final_score is None
        assert item.need_manual_review is True
        assert [reason["code"] for reason in item.review_reasons] == ["evidence_insufficient"]
        assert item.review_reasons[0]["rule_code"] == rule["rule_code"]
        assert "证据没有覆盖" in item.review_reason
    finally:
        db.close()
        engine.dispose()


def _persist_single(request, responses):
    outcome = core_engine.score_submission(
        request=deepcopy(request),
        checker_registry=_CheckerRegistry(),
        llm_runtime=_SemanticRuntime(responses),
        profile=_TechnicalProposalProfile(),
    )
    engine, db, rubric, criterion, paper = _database_graph(request)
    try:
        store = _DocumentSnapshotStore()
        run = CoreRunPersistence(db, document_snapshot_store=store).persist(
            request=deepcopy(request),
            outcome=outcome,
            paper_id=paper.id,
            rubric_id=rubric.id,
            criterion_id_by_code={criterion.code: criterion.id},
            workflow_profile="template_driven",
            document_snapshot_ref=store.put(request["document"]),
        )
        item = run.items[0]
        return item.review_reasons, item.review_reason
    finally:
        db.close()
        engine.dispose()


def test_core_persistence_explains_review_rules_and_leaves_plain_items_without_reasons():
    rule = _none_rule(code="proposal.review.v1", effect_type="review")
    request = _request_for((_criterion(mode="review_only"), rule))
    reasons, text = _persist_single(
        request,
        {rule["rule_code"]: _semantic_response(rule["rule_code"], occurrences=[_quote_occurrence(request)])},
    )
    assert [reason["code"] for reason in reasons] == ["rule_requires_review"]
    assert "要求人工复核" in text

    plain = _deduct_rule()
    reasons, text = _persist_single(
        _request_for((_criterion(), plain)),
        {plain["rule_code"]: _semantic_response(plain["rule_code"], status="not_triggered")},
    )
    assert reasons is None and text is None


def test_section_scope_selects_only_that_section(monkeypatch):
    from backend.app.services.llm import core_view
    from backend.app.services.scoring.retrieval.selection import build_v4_envelope
    from backend.app.tests.test_scoring_stage2 import BUDGET
    from backend.app.tests.test_scoring_stage2 import _thesis_envelope
    import json

    rule = {
        "schema_version": "atomic-rule-snapshot@5",
        "scope": {"type": "section", "sections": ["系统设计"]},
        "name": "数据库设计说明不足",
        "rule_text": "没有给出表结构与字段含义",
    }
    envelope = _thesis_envelope(rule)
    value = build_v4_envelope(envelope, top_k=12, judge_scope=True, **BUDGET).to_mapping()

    assert value["coverage_mode"] == "section"
    assert {unit["section_path"][0] for unit in value["evidence_units"]} == {"第3章 系统设计"}

    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    view = json.loads(core_view.build_core_request(build_v4_envelope(envelope, top_k=12, judge_scope=True, **BUDGET)).user)
    assert view["scope"] == {"type": "section", "sections": [
        {"role": "系统设计", "headings": ["第3章 系统设计"], "coverage": "complete"}]}
    assert view["rule"]["name"] == "数据库设计说明不足"

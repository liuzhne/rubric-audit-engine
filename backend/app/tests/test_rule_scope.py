"""评分细则适用范围：词表、起草返回、编译落库、范围识别任务（docs/评分细则适用范围方案.md）。

全部用合成数据与假模型，不调用真实模型。
"""

from __future__ import annotations

import importlib

import pytest
from sqlalchemy import select
from sqlalchemy import update

from backend.app.db import models
from backend.app.services.rubric_import.ai_rule_drafter import AIRuleDraftValidationError
from backend.app.services.rubric_import.ai_rule_drafter import _draft_output_schema
from backend.app.services.rubric_import.ai_rule_drafter import _lean_draft_output_schema
from backend.app.services.rubric_import.ai_rule_drafter import validate_ai_rule_draft
from backend.app.services.rubric_scope import ScopeError
from backend.app.services.rubric_scope import describe_scope
from backend.app.services.rubric_scope import normalize_scope
from backend.app.services.rubric_scope import role_names
from backend.app.services.rubric_scope import vocabulary_payload
from backend.app.services.rubrics import lifecycle


service = importlib.import_module("backend.app.services.ai_tasks.service")
runner = importlib.import_module("backend.app.services.work_queue.runner")

FP = "c" * 64


# --- 词表与校验 ---------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ({"type": "document", "sections": []}, {"type": "document", "sections": []}),
    ({"type": "document"}, {"type": "document", "sections": []}),
    ({"type": "section", "sections": [" 绪论 "]}, {"type": "section", "sections": ["绪论"]}),
    ({"type": "cross_section", "sections": ["绪论", "结论"]}, {"type": "cross_section", "sections": ["绪论", "结论"]}),
])
def test_normalize_scope_accepts_valid_shapes(value, expected):
    assert normalize_scope(value, "thesis") == expected


@pytest.mark.parametrize("value", [
    None, "绪论", {"type": "chapter", "sections": []},
    {"type": "document", "sections": ["绪论"]},
    {"type": "section", "sections": []}, {"type": "section", "sections": ["绪论", "结论"]},
    {"type": "cross_section", "sections": ["绪论"]},
    {"type": "cross_section", "sections": ["绪论", "绪论"]},
    {"type": "cross_section", "sections": ["摘要", "绪论", "结论", "致谢"]},
    {"type": "section", "sections": ["第三章"]},
])
def test_normalize_scope_rejects_invalid_shapes(value):
    with pytest.raises(ScopeError):
        normalize_scope(value, "thesis")


def test_vocabulary_is_per_profile():
    assert "系统设计" in role_names("thesis")
    assert "总体方案" in role_names("technical_proposal") and "系统设计" not in role_names("technical_proposal")
    assert role_names("unknown-profile") == role_names("thesis")
    with pytest.raises(ScopeError):
        normalize_scope({"type": "section", "sections": ["总体方案"]}, "thesis")
    payload = vocabulary_payload("general_document")
    assert payload["version"] == "section-roles@1" and {item["name"] for item in payload["sections"]} >= {"引言", "结论"}


def test_describe_scope():
    assert describe_scope(None) == "全文"
    assert describe_scope({"type": "document", "sections": []}) == "全文"
    assert describe_scope({"type": "cross_section", "sections": ["绪论", "结论"]}) == "跨章节：绪论、结论"
    assert describe_scope({"type": "bogus"}) == "全文"


# --- 起草返回范围 ---------------------------------------------------------------

def _criterion():
    return {"code": "T02", "name": "合成评分项", "max_score": 10, "description": "合成说明", "deduction_rules": []}


def _draft(scope=..., *, metadata=None):
    group = {"group_code": "G1", "issue": "合成问题", "mutex_group": "G1-SEVERITY", "cap_points": 3,
             "rules": [{"severity": "minor", "trigger": "合成条件", "points": 1, "reason": "合成原因",
                        "repeat_policy": "once", "source": "ai_inferred", "source_refs": ["/criterion/description"]}]}
    if scope is not ...:
        group["scope"] = scope
    return {"schema_version": "ai-deduction-draft@1", "criterion_code": "T02", "input_assessment": {},
            "rule_groups": [group],
            "generation_metadata": metadata if metadata is not None else
            {"business_profile_key": "thesis", "scope_vocabulary": "section-roles@1"}}


def test_strict_schema_enumerates_scope_and_lean_keeps_the_type_enum():
    strict = _draft_output_schema(maximum=10, allowed_source_refs=["S1"], business_profile_key="technical_proposal")
    scope = strict["properties"]["rule_groups"]["items"]["properties"]["scope"]
    assert scope["properties"]["sections"]["items"]["enum"] == role_names("technical_proposal")
    assert "scope" in strict["properties"]["rule_groups"]["items"]["required"]
    lean = _lean_draft_output_schema()["properties"]["rule_groups"]["items"]["properties"]["scope"]
    assert lean["properties"]["type"]["enum"] == ["document", "section", "cross_section"]
    assert "enum" not in lean["properties"]["sections"]["items"]


def test_validator_normalizes_scope_and_defaults_a_missing_one_to_document():
    checked = validate_ai_rule_draft(_draft({"type": "section", "sections": ["绪论 "]}), criterion=_criterion())
    assert checked["rule_groups"][0]["scope"] == {"type": "section", "sections": ["绪论"]}
    missing = validate_ai_rule_draft(_draft(), criterion=_criterion())
    assert missing["rule_groups"][0]["scope"] == {"type": "document", "sections": []}


def test_validator_rejects_an_invalid_scope():
    with pytest.raises(AIRuleDraftValidationError) as caught:
        validate_ai_rule_draft(_draft({"type": "section", "sections": ["第三章"]}), criterion=_criterion())
    assert caught.value.code == "AI_DRAFT_OUTPUT_INVALID"


def test_drafts_from_before_the_vocabulary_pass_untouched():
    old = validate_ai_rule_draft(_draft(metadata={}), criterion=_criterion())
    assert "scope" not in old["rule_groups"][0]


# --- 编译落库 -----------------------------------------------------------------

def _ai_row(index, scope, *, source="ai_inferred"):
    return {"match": f"合成条件{index}", "trigger": f"合成条件{index}", "points": index + 1,
            "reason": f"合成条款{index}", "repeat_policy": "once", "mutex_group": "T04-B01-G01-SEVERITY",
            "source": source, "confirmed": True, "scope": scope,
            "generation_fingerprint": FP, "draft_row_key": f"T04::T04-B01-G01::{index}"}


def _create_rubric(client, rows, *, name="合成适用范围"):
    response = client.post("/api/rubrics", json={"name": name, "version": "v1", "total_score": 10,
        "criteria": [{"code": "T04", "name": "合成格式", "max_score": 10, "scoring_mode": "deductive",
                      "deduction_rules_structured": rows}]})
    assert response.status_code == 200, response.text
    return response.json()


def _rules(client, rubric_id):
    with client.session_factory() as session:
        version = session.scalars(select(models.RubricVersion).where(
            models.RubricVersion.rubric_id == rubric_id)).one()
        return {rule.rule_code: rule.scope for rule in session.scalars(select(models.AtomicRule).where(
            models.AtomicRule.rubric_version_id == version.id))}


def test_ai_rows_compile_their_scope_and_manual_rows_never_do(client):
    section = {"type": "section", "sections": ["绪论"]}
    manual = {"match": "人工条件", "points": 1, "reason": "人工条款", "scope": section}
    rubric = _create_rubric(client, [_ai_row(0, section), _ai_row(1, section),
                                     _ai_row(2, {"type": "section", "sections": ["第三章"]}), manual])
    scopes = _rules(client, rubric["id"])
    assert scopes["manual.t04.deduct.1.v1"] == {**section, "source": "ai_draft", "vocabulary": "section-roles@1"}
    assert scopes["manual.t04.deduct.2.v1"]["sections"] == ["绪论"]
    assert scopes["manual.t04.deduct.3.v1"] is None  # 范围不合格：按全文
    assert scopes["manual.t04.deduct.4.v1"] is None  # 人工行：范围不由人指定


def test_version_hash_ignores_an_empty_scope(client):
    rubric = _create_rubric(client, [_ai_row(0, None), _ai_row(1, None)], name="合成哈希")
    with client.session_factory() as session:
        version = session.scalars(select(models.RubricVersion).where(
            models.RubricVersion.rubric_id == rubric["id"])).one()
        record = session.get(models.Rubric, rubric["id"])
        compilation = session.get(models.RubricCompilation, version.compilation_id)
        before = lifecycle._canonical_version_hash(session, record, compilation, version)
        rule = session.scalars(select(models.AtomicRule).where(models.AtomicRule.rubric_version_id == version.id)).first()
        rule.scope = {"type": "section", "sections": ["绪论"], "source": "ai_identified", "vocabulary": "section-roles@1"}
        session.flush()
        scoped = lifecycle._canonical_version_hash(session, record, compilation, version)
        rule.scope = None
        session.flush()
        cleared = lifecycle._canonical_version_hash(session, record, compilation, version)
        session.rollback()
    assert scoped != before and cleared == before


# --- 范围识别任务 ---------------------------------------------------------------

class FakeScopeModel:
    provider = "openai_compatible"
    model_name = "scope-task-model"
    model_version = "test-v1"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete_json(self, instructions, payload, response_schema=None, default_max_tokens=None,
                      attempts_limit=None, rate_limit_retries=None, default_timeout_seconds=None):
        self.calls.append({"instructions": instructions, "payload": payload, "schema": response_schema})
        result = self.script.pop(0)
        if callable(result):
            result = result(payload)
        return result


def _manual_row(index):
    return {"match": f"人工条件{index}", "points": index + 1, "reason": f"人工条款{index}"}


def _drain(client):
    count = 0
    while runner.execute_next(client.session_factory) is not None:
        count += 1
        assert count < 50


def _run_scope_task(client, monkeypatch, rubric_id, script):
    fake = FakeScopeModel(script)
    monkeypatch.setattr(service, "get_llm_scorer", lambda *args, **kwargs: fake)
    created = client.post(f"/api/rubrics/{rubric_id}/ai-tasks", json={"kind": "rule_scope", "params": {}})
    assert created.status_code in (200, 202), created.text
    _drain(client)
    task = client.get(f"/api/ai-tasks/{created.json()['id']}")
    assert task.status_code == 200, task.text
    return task.json(), fake


def test_scope_task_fills_only_rules_without_a_scope(client, monkeypatch):
    drafted = {"type": "document", "sections": []}
    rubric = _create_rubric(client, [_manual_row(0), _manual_row(1), _ai_row(2, drafted), _ai_row(3, drafted)],
                            name="合成范围识别")

    def answer(payload):
        codes = [rule["rule_code"] for rule in payload["rules"]]
        assert codes == ["manual.t04.deduct.1.v1", "manual.t04.deduct.2.v1"]
        return {"rules": [
            {"rule_code": codes[0], "scope": {"type": "cross_section", "sections": ["绪论", "结论"]}},
            {"rule_code": codes[1], "scope": {"type": "section", "sections": ["不存在的章"]}},
            {"rule_code": "manual.t99.deduct.1.v1", "scope": {"type": "document", "sections": []}},
        ]}

    task, fake = _run_scope_task(client, monkeypatch, rubric["id"], [answer])
    assert task["status"] == "succeeded", task
    assert task["result"]["identified"] == 1 and task["result"]["discarded"] == 2
    assert task["result"]["missing"] == ["manual.t04.deduct.2.v1"]
    schema_codes = fake.calls[0]["schema"]["properties"]["rules"]["items"]["properties"]["rule_code"]["enum"]
    assert schema_codes == ["manual.t04.deduct.1.v1", "manual.t04.deduct.2.v1"]
    assert "绪论" in {item["name"] for item in fake.calls[0]["payload"]["scope_vocabulary"]["sections"]}

    scopes = _rules(client, rubric["id"])
    assert scopes["manual.t04.deduct.1.v1"] == {"type": "cross_section", "sections": ["绪论", "结论"],
                                               "source": "ai_identified", "vocabulary": "section-roles@1"}
    assert scopes["manual.t04.deduct.2.v1"] is None
    assert scopes["manual.t04.deduct.3.v1"]["source"] == "ai_draft"
    with client.session_factory() as session:
        rule = session.scalars(select(models.AtomicRule).where(
            models.AtomicRule.rule_code == "manual.t04.deduct.1.v1",
            models.AtomicRule.criterion.has(models.RubricCriterion.rubric_id == rubric["id"]))).one()
        assert rule.status == "draft"
        events = [event for event in session.get(models.RubricCompilation, rule.rubric_version.compilation_id)
                  .human_changes if event.get("action") == "scope"]
        assert [event["rule_code"] for event in events] == ["manual.t04.deduct.1.v1"]


def test_scope_task_repairs_once_then_marks_the_criterion_failed(client, monkeypatch):
    rubric = _create_rubric(client, [_manual_row(0)], name="合成范围识别修正")
    task, fake = _run_scope_task(client, monkeypatch, rubric["id"], [{"oops": []}, {"still": "bad"}])
    assert task["status"] == "succeeded", task
    assert task["result"]["failed"] == ["T04"] and task["result"]["identified"] == 0
    assert len(fake.calls) == 2 and "上次输出未通过校验" in fake.calls[1]["instructions"]
    assert _rules(client, rubric["id"])["manual.t04.deduct.1.v1"] is None


def test_scope_task_with_nothing_to_identify_succeeds_without_calling_the_model(client, monkeypatch):
    drafted = {"type": "section", "sections": ["绪论"]}
    rubric = _create_rubric(client, [_ai_row(0, drafted), _ai_row(1, drafted)], name="合成范围已齐")
    task, fake = _run_scope_task(client, monkeypatch, rubric["id"], [])
    assert task["status"] == "succeeded" and task["result"]["status"] == "nothing_to_identify"
    assert fake.calls == []


def test_scope_task_refuses_a_rubric_that_is_not_a_draft(client, monkeypatch):
    rubric = _create_rubric(client, [_manual_row(0)], name="合成范围非草稿")
    with client.session_factory() as session:
        # 绕过 ORM 的状态转换校验，直接造一份“已提交审核”的评分标准。
        session.execute(update(models.Rubric).where(models.Rubric.id == rubric["id"]).values(status="review"))
        session.commit()
    monkeypatch.setattr(service, "get_llm_scorer", lambda *args, **kwargs: FakeScopeModel([]))
    response = client.post(f"/api/rubrics/{rubric['id']}/ai-tasks", json={"kind": "rule_scope", "params": {}})
    assert response.status_code == 409, response.text


def test_scope_task_treats_all_invalid_entries_as_invalid_output(client, monkeypatch):
    rubric = _create_rubric(client, [_manual_row(0)], name="合成范围全部无效")
    bad = {"rules": [{"rule_code": "manual.t04.deduct.1.v1", "scope": {"type": "section", "sections": ["第九章"]}}]}
    task, fake = _run_scope_task(client, monkeypatch, rubric["id"], [bad, bad])
    assert task["status"] == "succeeded", task
    assert len(fake.calls) == 2  # 全部不合格也带修正提示再问一次
    assert task["result"]["failed"] == ["T04"] and task["result"]["discarded"] == 1


def test_scope_write_back_refuses_a_superseded_draft(client):
    from backend.app.services.rubric_scope.identification import ScopeIdentificationError
    from backend.app.services.rubric_scope.identification import apply_identified_scopes

    rubric = _create_rubric(client, [_manual_row(0)], name="合成范围草稿已换")
    with client.session_factory() as session:
        old_version = session.scalars(select(models.RubricVersion).where(
            models.RubricVersion.rubric_id == rubric["id"])).one().id
    workspace = client.get(f"/api/rubrics/{rubric['id']}/review-workspace").json()
    criteria = client.get(f"/api/rubrics/{rubric['id']}").json()["criteria"]
    saved = client.post(f"/api/rubrics/{rubric['id']}/recompile", json={
        "supersedes_compilation_id": workspace["compilation_id"], "version": "v2", "criteria": criteria})
    assert saved.status_code == 200, saved.text
    with client.session_factory() as session:
        with pytest.raises(ScopeIdentificationError) as caught:
            apply_identified_scopes(session, rubric_id=rubric["id"], version_id=old_version, profile_key="thesis",
                                    scopes={"manual.t04.deduct.1.v1": {"type": "document", "sections": []}},
                                    actor_id=None)
        session.rollback()
    assert caught.value.code == "RUBRIC_VERSION_CHANGED"


def test_editing_the_evidence_policy_keeps_an_existing_judge_scope(client):
    rubric = _create_rubric(client, [_manual_row(0)], name="合成保留旧判定范围")
    with client.session_factory() as session:
        rule = session.scalars(select(models.AtomicRule).where(
            models.AtomicRule.criterion.has(models.RubricCriterion.rubric_id == rubric["id"]))).one()
        rule.evidence_policy = {**rule.evidence_policy, "judge_scope": "document"}
        code, policy = rule.rule_code, dict(rule.evidence_policy)
        session.commit()
    response = client.patch(f"/api/rubrics/{rubric['id']}/rules/{code}", json={
        "changes": {"evidence_policy": {**policy, "minimum_coverage": "1", "judge_scope": "section"}},
        "reason": "合成修改"})
    assert response.status_code == 200, response.text
    with client.session_factory() as session:
        rule = session.scalars(select(models.AtomicRule).where(models.AtomicRule.rule_code == code,
            models.AtomicRule.criterion.has(models.RubricCriterion.rubric_id == rubric["id"]))).one()
        assert rule.evidence_policy["judge_scope"] == "document"  # 已有的保留，新指定的不保存


def _recompile_same(client, rubric_id, *, version, edit=None):
    workspace = client.get(f"/api/rubrics/{rubric_id}/review-workspace").json()
    criteria = client.get(f"/api/rubrics/{rubric_id}").json()["criteria"]
    if edit is not None:
        edit(criteria)
    saved = client.post(f"/api/rubrics/{rubric_id}/recompile", json={
        "supersedes_compilation_id": workspace["compilation_id"], "version": version, "criteria": criteria})
    assert saved.status_code == 200, saved.text


def _current_scopes(client, rubric_id):
    workspace = client.get(f"/api/rubrics/{rubric_id}/review-workspace").json()
    return {rule["rule_code"]: rule["scope"] for rule in workspace["rules"]}


def test_identified_scope_survives_a_recompile_of_unchanged_rules(client):
    from backend.app.services.rubrics.review_workspace import content_token

    rubric = _create_rubric(client, [_manual_row(0), _manual_row(1)], name="合成范围随重新编译保留")
    identified = {"type": "section", "sections": ["绪论"], "source": "ai_identified", "vocabulary": "section-roles@1"}
    with client.session_factory() as session:
        rules = {rule.rule_code: rule for rule in session.scalars(select(models.AtomicRule).where(
            models.AtomicRule.criterion.has(models.RubricCriterion.rubric_id == rubric["id"])))}
        before = content_token(rules["manual.t04.deduct.1.v1"])
        for rule in rules.values():
            rule.scope = dict(identified)
        session.flush()
        # 范围不是老师编辑的内容：写上范围不让打开着的表单失效。
        assert content_token(rules["manual.t04.deduct.1.v1"]) == before
        session.commit()

    def change_second(criteria):
        criteria[0]["deduction_rules_structured"][1]["reason"] = "改过的合成条款"

    _recompile_same(client, rubric["id"], version="v2", edit=change_second)
    scopes = _current_scopes(client, rubric["id"])
    assert scopes["manual.t04.deduct.1.v1"]["sections"] == ["绪论"]  # 内容未变：范围带过去
    assert scopes["manual.t04.deduct.2.v1"] is None  # 内容改了：留给下一次识别

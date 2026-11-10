"""AI 起草细则的互斥组在编译时规整（AI起草互斥组规整与区间拆档方案 §4.1）。

单档的 AI 组不写互斥组，同名的两次起草不并成一组；评分项投影保留起草时的原名，
人工行原样，写错了照旧由发布校验挡住。
"""

import pytest

from backend.app.scripts import inspect_ai_draft_rules as script
from backend.app.services.rubric_import.compiler import point_range
from backend.app.services.rubric_import.deduction_caps import effective_mutex_groups
from backend.app.tests.test_m8_web_rubric_lifecycle import _import_rubric
from backend.app.tests.test_rubric_review_workspace import workspace

FP_A = "a" * 64
FP_B = "b" * 64
MUTEX = "T04-B01-G01-SEVERITY"


def ai_row(index, *, fingerprint=FP_A, mutex=MUTEX, group="T04-B01-G01", points=None, criterion="T04", **extra):
    return {
        "match": f"合成条件{index}", "trigger": f"合成条件{index}", "points": points or index + 1,
        "reason": f"合成条款{index}", "repeat_policy": "once", "mutex_group": mutex,
        "source": "ai_inferred", "confirmed": True,
        "severity": ["minor", "moderate", "severe"][index % 3],
        "generation_fingerprint": fingerprint, "draft_row_key": f"{criterion}::{group}::{index}",
        **extra,
    }


def manual_row(mutex, points=1):
    return {"match": "人工条件", "points": points, "reason": "人工条款", "mutex_group": mutex}


# --- 规整函数 ---------------------------------------------------------------

def test_single_compiled_row_drops_mutex():
    assert effective_mutex_groups([ai_row(0)], compiled={0}) == [None]


def test_group_with_two_compiled_rows_keeps_drafted_name():
    assert effective_mutex_groups([ai_row(0), ai_row(1)], compiled={0, 1}) == [MUTEX, MUTEX]


def test_rows_that_do_not_compile_are_not_counted():
    # 第二档未确认或分值无效时不会生成细则，剩下的那条就是单档。
    assert effective_mutex_groups([ai_row(0), ai_row(1)], compiled={0})[0] is None


def test_later_generation_with_same_name_gets_fingerprint_suffix():
    rows = [ai_row(0), ai_row(1), ai_row(0, fingerprint=FP_B), ai_row(1, fingerprint=FP_B)]
    assert effective_mutex_groups(rows, compiled={0, 1, 2, 3}) == [
        MUTEX, MUTEX, f"{MUTEX}-{FP_B[:8]}", f"{MUTEX}-{FP_B[:8]}"]


def test_lone_earlier_generation_does_not_take_the_name():
    rows = [ai_row(0), ai_row(0, fingerprint=FP_B), ai_row(1, fingerprint=FP_B)]
    assert effective_mutex_groups(rows, compiled={0, 1, 2}) == [None, MUTEX, MUTEX]


def test_ai_group_yields_to_manual_row_regardless_of_order():
    rows = [ai_row(0), ai_row(1), manual_row(MUTEX), manual_row(MUTEX)]
    result = effective_mutex_groups(rows, compiled={0, 1, 2, 3})
    assert result == [f"{MUTEX}-{FP_A[:8]}", f"{MUTEX}-{FP_A[:8]}", MUTEX, MUTEX]


@pytest.mark.parametrize("row", [
    manual_row("人工组"),
    {**ai_row(0), "generation_fingerprint": None},
    {**ai_row(0), "draft_row_key": None},
    {**ai_row(0), "source": "user_text"},
])
def test_manual_and_unattributable_rows_are_left_alone(row):
    assert effective_mutex_groups([row], compiled={0}) == [row["mutex_group"]]


# --- 编译（评分项投影重新编译） -------------------------------------------

def _create(client, rows, *, name="合成互斥组规整", max_score=10):
    response = client.post("/api/rubrics", json={"name": name, "version": "v1", "total_score": max_score,
        "criteria": [{"code": "T04", "name": "合成格式", "max_score": max_score,
                      "scoring_mode": "deductive", "deduction_rules_structured": rows}]})
    assert response.status_code == 200, response.text
    return response.json()


def _codes(view):
    return {item["code"] for item in view["structural_blockers"]}


def _mutex_by_code(view):
    return {rule["rule_code"]: rule["mutex_group"] for rule in view["rules"]}


def test_single_tier_ai_group_compiles_without_mutex_and_keeps_projection(client):
    rubric = _create(client, [ai_row(0)])
    view = workspace(client, rubric["id"])
    assert "mutex_definition_invalid" not in _codes(view)
    assert _mutex_by_code(view) == {"manual.t04.deduct.1.v1": None}
    projected = client.get(f"/api/rubrics/{rubric['id']}").json()["criteria"][0]["deduction_rules_structured"]
    assert projected[0]["mutex_group"] == MUTEX


def test_applying_remaining_tiers_later_restores_the_group(client):
    rubric = _create(client, [ai_row(0)])
    before = workspace(client, rubric["id"])
    criteria = client.get(f"/api/rubrics/{rubric['id']}").json()["criteria"]
    criteria[0]["deduction_rules_structured"] += [ai_row(1), ai_row(2)]
    saved = client.post(f"/api/rubrics/{rubric['id']}/recompile", json={
        "supersedes_compilation_id": before["compilation_id"], "version": "v2", "criteria": criteria})
    assert saved.status_code == 200, saved.text
    after = workspace(client, rubric["id"])
    assert set(_mutex_by_code(after).values()) == {MUTEX}
    assert "mutex_definition_invalid" not in _codes(after)


def test_two_generations_with_the_same_name_are_not_merged(client):
    rubric = _create(client, [ai_row(0), ai_row(1),
                              ai_row(0, fingerprint=FP_B, points=1.5), ai_row(1, fingerprint=FP_B, points=2.5)])
    view = workspace(client, rubric["id"])
    assert _mutex_by_code(view) == {
        "manual.t04.deduct.1.v1": MUTEX, "manual.t04.deduct.2.v1": MUTEX,
        "manual.t04.deduct.3.v1": f"{MUTEX}-{FP_B[:8]}", "manual.t04.deduct.4.v1": f"{MUTEX}-{FP_B[:8]}",
    }
    assert "mutex_definition_invalid" not in _codes(view)


def test_manual_single_member_mutex_still_blocks(client):
    rubric = _create(client, [manual_row("人工组", points=2)])
    view = workspace(client, rubric["id"])
    assert _mutex_by_code(view) == {"manual.t04.deduct.1.v1": "人工组"}
    assert "mutex_definition_invalid" in _codes(view)


# --- 追加路径（导入的评分标准） --------------------------------------------

def _recompile(client, rubric_id, rows, version, atomic_rules=None):
    before = workspace(client, rubric_id)
    criteria = client.get(f"/api/rubrics/{rubric_id}").json()["criteria"]
    criteria[0]["scoring_mode"] = "deductive"
    criteria[0]["deduction_rules_structured"] = rows
    payload = {"supersedes_compilation_id": before["compilation_id"], "version": version, "criteria": criteria}
    if atomic_rules is not None:
        payload["atomic_rules"] = atomic_rules(before)
    saved = client.post(f"/api/rubrics/{rubric_id}/recompile", json=payload)
    assert saved.status_code == 200, saved.text
    return workspace(client, rubric_id)


def _method_rows(count):
    return [ai_row(index, mutex="METHOD-B01-G01-SEVERITY", group="METHOD-B01-G01", criterion="METHOD")
            for index in range(count)]


def test_append_path_restores_group_for_previously_lone_rule(client):
    rubric = _import_rubric(client, name="合成追加规整")["rubric"]
    first = _recompile(client, rubric["id"], _method_rows(1), "v2")
    assert _mutex_by_code(first)["manual.method.deduct.1.v1"] is None
    after = _recompile(client, rubric["id"], _method_rows(3), "v3")
    mutex = _mutex_by_code(after)
    assert [mutex[f"manual.method.deduct.{n}.v1"] for n in (1, 2, 3)] == ["METHOD-B01-G01-SEVERITY"] * 3


def test_append_path_keeps_a_name_the_teacher_typed(client):
    rubric = _import_rubric(client, name="合成追加保留人工互斥组")["rubric"]
    _recompile(client, rubric["id"], _method_rows(1), "v2")

    def edit(view):
        return [{"id": rule["id"], "content_token": rule["content_token"],
                 "changes": {"mutex_group": "老师自定"} if rule["rule_code"] == "manual.method.deduct.1.v1" else {}}
                for rule in view["rules"]]

    edited = _recompile(client, rubric["id"], _method_rows(1), "v3", atomic_rules=edit)
    assert _mutex_by_code(edited)["manual.method.deduct.1.v1"] == "老师自定"
    after = _recompile(client, rubric["id"], _method_rows(3), "v4")
    mutex = _mutex_by_code(after)
    assert mutex["manual.method.deduct.1.v1"] == "老师自定"
    assert mutex["manual.method.deduct.2.v1"] == mutex["manual.method.deduct.3.v1"] == "METHOD-B01-G01-SEVERITY"


# --- 区间识别与核对脚本 ----------------------------------------------------

@pytest.mark.parametrize(("text", "expected"), [
    ("格式不规范扣1-3分", ("1", "3")),
    ("扣1～3分：格式不规范", ("1", "3")),
    ("扣 1 至 3 分", ("1", "3")),
    ("每处扣0.5-1分", ("0.5", "1")),
    ("A、B、C 各扣1-2分", ("1", "2")),
    ("字数少于8000-10000字扣2分", None),
    ("正文 3~5 页以内扣1分", None),
    ("答辩超时1-3分钟扣1分", None),
    ("缺少摘要扣2分", None),
    ("扣3-1分", None),
])
def test_point_range(text, expected):
    result = point_range(text)
    assert (None if result is None else tuple(str(value) for value in result)) == expected


def test_inspect_script_reports_lone_collided_and_single_tier_ranges(client):
    from backend.app.db import models

    rubric = _create(client, [ai_row(0, mutex="G1", group="G1", source_refs=["/criteria/T04/deduction_rules/0"]),
                              ai_row(1, mutex="G2", group="G2"), ai_row(2, mutex="G2", group="G2"),
                              ai_row(0, mutex="G2", group="G2", fingerprint=FP_B, points=1.5),
                              ai_row(1, mutex="G2", group="G2", fingerprint=FP_B, points=2.5)],
                     name="合成核对脚本")
    with client.session_factory() as session:
        criterion = session.get(models.Rubric, rubric["id"]).criteria[0]
        criterion.deduction_rules = ["格式不规范扣1-3分"]
        # 还原修复前的落库形态：单档组写着互斥组，两次起草共用同一个名字。
        rules =session.query(models.AtomicRule).filter(models.AtomicRule.criterion_id == criterion.id).all()
        for rule in rules:
            index = int(rule.rule_code.split(".")[3])
            rule.mutex_group = "G1" if index == 1 else "G2"
        session.commit()
        report = script.inspect_rubric(session, rubric["id"])
        session.rollback()

    assert report["counts"]["lone_mutex_groups"] == 1
    assert report["counts"]["lone_blocking_publish"] == 1
    assert report["lone_mutex_groups"][0]["mutex_group"] == "G1"
    assert report["collided_mutex_groups"] == [{
        "criterion_code": "T04", "mutex_group": "G2", "generations": 2,
        "rule_codes": [f"manual.t04.deduct.{n}.v1" for n in (2, 3, 4, 5)]}]
    assert report["single_tier_ranges"] == [{
        "criterion_code": "T04", "draft_group": "G1", "fingerprint_prefix": FP_A[:8],
        "rule_codes": ["manual.t04.deduct.1.v1"], "range_segments": [0], "unchecked_refs": 0}]
    dumped = str(report)
    assert "合成条款" not in dumped and "格式不规范" not in dumped


def test_inspect_script_unknown_rubric(client):
    with client.session_factory() as session:
        assert script.inspect_rubric(session, "missing")["found"] is False

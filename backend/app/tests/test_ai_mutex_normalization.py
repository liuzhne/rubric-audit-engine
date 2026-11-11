"""AI 起草细则的互斥组在编译时规整（AI起草互斥组规整与区间拆档方案 §4.1）。

单档的 AI 组不写互斥组，同名的两次起草不并成一组；评分项投影保留起草时的原名，
人工行原样，写错了照旧由发布校验挡住。
"""

import pytest

from backend.app.scripts import inspect_ai_draft_rules as script
from backend.app.services.rubric_import.compiler import point_range
from backend.app.services.rubric_import.deduction_caps import MUTEX_GROUP_MAX_CHARS
from backend.app.services.rubric_import.deduction_caps import effective_mutex_groups
from backend.app.services.rubric_import.deduction_caps import suffixed_mutex_groups
from backend.app.services.rubrics.atomic_recompile import _sync_ai_mutex_groups
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


# --- 复审补充：追加同步的口径 ----------------------------------------------

def test_append_path_leaves_non_ai_row_with_fingerprint_alone(client):
    # 带指纹和行键、来源却不是 ai_ 的行，编译时不规整，追加同步也不能把老师清空的互斥组改回去。
    rubric = _import_rubric(client, name="合成追加非 AI 行")["rubric"]
    typed = {**_method_rows(1)[0], "source": "user_text"}
    _recompile(client, rubric["id"], [typed], "v2")

    def clear(view):
        return [{"id": rule["id"], "content_token": rule["content_token"],
                 "changes": {"mutex_group": None} if rule["rule_code"] == "manual.method.deduct.1.v1" else {}}
                for rule in view["rules"]]

    cleared = _recompile(client, rubric["id"], [typed], "v3", atomic_rules=clear)
    assert _mutex_by_code(cleared)["manual.method.deduct.1.v1"] is None
    extra = ai_row(1, mutex="METHOD-B02-G01-SEVERITY", group="METHOD-B02-G01", criterion="METHOD",
                   fingerprint=FP_B)
    after = _recompile(client, rubric["id"], [typed, extra], "v4")
    assert _mutex_by_code(after)["manual.method.deduct.1.v1"] is None


def test_append_path_matches_reordered_rows_by_their_own_compiled_rule(client):
    # 行上写了已有细则的编号、位置却变了：同步要取这一行自己编译出的互斥组，不能取同位置别的行的。
    rubric = _import_rubric(client, name="合成追加重排")["rubric"]
    first = ai_row(0, mutex="METHOD-A", group="A", criterion="METHOD")
    second = ai_row(0, mutex="METHOD-B", group="B", criterion="METHOD", fingerprint=FP_B)
    before = _recompile(client, rubric["id"], [first, second], "v2")
    assert _mutex_by_code(before)["manual.method.deduct.1.v1"] is None
    assert _mutex_by_code(before)["manual.method.deduct.2.v1"] is None
    reordered = [{**second, "rule_code": "manual.method.deduct.2.v1"},
                 {**first, "rule_code": "manual.method.deduct.1.v1"},
                 ai_row(1, mutex="METHOD-B", group="B", criterion="METHOD", fingerprint=FP_B)]
    after = _mutex_by_code(_recompile(client, rubric["id"], reordered, "v3"))
    assert after["manual.method.deduct.1.v1"] is None
    assert after["manual.method.deduct.2.v1"] == after["manual.method.deduct.3.v1"] == "METHOD-B"


def test_inspect_script_reports_band_and_cross_criterion_groups(client):
    from backend.app.db import models

    rubric = _create(client, [manual_row("G3", points=1), manual_row("G3", points=2)], name="合成核对分档组")
    with client.session_factory() as session:
        rule = session.query(models.AtomicRule).filter(
            models.AtomicRule.rule_code == "manual.t04.deduct.1.v1",
            models.AtomicRule.criterion_id == session.get(models.Rubric, rubric["id"]).criteria[0].id).one()
        rule.direction = "band"
        session.commit()
        report = script.inspect_rubric(session, rubric["id"])
        session.rollback()
    assert report["counts"]["invalid_mutex_groups"] == 1
    assert report["invalid_mutex_groups"] == [{
        "criterion_codes": ["T04"], "mutex_group": "G3", "cross_criterion": False, "has_band": True,
        "rule_codes": ["manual.t04.deduct.1.v1", "manual.t04.deduct.2.v1"]}]
    assert report["lone_mutex_groups"] == []


def test_inspect_script_lone_band_group_blocks_even_after_exclusion(client):
    from backend.app.db import models
    from backend.app.services.rubrics.rule_status import EXCLUDED_STATUS

    rubric = _create(client, [manual_row("G5", points=1), manual_row("G5", points=2)], name="合成核对排除后的分档组")
    with client.session_factory() as session:
        criterion = session.get(models.Rubric, rubric["id"]).criteria[0]
        rules = {rule.rule_code: rule for rule in session.query(models.AtomicRule).filter(
            models.AtomicRule.criterion_id == criterion.id)}
        # 排除一条后只剩 1 条，但剩下的是分档细则：发布校验不看排除，照样报 mutex_definition_invalid。
        rules["manual.t04.deduct.1.v1"].direction = "band"
        rules["manual.t04.deduct.2.v1"].status = EXCLUDED_STATUS
        session.commit()
        report = script.inspect_rubric(session, rubric["id"])
        session.rollback()
    assert report["lone_mutex_groups"] == [{
        "criterion_code": "T04", "mutex_group": "G5", "rule_codes": ["manual.t04.deduct.1.v1"],
        "blocks_publish": True}]
    assert report["counts"]["lone_blocking_publish"] == 1


LONG_MUTEX = "长" * 100  # 接口直接写进来的 AI 行可以比起草生成的原名长，正好顶到列宽


def test_suffixed_name_stays_within_the_column_width():
    rows = [ai_row(0, mutex=LONG_MUTEX), ai_row(1, mutex=LONG_MUTEX),
            ai_row(0, mutex=LONG_MUTEX, fingerprint=FP_B), ai_row(1, mutex=LONG_MUTEX, fingerprint=FP_B)]
    result = effective_mutex_groups(rows, compiled={0, 1, 2, 3})
    assert result[:2] == [LONG_MUTEX, LONG_MUTEX]
    (name,) = suffixed_mutex_groups(LONG_MUTEX, FP_B)  # 放不下“原名-指纹”，只剩带原名哈希的写法
    assert result[2] == result[3] == name
    assert len(name) == MUTEX_GROUP_MAX_CHARS and name.endswith(f"-{FP_B[:8]}")


def test_truncated_names_with_the_same_prefix_stay_apart():
    # 两个超长原名前 91 个字相同、都被人工组占用：截短后不能并成同一个互斥组。
    first, second = "长" * 95 + "甲", "长" * 95 + "乙"
    rows = [manual_row(first), manual_row(first), manual_row(second), manual_row(second),
            ai_row(0, mutex=first), ai_row(1, mutex=first),
            ai_row(0, mutex=second, group="T04-B01-G02"), ai_row(1, mutex=second, group="T04-B01-G02")]
    result = effective_mutex_groups(rows, compiled=set(range(8)))
    assert result[4] == result[5] and result[6] == result[7]
    assert result[4] != result[6]
    assert {len(result[4]), len(result[6])} == {MUTEX_GROUP_MAX_CHARS}
    assert not {result[4], result[6]} & {first, second}


def test_suffixed_name_taken_by_a_manual_row_falls_back_to_the_hashed_name():
    preferred, hashed = suffixed_mutex_groups(MUTEX, FP_B)
    rows = [manual_row(MUTEX), manual_row(MUTEX), manual_row(preferred), manual_row(preferred),
            ai_row(0, fingerprint=FP_B), ai_row(1, fingerprint=FP_B)]
    assert effective_mutex_groups(rows, compiled=set(range(6)))[4:] == [hashed, hashed]


@pytest.mark.parametrize("which", [0, -1])
def test_append_sync_recognizes_every_machine_name(which):
    row = ai_row(0, mutex=MUTEX)
    existing = [{"rule_code": "manual.t04.deduct.1.v1", "mutex_group": suffixed_mutex_groups(MUTEX, FP_A)[which]}]
    generated = [{"rule_code": "manual.t04.deduct.1.v1", "mutex_group": None}]
    command = {"rubric": {"criteria": [{"code": "T04", "deduction_rules_structured": [row]}]}}
    _sync_ai_mutex_groups(existing, generated, command)
    assert existing[0]["mutex_group"] is None


def test_append_sync_recognizes_a_truncated_suffixed_name():
    row = ai_row(0, mutex=LONG_MUTEX)
    existing = [{"rule_code": "manual.t04.deduct.1.v1", "mutex_group": suffixed_mutex_groups(LONG_MUTEX, FP_A)[0]}]
    generated = [{"rule_code": "manual.t04.deduct.1.v1", "mutex_group": None}]
    command = {"rubric": {"criteria": [{"code": "T04", "deduction_rules_structured": [row]}]}}
    _sync_ai_mutex_groups(existing, generated, command)
    assert existing[0]["mutex_group"] is None


def _rows_with_every_name_taken():
    names = (MUTEX, *suffixed_mutex_groups(MUTEX, FP_B))
    return [row for name in names for row in (manual_row(name), manual_row(name))] + [
        ai_row(0, fingerprint=FP_B), ai_row(1, fingerprint=FP_B)]


def test_every_candidate_taken_raises_instead_of_merging():
    rows = _rows_with_every_name_taken()
    with pytest.raises(ValueError, match="互斥组"):
        effective_mutex_groups(rows, compiled=set(range(len(rows))))


def test_every_candidate_taken_rejects_creation(client):
    response = client.post("/api/rubrics", json={"name": "合成候选名全被占用", "version": "v1", "total_score": 10,
        "criteria": [{"code": "T04", "name": "合成格式", "max_score": 10, "scoring_mode": "deductive",
                      "deduction_rules_structured": _rows_with_every_name_taken()}]})
    assert response.status_code == 400
    assert "互斥组" in response.json()["detail"]


def test_every_candidate_taken_blocks_the_append_path_without_writing(client):
    rubric = _import_rubric(client, name="合成追加候选名全被占用")["rubric"]
    before = workspace(client, rubric["id"])
    names = ("METHOD-B01-G01-SEVERITY", *suffixed_mutex_groups("METHOD-B01-G01-SEVERITY", FP_A))
    rows = [row for name in names for row in (manual_row(name), manual_row(name))] + _method_rows(2)
    criteria = client.get(f"/api/rubrics/{rubric['id']}").json()["criteria"]
    criteria[0]["scoring_mode"] = "deductive"
    criteria[0]["deduction_rules_structured"] = rows
    response = client.post(f"/api/rubrics/{rubric['id']}/recompile", json={
        "supersedes_compilation_id": before["compilation_id"], "version": "v2", "criteria": criteria})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "RUBRIC_RECOMPILE_BLOCKED" and "互斥组" in detail["context"]["reason"]
    assert workspace(client, rubric["id"])["compilation_id"] == before["compilation_id"]

"""导入时的表格结构自动识别（docs/评分表结构自动识别方案.md）。

确定性解析后仍有强信号时自动建一次结构识别任务；对照由代码按模型结构从原表取数；
用户采用后才替换草稿，保留只记选择。模型一律用替身，不调用真实模型。
"""

from io import BytesIO

import pytest
from openpyxl import Workbook

from backend.app.core.config import settings
from backend.app.db import models
from backend.app.tests import rubric_parse_fixtures as fx
from backend.app.tests.test_rubric_structure_api import XLSX
from backend.app.tests.test_rubric_structure_api import _drain
from backend.app.tests.test_rubric_structure_api import use_scorer  # noqa: F401  pytest fixture


def _ambiguous_scores_xlsx():
    """两列“分值”都每行一个值：C 列逐行重复上级分值，E 列是评分项分值。确定性抽取分不清，取到 C 列。"""

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "评分表"
    sheet.append(["开题报告评分表（满分 25 分）"])  # R1
    sheet.append(["一级指标", "分值", "评分项", "分值", "打分规则"])  # R2
    sheet.append(["一、基本信息", 10, "封面信息", 4, "每缺1项扣1分"])  # R3
    sheet.append(["一、基本信息", 10, "填写规范", 2, "每处不规范扣1分"])  # R4
    sheet.append(["一、基本信息", 10, "论文题目", 4, "不一致记0分"])  # R5
    sheet.append(["二、选题背景", 15, "选题背景", 8, "充分8分"])  # R6
    sheet.append(["二、选题背景", 15, "研究目的", 7, "明确7分"])  # R7
    sheet.append(["合计（满分 25 分）"])  # R8
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


MODEL_STRUCTURE = {
    "sheet": "评分表", "header_row": "R2",
    "column_mapping": [
        {"col": "C1", "field": "dimension", "reason": "一级指标名称"},
        {"col": "C2", "field": "ignore", "reason": "一级指标分值，与右侧评分项分值同名"},
        {"col": "C3", "field": "name", "reason": "评分项名称"},
        {"col": "C4", "field": "max_score", "reason": "每行一个分值，五项之和等于 25"},
        {"col": "C5", "field": "description", "reason": "规则正文"},
    ],
    "row_types": [{"row": "R8", "type": "total", "reason": "合计行"}],
    "unresolved": [],
}


@pytest.fixture
def auto_on(monkeypatch):
    monkeypatch.setattr(settings, "IMPORT_STRUCTURE_AUTO_ENABLED", True)


def _create(client, data):
    response = client.post("/api/rubrics/import-sessions", data={"name": "开题报告", "version": "v1"},
                           files={"rules_file": ("rules.xlsx", data, XLSX)})
    assert response.status_code == 201, response.text
    return response.json()


def _session(client, session_id):
    response = client.get(f"/api/rubrics/import-sessions/{session_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_switch_off_keeps_import_unchanged(client, use_scorer):
    scorer = use_scorer(MODEL_STRUCTURE)
    body = _create(client, _ambiguous_scores_xlsx())
    assert body["structure_check"] is None
    assert scorer.calls == []


def test_no_strong_signal_does_not_call_model(client, use_scorer, auto_on):
    scorer = use_scorer(MODEL_STRUCTURE)
    body = _create(client, fx.two_level_score_xlsx())
    assert [item["max_score"] for item in body["criteria"]] == [4, 2, 4, 8, 7]
    assert body["structure_check"] is None
    _drain(client)
    assert scorer.calls == []


def test_strong_signal_runs_identification_and_apply_replaces_draft(client, use_scorer, auto_on):
    scorer = use_scorer(MODEL_STRUCTURE)
    body = _create(client, _ambiguous_scores_xlsx())
    # 确定性抽取取到了逐行重复的上级分值
    assert [item["max_score"] for item in body["criteria"]] == [10, 10, 10, 15, 15]
    check = body["structure_check"]
    assert check["status"] == "running"
    assert {item["code"] for item in check["signals"]} >= {"E4", "E10"}

    _drain(client)
    assert len(scorer.calls) == 1
    payload = scorer.calls[0][1]
    # 满分列被信号指认为可疑，不作为已知映射发给模型；合计行补发
    assert "max_score" not in payload["known_mapping"]
    assert "E10" in payload["failure_codes"]

    body = _session(client, body["id"])
    check = body["structure_check"]
    assert check["status"] == "differs"
    comparison = check["comparison"]
    assert comparison["current_total"] == 60 and comparison["proposed_total"] == 25
    assert comparison["declared_total"] == 25
    first = comparison["rows"][0]
    assert first == {
        "row_number": 3, "change": "changed", "fields": ["max_score"],
        "current_name": "封面信息", "current_max_score": 10.0,
        "proposed_name": "封面信息", "proposed_max_score": 4.0, "proposed_dimension": "一、基本信息",
    }
    assert {"column": 2, "header": "分值", "current_field": "max_score", "proposed_field": None} in comparison["columns"]
    assert {"ref": "C4", "reason": "每行一个分值，五项之和等于 25"} in comparison["reasons"]

    response = client.post(f"/api/rubrics/import-sessions/{body['id']}/structure/apply",
                           json={"expected_state_version": body["state_version"]})
    assert response.status_code == 200, response.text
    applied = response.json()
    assert [item["max_score"] for item in applied["criteria"]] == [4, 2, 4, 8, 7]
    assert applied["total_score"] == 25
    assert applied["state_version"] == body["state_version"] + 1
    assert applied["structure_check"]["status"] == "applied"
    assert all(item["parse_status"] == "parsed" for item in applied["criteria"])
    # 已采用后不能再次采用
    again = client.post(f"/api/rubrics/import-sessions/{body['id']}/structure/apply",
                        json={"expected_state_version": applied["state_version"]})
    assert again.status_code == 409


def test_edit_during_identification_makes_result_stale(client, use_scorer, auto_on):
    use_scorer(MODEL_STRUCTURE)
    body = _create(client, _ambiguous_scores_xlsx())
    edited = client.patch(f"/api/rubrics/import-sessions/{body['id']}",
                          json={"expected_state_version": body["state_version"], "name": "改过名字"})
    assert edited.status_code == 200, edited.text
    _drain(client)
    current = _session(client, body["id"])
    assert current["structure_check"]["status"] == "stale"
    response = client.post(f"/api/rubrics/import-sessions/{body['id']}/structure/apply",
                           json={"expected_state_version": current["state_version"]})
    assert response.status_code == 409
    assert [item["max_score"] for item in _session(client, body["id"])["criteria"]] == [10, 10, 10, 15, 15]


def test_dismiss_keeps_draft_and_records_choice(client, use_scorer, auto_on):
    use_scorer(MODEL_STRUCTURE)
    body = _create(client, _ambiguous_scores_xlsx())
    _drain(client)
    response = client.post(f"/api/rubrics/import-sessions/{body['id']}/structure/dismiss",
                           json={"expected_state_version": body["state_version"]})
    assert response.status_code == 200, response.text
    dismissed = response.json()
    assert dismissed["structure_check"]["status"] == "dismissed"
    assert dismissed["structure_check"]["comparison"]["proposed_total"] == 25
    assert [item["max_score"] for item in dismissed["criteria"]] == [10, 10, 10, 15, 15]
    assert dismissed["state_version"] == body["state_version"]


def test_consistent_result_is_reported_quietly(client, use_scorer, auto_on):
    """模型结构与确定性结果一致：状态 consistent，页面只显示一行小字。"""

    use_scorer({**MODEL_STRUCTURE, "column_mapping": [
        {"col": "C1", "field": "dimension", "reason": "维度"},
        {"col": "C2", "field": "max_score", "reason": "第一列分值"},
        {"col": "C3", "field": "name", "reason": "名称"},
        {"col": "C5", "field": "description", "reason": "规则"},
    ]})
    body = _create(client, _ambiguous_scores_xlsx())
    _drain(client)
    assert _session(client, body["id"])["structure_check"]["status"] == "consistent"


def test_failed_identification_does_not_block_import(client, use_scorer, auto_on):
    use_scorer({"bad": "output"}, {"still": "bad"})
    body = _create(client, _ambiguous_scores_xlsx())
    _drain(client)
    current = _session(client, body["id"])
    assert current["structure_check"]["status"] == "failed"
    assert current["structure_check"]["message"]
    assert len(current["criteria"]) == 5


def test_result_survives_pruning_of_upload_tasks(client, use_scorer, auto_on):
    """同一用户下一次导入前识别会清理已结束的旧任务；结果第一次读取时已缓存进会话。"""

    use_scorer(MODEL_STRUCTURE, MODEL_STRUCTURE)
    first = _create(client, _ambiguous_scores_xlsx())
    _drain(client)
    assert _session(client, first["id"])["structure_check"]["status"] == "differs"
    second = _create(client, _ambiguous_scores_xlsx() + b"")
    with client.session_factory() as db:
        row = db.get(models.RubricImportSession, first["id"])
        task_id = row.draft_data["structure_check"]["task_id"]
        assert row.draft_data["structure_check"]["result"]["override"]["column_mapping"]["max_score"] == 4
        db.query(models.AITaskItem).filter(models.AITaskItem.task_id == task_id).delete()
        db.query(models.AITask).filter(models.AITask.id == task_id).delete()
        db.commit()
    assert second["structure_check"] is not None
    assert _session(client, first["id"])["structure_check"]["status"] == "differs"


def test_reupload_of_existing_rubric_starts_draft_identification(client, use_scorer, auto_on):
    scorer = use_scorer(MODEL_STRUCTURE)
    created = _create(client, fx.two_level_score_xlsx())
    confirmed = client.post(f"/api/rubrics/import-sessions/{created['id']}/confirm",
                            json={"expected_state_version": created["state_version"], "idempotency_key": "auto-reupload"})
    assert confirmed.status_code == 200, confirmed.text
    rubric_id = confirmed.json()["rubric"]["id"]
    preview = client.post(f"/api/rubrics/{rubric_id}/reupload-preview",
                          files={"rules_file": ("rules.xlsx", _ambiguous_scores_xlsx(), XLSX)})
    assert preview.status_code == 200, preview.text
    applied = client.post(f"/api/rubrics/{rubric_id}/reupload-confirm",
                          data={"fingerprint": preview.json()["fingerprint"]},
                          files={"rules_file": ("rules.xlsx", _ambiguous_scores_xlsx(), XLSX)})
    assert applied.status_code == 200, applied.text
    _drain(client)
    assert len(scorer.calls) == 1
    assert "max_score" not in scorer.calls[0][1]["known_mapping"]
    suggestion = client.get(f"/api/rubrics/{rubric_id}/parse-coverage").json().get("structure_suggestions")
    assert suggestion is not None

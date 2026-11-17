"""导入时的表格结构自动识别（docs/评分表结构自动识别方案.md §5–§7）。

确定性解析后仍有强信号时，用会话里保存的原文件建一次结构识别任务（``structure_suggestion``，
导入前目标，不挂评分标准）。模型只给结构；对照由代码把“按模型结构从原表取出的评分项”与
当前草稿逐行比较得出；用户采用后用同一份文件带 ``structure_override`` 重新解析，替换草稿。

状态放在 ``draft_data.structure_check``，不建表：

- ``task_id`` / ``signals`` / ``state_version``：建任务时的会话版本，采用时据此判断草稿是否被改过；
- ``result``：任务成功后第一次读取时缓存的识别结果（导入前任务会在同一用户下次发起时被清理）；
- ``dismissed`` / ``applied``：用户的选择；``unavailable``：没能建任务的原因。
"""

from __future__ import annotations

import logging
from copy import deepcopy

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db import models
from backend.app.services.ai_tasks.errors import AITaskProblem
from backend.app.services.ai_tasks.service import create_ai_task
from backend.app.services.ai_tasks.service import get_ai_task
from backend.app.services.ai_tasks.state import ACTIVE_TASK_STATUSES
from backend.app.services.rubric_import import import_sessions
from backend.app.services.rubric_import import structure_state
from backend.app.services.rubric_import.extraction.triggers import strong_triggers
from backend.app.services.rubric_import.parse_state import ParseStateError

logger = logging.getLogger(__name__)
KIND = "structure_suggestion"
_KEY = "structure_check"


class StructureCheckError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def signals_of(prepared: dict) -> list[dict]:
    raw = (prepared.get("compilation") or {}).get("raw_parse_output") or {}
    return [
        {"code": item["code"], "message": item["message"]}
        for item in strong_triggers(raw.get("triggers"))
    ]


def _store(row: models.RubricImportSession, check: dict | None) -> None:
    draft = deepcopy(row.draft_data or {})
    if check is None:
        draft.pop(_KEY, None)
    else:
        draft[_KEY] = check
    row.draft_data = draft


def _check(row: models.RubricImportSession) -> dict | None:
    value = (row.draft_data or {}).get(_KEY)
    return deepcopy(value) if isinstance(value, dict) else None


def start(db: Session, row: models.RubricImportSession, principal) -> models.AITask | None:
    """会话刚创建或刚换过文件：有强信号就建识别任务。返回新建的任务（需要叫醒），否则 None。

    建任务失败（没有可用模型、表格超出预算等）不影响导入，只把原因记下给页面显示。
    """

    if not settings.IMPORT_STRUCTURE_AUTO_ENABLED or not row.rules_file_bytes:
        _store(row, None)
        return None
    signals = signals_of(row.prepared_graph or {})
    if not signals:
        _store(row, None)
        return None
    check = {"signals": signals, "state_version": row.state_version, "task_id": None,
             "result": None, "dismissed": False, "applied": False, "unavailable": None}
    task = None
    try:
        upload = structure_state.import_structure_request(
            rules_bytes=row.rules_file_bytes,
            template_bytes=None,
            business_profile_key=import_sessions.profile_key(row),
        )
        task, created = create_ai_task(
            db, rubric_id=None, kind=KIND, params={"upload": upload}, principal=principal
        )
        check["current_columns"] = upload.get("current_columns") or []
        check["task_id"] = task.id
        if not created:
            task = None if task.status not in ACTIVE_TASK_STATUSES else task
    except AITaskProblem as exc:
        check["unavailable"] = exc.message
        task = None
    except ParseStateError as exc:
        check["unavailable"] = exc.message
        task = None
    _store(row, check)
    return task


def start_for_rubric(db: Session, rubric_id: str, principal) -> models.AITask | None:
    """已建评分标准重新上传后：新草稿有强信号就自动建草稿结构识别任务（结果进现有的结构建议面板）。"""

    if not settings.IMPORT_STRUCTURE_AUTO_ENABLED:
        return None
    try:
        compilation = structure_state._require_reparse_source(db, rubric_id)
    except ParseStateError:
        return None
    raw = compilation.raw_parse_output or {}
    if not strong_triggers(raw.get("triggers")) or (compilation.raw_model_output or {}).get("structure_suggestions"):
        return None
    try:
        task, created = create_ai_task(
            db, rubric_id=rubric_id, kind=KIND, params={"target": "draft"}, principal=principal
        )
    except AITaskProblem as exc:
        logger.info("rubric_structure_auto_skipped rubric_id=%s code=%s", rubric_id, exc.code)
        return None
    return task if created and task.status in ACTIVE_TASK_STATUSES else None


def _rows_by_number(criteria) -> dict[int, dict]:
    rows = {}
    for item in criteria:
        if item.get("deleted"):
            continue
        for ref in item.get("source_refs") or []:
            if ref.get("kind") == "excel" and ref.get("row_number"):
                rows.setdefault(int(ref["row_number"]), item)
                break
    return rows


def _column_changes(current_columns: list[dict], proposed_columns: list[dict]) -> list[dict]:
    now = {item["column"]: item for item in current_columns}
    new = {item["column"]: item for item in proposed_columns}
    changes = []
    for column in sorted(set(now) | set(new)):
        before, after = (now.get(column) or {}).get("field"), (new.get(column) or {}).get("field")
        if before != after:
            header = (new.get(column) or now.get(column) or {}).get("header") or ""
            changes.append({"column": column, "header": header, "current_field": before, "proposed_field": after})
    return changes


def _reasons(result: dict) -> list[dict]:
    reasons = result.get("reasons") or {}
    items = []
    for group in ("columns", "rows"):
        for ref, reason in (reasons.get(group) or {}).items():
            if reason:
                items.append({"ref": ref, "reason": str(reason)})
    return items


def compare(criteria: list[dict], result: dict, current_columns: list[dict] | None = None) -> dict:
    """按原表行号对齐当前草稿与模型结构下的评分项，只用代码从原表取出的数据。"""

    current = _rows_by_number(criteria)
    proposed = {int(item["row_number"]): item for item in result.get("preview") or []}
    rows = []
    for number in sorted(set(current) | set(proposed)):
        now, new = current.get(number), proposed.get(number)
        if now and new:
            changes = [key for key, a, b in (("name", now["name"], new["name"]),
                                             ("max_score", float(now["max_score"]), float(new["max_score"])))
                       if a != b]
            change = "changed" if changes else "same"
        else:
            changes = []
            change = "added" if new else "removed"
        rows.append({
            "row_number": number,
            "change": change,
            "fields": changes,
            "current_name": now["name"] if now else None,
            "current_max_score": now["max_score"] if now else None,
            "proposed_name": new["name"] if new else None,
            "proposed_max_score": new["max_score"] if new else None,
            "proposed_dimension": new.get("dimension") if new else None,
        })
    manual = [item for item in criteria if not item.get("deleted") and not _rows_by_number([item])]
    current_total = sum(float(item["max_score"]) for item in criteria if not item.get("deleted"))
    proposed_total = sum(float(item["max_score"]) for item in result.get("preview") or []) + sum(
        float(item["max_score"]) for item in manual
    )
    return {
        "rows": rows,
        "current_total": current_total,
        "proposed_total": proposed_total,
        "declared_total": result.get("declared_total"),
        "columns": _column_changes(current_columns or [], result.get("proposed_columns") or []),
        "reasons": _reasons(result),
        "consistent": all(item["change"] == "same" for item in rows),
    }


def refresh(db: Session, row: models.RubricImportSession) -> bool:
    """任务成功后把结果缓存进会话（任务可能被同一用户的下一次导入前识别清理掉）。"""

    check = _check(row)
    if not check or not check.get("task_id") or check.get("result") is not None:
        return False
    task = get_ai_task(db, check["task_id"])
    if task is None or task.status != "succeeded" or not isinstance(task.result, dict):
        return False
    check["result"] = deepcopy(task.result)
    _store(row, check)
    return True


def _task_state(db: Session, check: dict) -> tuple[str, str | None]:
    if check.get("result") is not None:
        return "succeeded", None
    if not check.get("task_id"):
        return "unavailable", check.get("unavailable")
    task = get_ai_task(db, check["task_id"])
    if task is None:
        return "failed", "识别结果已失效，请重新上传文件。"
    if task.status in ACTIVE_TASK_STATUSES:
        return "running", None
    if task.status == "succeeded":
        return "succeeded", None
    return "failed", task.error_message or "AI 识别没有完成。"


def view(db: Session, row: models.RubricImportSession) -> dict | None:
    check = _check(row)
    if not check:
        return None
    state, message = _task_state(db, check)
    out = {"signals": check.get("signals") or [], "task_id": check.get("task_id"), "message": message,
           "comparison": None}
    if state != "succeeded":
        out["status"] = state
        return out
    result = check.get("result")
    if result is None:
        task = get_ai_task(db, check["task_id"])
        result = task.result if task is not None else None
    if not isinstance(result, dict):
        out.update(status="failed", message="识别结果已失效，请重新上传文件。")
        return out
    comparison = compare((row.draft_data or {}).get("criteria") or [], result, check.get("current_columns"))
    out["comparison"] = comparison
    if check.get("applied"):
        out["status"] = "applied"
    elif comparison["consistent"]:
        out["status"] = "consistent"
    elif check.get("dismissed"):
        out["status"] = "dismissed"
    elif check.get("state_version") != row.state_version:
        out["status"] = "stale"
        out["message"] = "识别期间草稿已修改，识别结果不能直接采用；可以重新上传文件再识别。"
    else:
        out["status"] = "differs"
    return out


def override_for_apply(db: Session, row: models.RubricImportSession, *, expected_state_version: int) -> dict:
    if row.status != "draft":
        raise StructureCheckError(409, "只有待确认的导入会话可以采用识别结果")
    if row.state_version != expected_state_version:
        raise StructureCheckError(409, "导入草稿已更新，请刷新后再采用")
    check = _check(row)
    if not check:
        raise StructureCheckError(404, "这次导入没有表格结构识别结果")
    refresh(db, row)
    current = view(db, row)
    if current["status"] != "differs":
        raise StructureCheckError(409, current.get("message") or "当前没有可以采用的识别结果")
    return deepcopy(_check(row)["result"]["override"])


def mark(row: models.RubricImportSession, **values) -> None:
    check = _check(row)
    if not check:
        raise StructureCheckError(404, "这次导入没有表格结构识别结果")
    check.update(values)
    _store(row, check)


__all__ = [
    "StructureCheckError", "compare", "mark", "override_for_apply", "refresh", "signals_of", "start",
    "start_for_rubric", "view",
]

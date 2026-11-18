"""抽取器 LLM 兜底的触发条件 E1–E8（解析重构方案 §5.1）。

这里只做确定性判定：命中时由界面提示用户（原因 / 范围 / 代价），
用户确认后才会调用 LLM，绝不在后台自动触发。``unit_ids`` 是建议发送给模型的局部范围。
阈值是起点，按真实模板统计校准。
"""

from __future__ import annotations

from backend.app.services.rubric_import.coverage import unit_signals
from backend.app.services.rubric_import.extraction.table_extractor import TableExtraction
from backend.app.services.rubric_import.sources.units import SourceLedger

UNMAPPED_COLUMN_RATIO = 0.3
DROPPED_ROW_RATIO = 0.2
MIN_RULES_COVERAGE = 0.9
HEADER_SCAN_UNITS = 60


# 强信号：确定性解析之后仍然自相矛盾，导入时自动请模型识别结构（见
# docs/评分表结构自动识别方案.md §4）。其余是弱信号，只提示、给手动入口。
STRONG_CODES = frozenset({"E4", "E9", "E10", "E11"})


def _trigger(code: str, message: str, unit_ids, details=None, *, strong: bool | None = None) -> dict:
    return {"code": code, "message": message, "unit_ids": list(dict.fromkeys(unit_ids)), "details": details or {},
            "strong": code in STRONG_CODES if strong is None else strong}


def strong_triggers(triggers) -> list[dict]:
    return [item for item in triggers or [] if item.get("strong")]


def _is_number(unit) -> bool:
    return unit.context.get("value_type") == "num"


def _numeric_column(ledger: SourceLedger, unit_ids: list[str]) -> bool:
    """数字占多数、且不是 1、2、3…… 连续编号的列，像是没认出来的分值列。"""

    units = [ledger.unit(unit_id) for unit_id in dict.fromkeys(unit_ids) if ledger.has(unit_id)]
    if not units:
        return False
    numbers = []
    for unit in units:
        if not _is_number(unit):
            continue
        try:
            numbers.append(float(unit.text))
        except ValueError:
            continue
    if len(numbers) * 2 < len(units):
        return False
    return not _is_numbering(numbers)


def _is_numbering(numbers: list[float]) -> bool:
    """编号列：整段 1、2、3……，或分组后每段都从 1 重新编号（1、2、3、1、2）。

    每段都只有一个 1（1、1、1……）更像都是 1 分的分值列，不算编号。"""

    if not numbers or not all(value.is_integer() for value in numbers):
        return False
    if numbers == [numbers[0] + i for i in range(len(numbers))]:
        return True
    runs, length = [], 0
    for index, value in enumerate(numbers):
        if index and value == numbers[index - 1] + 1:
            length += 1
            continue
        if value != 1:
            return False
        if length:
            runs.append(length)
        length = 1
    runs.append(length)
    return max(runs) >= 2


def header_not_found_trigger(ledger: SourceLedger) -> dict:
    """E1：所有工作表都找不到表头（此时导入失败，没有评分项草稿）。"""

    units = [unit.unit_id for unit in ledger.units() if unit.doc_role == "rules"][:HEADER_SCAN_UNITS]
    return _trigger("E1", "未识别到评分规则表头，可使用 AI 识别表格结构。", units)


def detect_triggers(extraction: TableExtraction, ledger: SourceLedger, coverage: dict, *, profile_terms=()) -> list[dict]:
    triggers: list[dict] = []
    header_units = [u for u in _row_units(ledger, extraction, extraction.header_index + 1) if u]

    if "max_score" not in extraction.mapping:
        triggers.append(
            _trigger("E2", "未识别到“满分”列，分值来自评分项文字，请核对。", header_units,
                     {"mapping": sorted(extraction.mapping)})
        )

    wide = [
        column for column in extraction.unmapped_columns
        if column["data_rows"] and column["non_empty"] / column["data_rows"] > UNMAPPED_COLUMN_RATIO
    ]
    if wide:
        units = list(header_units)
        numeric_columns = []
        for column in wide:
            index = column["column"] - 1
            cells = [row.unit_ids[index] for row in extraction.records
                     if index < len(row.unit_ids) and row.unit_ids[index]]
            units.extend(cells)
            if _numeric_column(ledger, cells):
                numeric_columns.append(column["header"])
        triggers.append(
            _trigger("E3", "存在未识别用途但有内容的列：" + "、".join(c["header"] for c in wide), units,
                     {"columns": [c["header"] for c in wide], "numeric_columns": numeric_columns},
                     strong=bool(numeric_columns))
        )

    parsed_total = round(sum(row.criterion.max_score for row in extraction.records), 6)
    declared_sources = []
    if (extraction.total_row or {}).get("declared_total") is not None:
        declared_sources.append(("合计行", extraction.total_row["declared_total"],
                                 [u for u in _row_units(ledger, extraction, extraction.total_row["row_number"]) if u]))
    if extraction.declared_total:
        declared_sources.append(("表头声明的满分", extraction.declared_total["value"],
                                 list(extraction.declared_total["unit_ids"])))
    mismatch = next((item for item in declared_sources if abs(item[1] - parsed_total) > 1e-6), None)
    if mismatch:
        label, declared, units = mismatch
        units = list(units)
        for item in extraction.dropped_rows:
            units.extend(item["unit_ids"])
        triggers.append(
            _trigger("E4", f"评分项满分合计 {parsed_total:g} 与{label} {declared:g} 不一致。", units,
                     {"declared_total": declared, "parsed_total": parsed_total, "source": label})
        )

    data_rows = len(extraction.records) + len(extraction.dropped_rows)
    if data_rows and len(extraction.dropped_rows) / data_rows > DROPPED_ROW_RATIO:
        units = [u for item in extraction.dropped_rows for u in item["unit_ids"]]
        triggers.append(
            _trigger("E5", f"表头之后有 {len(extraction.dropped_rows)} 行未能识别为评分项。", units,
                     {"dropped_rows": [item["row_number"] for item in extraction.dropped_rows]})
        )

    rules_doc = next((d for d in coverage.get("documents", []) if d["doc_role"] == "rules"), None)
    if rules_doc and rules_doc["ratio"] is not None and rules_doc["ratio"] < MIN_RULES_COVERAGE:
        units = [item["unit_id"] for item in coverage["unclaimed"] if item["doc_role"] == "rules"]
        triggers.append(
            _trigger("E6", f"规则文档覆盖率 {rules_doc['ratio']:.0%}，有内容未被识别。", units,
                     {"ratio": rules_doc["ratio"]})
        )

    ignored_titles = {sheet["title"] for sheet in extraction.ignored_sheets}
    suspicious = [
        unit.unit_id for unit in ledger.units()
        if unit.context.get("sheet") in ignored_titles and unit_signals(unit, profile_terms)
    ]
    if suspicious:
        triggers.append(
            _trigger("E8", "其他工作表中存在疑似评分规则：" + "、".join(sorted(ignored_titles)), suspicious,
                     {"sheets": sorted(ignored_titles)})
        )
    score_column = extraction.mapping.get("max_score")
    if score_column is not None:
        shared: dict[str, list] = {}
        for row in extraction.records:
            if score_column < len(row.unit_ids) and row.unit_ids[score_column]:
                shared.setdefault(row.unit_ids[score_column], []).append(row)
        groups = [(unit_id, rows) for unit_id, rows in shared.items() if len(rows) > 1]
        if groups:
            triggers.append(
                _trigger(
                    "E11",
                    "多个评分项共用同一个满分单元格，满分可能取成了上级指标的分值。",
                    [unit_id for unit_id, _ in groups] + header_units,
                    {"criterion_codes": [row.criterion.code for _, rows in groups for row in rows]},
                )
            )
    for issue in extraction.structure_issues:
        if issue.get("code") == "DUPLICATE_SCORE_COLUMN":
            triggers.append(
                _trigger("E10", "表头有多列都像满分列，" + issue.get("reason", ""), issue.get("unit_ids") or [],
                         {"columns": list(issue.get("columns") or []), "reason": issue.get("reason")})
            )
            continue
        if issue.get("code") != "MERGED_NAME_AMBIGUOUS":
            continue
        triggers.append(
            _trigger(
                "E9",
                "多个独立评分项共享同一个合并名称，疑似把父级评价项目解析成了评分项名称，请核对表格结构。",
                issue.get("unit_ids") or [],
                {
                    "reason": issue.get("reason"),
                    "row_numbers": list(issue.get("row_numbers") or []),
                    "criterion_codes": list(issue.get("criterion_codes") or []),
                },
            )
        )
    return triggers


def _row_units(ledger: SourceLedger, extraction: TableExtraction, row_number: int) -> list[str]:
    prefix = f"xlsx:{extraction.sheet_title}!R{row_number}C"
    return [unit.unit_id for unit in ledger.units() if unit.unit_id.startswith(prefix)]

import pytest
from io import BytesIO

from openpyxl import Workbook

from backend.app.services.rubric_import.coverage import compute_coverage
from backend.app.services.rubric_import.extraction.table_extractor import extract_table
from backend.app.services.rubric_import.extraction.triggers import detect_triggers
from backend.app.services.rubric_import.extraction.triggers import header_not_found_trigger
from backend.app.services.rubric_import.sources.units import SourceLedger
from backend.app.services.rubric_import.sources.xlsx_adapter import load_xlsx
from backend.app.tests import rubric_parse_fixtures as fx


def _triggers(data):
    ledger = SourceLedger()
    extraction = extract_table(load_xlsx(data, ledger, doc_id="excel", doc_role="rules"), ledger)
    return {t["code"]: t for t in detect_triggers(extraction, ledger, compute_coverage(ledger))}


def _book(rows):
    workbook = Workbook()
    for row in rows:
        workbook.active.append(row)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_clean_table_has_no_triggers():
    assert _triggers(fx.simple_rules_xlsx()) == {}


def test_complex_table_triggers_unmapped_column_total_mismatch_low_coverage_and_other_sheet():
    triggers = _triggers(fx.complex_rules_xlsx())
    # 阶段 3 修复表尾说明误解析后，合计与三项之和一致（无 E4），丢弃 2/5 行（E5）
    assert set(triggers) == {"E3", "E5", "E6", "E8"}
    assert triggers["E3"]["details"]["columns"] == ["备注"]
    assert triggers["E5"]["details"] == {"dropped_rows": [7, 8]}
    assert "xlsx:附加扣分!R2C3" in triggers["E8"]["unit_ids"]
    for trigger in triggers.values():
        assert trigger["message"]
        assert all(isinstance(unit, str) for unit in trigger["unit_ids"])


def test_total_mismatch_triggers_e4():
    triggers = _triggers(_book([["评分项", "分值"], ["选题", 10], ["写作", 20], ["合计", 40]]))
    assert triggers["E4"]["details"] == {"declared_total": 40.0, "parsed_total": 30.0, "source": "合计行"}
    assert triggers["E4"]["strong"] is True


def test_dropped_row_ratio_triggers_e5():
    triggers = _triggers(_book([["评分项", "分值"], ["选题", 10], ["写作", None], ["答辩", None]]))
    assert "E5" in triggers
    assert len(triggers["E5"]["unit_ids"]) == 2


def test_embedded_scores_without_score_column_trigger_e2():
    triggers = _triggers(_book([["打分项", "评价内容"], ["指导教师成绩项1（20分）", "选题"]]))
    assert "E2" in triggers


def test_other_sheet_without_rule_signal_does_not_trigger_e8():
    workbook = Workbook()
    workbook.active.append(["评分项", "分值"])
    workbook.active.append(["选题", 10])
    workbook.create_sheet("封面").append(["学院名称"])
    buffer = BytesIO()
    workbook.save(buffer)
    assert "E8" not in _triggers(buffer.getvalue())


def test_header_not_found_trigger_lists_leading_units():
    ledger = SourceLedger()
    load_xlsx(fx.unmapped_header_xlsx(), ledger, doc_id="excel", doc_role="rules")
    trigger = header_not_found_trigger(ledger)
    assert trigger["code"] == "E1"
    assert trigger["unit_ids"][:3] == ["xlsx:Sheet!R1C1", "xlsx:Sheet!R1C2", "xlsx:Sheet!R1C3"]


def test_merged_parent_without_stable_item_labels_triggers_e9():
    triggers = _triggers(fx.merged_parent_dimension_xlsx(with_item_labels=False))

    assert "E9" in triggers
    assert triggers["E9"]["unit_ids"] == ["xlsx:Sheet!R2C2"]
    assert triggers["E9"]["details"]["criterion_codes"] == ["T02", "T03", "T04", "T05"]


def test_two_level_table_after_fix_has_no_strong_trigger():
    triggers = _triggers(fx.two_level_score_xlsx())
    assert not [code for code, item in triggers.items() if item["strong"]]
    # 序号列是连续编号，不算像分值的列
    assert "序号" not in (triggers.get("E3") or {"details": {}})["details"].get("numeric_columns", [])


def test_wrong_parent_sum_triggers_e10():
    triggers = _triggers(fx.two_level_score_xlsx(parent_scores=(12, 15)))
    assert triggers["E10"]["strong"] is True
    assert triggers["E10"]["details"]["columns"] == [3, 5]


def test_header_declared_total_mismatch_triggers_e4():
    triggers = _triggers(fx.two_level_score_xlsx(total_label="合计"))
    assert "E4" not in triggers
    book = _book([["本表满分 100 分"], ["评分项", "分值"], ["选题", 10], ["写作", 20]])
    triggers = _triggers(book)
    assert triggers["E4"]["details"] == {"declared_total": 100.0, "parsed_total": 30.0, "source": "表头声明的满分"}


def test_shared_score_cell_triggers_e11_and_numeric_unmapped_column_is_strong():
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["评分项", "分值", "参考分"])
    sheet.append(["选题", 10, 6])
    sheet.append(["写作", None, 4])
    sheet.merge_cells("B2:B3")
    buffer = BytesIO()
    workbook.save(buffer)
    triggers = _triggers(buffer.getvalue())
    assert triggers["E11"]["strong"] is True
    assert triggers["E11"]["details"]["criterion_codes"] == ["C01", "C02"]
    assert triggers["E3"]["details"]["numeric_columns"] == ["参考分"]
    assert triggers["E3"]["strong"] is True


@pytest.mark.parametrize("numbers,strong", [
    ([1, 2, 3, 1, 2], False),  # 分组重新编号
    ([1, 2, 3, 4], False),     # 连续编号
    ([1, 1, 1], True),         # 都是 1 分：像分值
    ([5, 3, 2], True),
])
def test_unmapped_numbering_is_not_a_strong_signal(numbers, strong):
    rows = [["评分项", "分值", "序号"]] + [[f"项{i}", 10, n] for i, n in enumerate(numbers)]
    triggers = _triggers(_book(rows))
    assert triggers["E3"]["strong"] is strong

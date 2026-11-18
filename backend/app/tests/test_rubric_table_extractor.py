from io import BytesIO

import pytest
from openpyxl import Workbook

from backend.app.services.rubric_import.coverage import compute_coverage
from backend.app.services.rubric_import.extraction.table_extractor import extract_table
from backend.app.services.rubric_import.sources.units import SourceLedger
from backend.app.services.rubric_import.sources.xlsx_adapter import load_xlsx
from backend.app.tests import rubric_parse_fixtures as fx


def _extract(data):
    ledger = SourceLedger()
    sheets = load_xlsx(data, ledger, doc_id="rules", doc_role="rules")
    return extract_table(sheets, ledger), ledger


def test_simple_table_is_fully_claimed():
    result, ledger = _extract(fx.simple_rules_xlsx())
    assert result.sheet_title == "评分规则"
    assert [row.criterion.name for row in result.records] == ["研究方法", "文献综述"]
    assert [row.row_number for row in result.records] == [2, 3]
    assert ledger.status("xlsx:评分规则!R1C2").status == "structural"
    assert ledger.status("xlsx:评分规则!R2C3").claimed_by == ("C01.max_score",)
    assert compute_coverage(ledger)["documents"][0]["ratio"] == 1.0
    assert result.unmapped_columns == []


def test_complex_table_reports_losses():
    result, ledger = _extract(fx.complex_rules_xlsx())
    assert [row.criterion.name for row in result.records] == ["选题意义", "文献综述", "方法设计"]
    # 合并的维度单元被两个评分项认领
    assert ledger.status("xlsx:评分表!R4C1").claimed_by == ("C01.dimension", "C02.dimension")
    # 未映射列“备注”的内容未被认领
    assert ledger.status("xlsx:评分表!R5C6").status == "unclaimed"
    # R5 的备注 + R8 整行合并（A:F）覆盖到备注列
    assert result.unmapped_columns == [{"column": 6, "header": "备注", "non_empty": 2, "data_rows": 5}]
    # 无满分行与整行合并的说明行被丢弃并记录
    assert [(row["row_number"], row["reason"]) for row in result.dropped_rows] == [
        (7, "未找到满分"), (8, "整行合并的说明行")]
    assert ledger.status("xlsx:评分表!R8C1").status == "unclaimed"
    assert ledger.status("xlsx:评分表!R7C2").status == "unclaimed"
    # 合计行为结构性内容，并给出声明总分
    assert ledger.status("xlsx:评分表!R9C1").status == "structural"
    assert result.total_row == {"row_number": 9, "declared_total": 50.0}
    # 表头上方的标题未被认领；其他工作表按规则忽略
    assert ledger.status("xlsx:评分表!R1C1").status == "unclaimed"
    assert ledger.status("xlsx:附加扣分!R2C3").status == "ignored_by_rule"
    assert result.ignored_sheets == [{"title": "附加扣分", "reason": "仅解析第一张识别到评分项的工作表"}]


def test_sheet_without_header_before_rules_sheet_is_ignored_with_warning():
    workbook = Workbook()
    workbook.active.title = "说明"
    workbook.active.append(["本表用于毕业论文评分"])
    sheet = workbook.create_sheet("规则")
    sheet.append(["评分项", "分值"])
    sheet.append(["选题", 10])
    buffer = BytesIO()
    workbook.save(buffer)
    result, ledger = _extract(buffer.getvalue())
    assert result.sheet_title == "规则"
    assert result.warnings == ["工作表 说明 未识别到评分规则表头，已跳过。"]
    assert ledger.status("xlsx:说明!R1C1").status == "ignored_by_rule"


def test_no_rules_sheet_raises_same_error_as_before():
    with pytest.raises(ValueError, match="Excel 未解析到有效评分项"):
        _extract(fx.unmapped_header_xlsx())


def test_custom_header_aliases_are_supported():
    ledger = SourceLedger()
    sheets = load_xlsx(fx.unmapped_header_xlsx(), ledger, doc_id="rules", doc_role="rules")
    result = extract_table(sheets, ledger, header_aliases={"name": ["考核内容"], "max_score": ["配分"]})
    assert [row.criterion.name for row in result.records] == ["选题"]
    assert result.unmapped_columns == [{"column": 3, "header": "细则", "non_empty": 1, "data_rows": 1}]


def test_header_aliases_are_resolved_per_profile_with_thesis_defaults():
    from backend.app.services.rubric_import.extraction.table_extractor import HEADER_ALIASES
    from backend.app.services.rubric_import.extraction.table_extractor import PROFILE_HEADER_ALIASES
    from backend.app.services.rubric_import.extraction.table_extractor import header_aliases_for

    assert header_aliases_for("thesis") == HEADER_ALIASES
    assert header_aliases_for(None) == HEADER_ALIASES
    PROFILE_HEADER_ALIASES["demo_profile"] = {"max_score": ["配分"]}
    try:
        merged = header_aliases_for("demo_profile")
        assert merged["max_score"][-1] == "配分"
        assert merged["name"] == HEADER_ALIASES["name"]
        assert HEADER_ALIASES["max_score"][-1] != "配分"  # 默认表不被修改
    finally:
        PROFILE_HEADER_ALIASES.pop("demo_profile")


def test_merged_parent_name_is_reinterpreted_as_dimension():
    result, ledger = _extract(fx.merged_parent_dimension_xlsx())

    assert result.mapping == {"item_label": 0, "description": 2, "dimension": 1}
    assert [row.criterion.code for row in result.records] == ["T02", "T03", "T04", "T05"]
    assert [row.criterion.name for row in result.records] == [
        "分析与解决问题1", "分析与解决问题2", "分析与解决问题3", "分析与解决问题4",
    ]
    assert {row.criterion.dimension for row in result.records} == {"分析与解决问题"}
    assert [row.criterion.max_score for row in result.records] == [20.0, 20.0, 10.0, 10.0]
    assert ledger.status("xlsx:Sheet!R2C2").claimed_by == (
        "T02.dimension", "T03.dimension", "T04.dimension", "T05.dimension",
    )
    assert result.structure_issues == []
    assert result.warnings == ["检测到纵向合并的父级评价项目，已按评分维度解析。"]


def test_independent_duplicate_names_are_not_reinterpreted_as_dimension():
    result, _ = _extract(fx.merged_parent_dimension_xlsx(independently_repeated=True))

    assert result.mapping["name"] == 1
    assert "dimension" not in result.mapping
    assert {row.criterion.name for row in result.records} == {"分析与解决问题"}
    assert result.structure_issues == []


def test_merged_name_does_not_overwrite_existing_dimension_mapping():
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["打分项", "评价项目", "评分维度", "具体要求"])
    sheet.append(["指导教师成绩项2（20分）", "分析与解决问题", "专业能力", "要求A"])
    sheet.append(["指导教师成绩项3（10分）", "分析与解决问题", "专业能力", "要求B"])
    sheet.merge_cells("B2:B3")
    buffer = BytesIO()
    workbook.save(buffer)

    result, _ = _extract(buffer.getvalue())

    assert result.mapping == {"item_label": 0, "name": 1, "dimension": 2, "description": 3}
    assert {row.criterion.name for row in result.records} == {"分析与解决问题"}
    assert {row.criterion.dimension for row in result.records} == {"专业能力"}
    assert result.structure_issues[0]["code"] == "MERGED_NAME_AMBIGUOUS"
    assert result.structure_issues[0]["reason"] == "已经存在维度列，无法自动重新解释合并名称列。"


def test_duplicate_score_headers_pick_item_column_and_parent_dimension():
    result, ledger = _extract(fx.two_level_score_xlsx())
    rows = [(row.criterion.name, row.criterion.max_score, row.criterion.dimension) for row in result.records]
    assert rows == [
        ("封面信息", 4.0, "一、基本信息"), ("填写规范", 2.0, "一、基本信息"), ("论文题目", 4.0, "一、基本信息"),
        ("选题背景", 8.0, "二、选题背景"), ("研究目的", 7.0, "二、选题背景"),
    ]
    assert result.mapping["max_score"] == 4 and result.mapping["dimension"] == 1
    assert result.dimension_score_columns == [2]
    assert result.structure_issues == []
    # 带括号说明的合计行不是评分项，它声明的满分进入核对
    assert result.total_row == {"row_number": 10, "declared_total": 25.0}
    assert result.declared_total["value"] == 25.0
    # “打分规则（分档 / 扣分）”是规则正文，不拆成档位
    first = result.records[0].criterion
    assert first.description == "六项齐全；每缺1项扣1分，扣完为止。"
    assert first.rubric_levels == [] and first.scoring_mode == "llm_direct"
    assert ledger.status("xlsx:评分表!R5C3").status == "structural"
    assert 3 not in [column["column"] for column in result.unmapped_columns]


def test_duplicate_score_headers_with_wrong_parent_sum_are_reported():
    result, _ = _extract(fx.two_level_score_xlsx(parent_scores=(12, 15)))
    # 仍按每行独立的列解析，但记一条结构问题交给模型核对
    assert [row.criterion.max_score for row in result.records] == [4.0, 2.0, 4.0, 8.0, 7.0]
    assert [issue["code"] for issue in result.structure_issues] == ["DUPLICATE_SCORE_COLUMN"]


@pytest.mark.parametrize("label,is_total", [
    ("合计", True), ("合计（满分 100 分）", True), ("总分：100", True), ("总计 ", True),
    ("总计划安排", False), ("总分析", False), ("综合合计", False),
])
def test_total_name_prefix(label, is_total):
    from backend.app.services.rubric_import.extraction.table_extractor import is_total_name

    assert is_total_name(label) is is_total


def test_band_header_matched_by_containment_falls_back_to_description_for_prose():
    workbook = Workbook()
    workbook.active.append(["评分项", "分值", "分档与扣分"])
    workbook.active.append(["格式", 5, "每类不符扣1分，扣完为止。"])
    workbook.active.append(["文字", 3, "错别字每处扣0.5分，超过10处扣3分。"])
    buffer = BytesIO()
    workbook.save(buffer)
    result, _ = _extract(buffer.getvalue())
    assert "rubric_levels" not in result.mapping
    assert [row.criterion.description for row in result.records] == [
        "每类不符扣1分，扣完为止。", "错别字每处扣0.5分，超过10处扣3分。"]


def test_band_header_with_tidy_levels_stays_levels():
    workbook = Workbook()
    workbook.active.append(["评分项", "分值", "分档"])
    workbook.active.append(["选题", 10, "优:10 良:8 中:6"])
    buffer = BytesIO()
    workbook.save(buffer)
    result, _ = _extract(buffer.getvalue())
    assert result.records[0].criterion.rubric_levels == [
        {"label": "优", "points": 10.0}, {"label": "良", "points": 8.0}, {"label": "中", "points": 6.0}]


def _book(build):
    workbook = Workbook()
    build(workbook.active)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _two_level(sheet, item_scores=(4, 6), footer_rows=1):
    sheet.title = "评分表"
    sheet.append(["一级指标", "分值", "评分项", "分值"])  # R1
    sheet.append(["一、基本信息", 10, "封面信息", item_scores[0]])  # R2
    sheet.append([None, None, "填写规范", item_scores[1]])  # R3
    sheet.merge_cells("A2:A3")
    sheet.merge_cells("B2:B3")
    sheet.append(["评分说明：逐项打分，扣分不超过该项分值。"])  # R4
    sheet.merge_cells(f"A4:D{3 + footer_rows}")


def test_score_text_with_unit_counts_as_item_score_column():
    result, _ = _extract(_book(lambda sheet: _two_level(sheet, item_scores=("4分", "6分"))))
    assert [row.criterion.max_score for row in result.records] == [4.0, 6.0]
    assert result.mapping["max_score"] == 3 and result.dimension_score_columns == [1]


def test_score_text_with_wrong_parent_sum_is_reported():
    result, _ = _extract(_book(lambda sheet: _two_level(sheet, item_scores=("4分", "5分"))))
    assert [row.criterion.max_score for row in result.records] == [4.0, 5.0]
    assert [issue["code"] for issue in result.structure_issues] == ["DUPLICATE_SCORE_COLUMN"]


def test_footer_spanning_rows_and_columns_does_not_hide_item_score_column():
    result, _ = _extract(_book(lambda sheet: _two_level(sheet, footer_rows=2)))
    assert [row.criterion.max_score for row in result.records] == [4.0, 6.0]
    assert result.structure_issues == []


@pytest.mark.parametrize("label,declared", [("总分：100", 100.0), ("合计 100 分", 100.0), ("合计（满分 100 分）", 100.0)])
def test_total_label_declares_value(label, declared):
    def build(sheet):
        sheet.append(["评分项", "分值"])
        sheet.append(["选题", 10])
        sheet.append([label, None])
    result, _ = _extract(_book(build))
    assert result.total_row == {"row_number": 3, "declared_total": declared}


def test_override_reparse_keeps_header_declared_total():
    from backend.app.services.rubric_import.extraction.structure_override import extract_with_override

    data = _book(lambda sheet: (sheet.append(["满分 100 分"]), sheet.append(["评分项", "分值"]),
                                sheet.append(["选题", 10]), sheet.append(["写作", 20])))
    ledger = SourceLedger()
    sheets = load_xlsx(data, ledger, doc_id="rules", doc_role="rules")
    override = {"sheet": sheets[0].title, "header_row": 2, "column_mapping": {"name": 1, "max_score": 2}, "row_types": {}}
    result = extract_with_override(sheets, ledger, override)
    assert result.declared_total["value"] == 100.0
    assert ledger.status(result.declared_total["unit_ids"][0]).status == "structural"

"""章节树、按范围原样取证与稳定前缀（docs/论文章节树与稳定前缀取证方案.md 阶段 1、2）。

全部用合成数据，不调用真实模型。
"""

from __future__ import annotations

import json

from backend.app.core.config import settings
from backend.app.services.llm import core_view
from backend.app.services.scoring.core.contracts import PromptEnvelopeV3
from backend.app.services.scoring.retrieval import section_tree
from backend.app.services.scoring.retrieval.selection import build_v4_envelope
from backend.app.tests import test_rule_scope_scoring as scoped_run
from backend.app.tests.test_scoring_stage2 import BUDGET
from backend.app.tests.test_scoring_stage2 import _base_mapping
from backend.app.tests.test_scoring_stage2 import _envelope
from backend.app.tests.test_scoring_stage2 import _scoped_rule
from backend.app.tests.test_scoring_stage2 import _unit


def _units(sections):
    content_hash = _base_mapping()["evidence_units"][0]["normalized_content_hash"]
    return [
        _unit(content_hash, path, section_ordinal, unit_ordinal, text)
        for section_ordinal, (path, texts) in enumerate(sections)
        for unit_ordinal, text in enumerate(texts)
    ]


def _thesis(units, rule):
    value = _envelope(units=units, rule=rule).to_mapping()
    value["runtime_identity"]["profile_key"] = "thesis"
    return PromptEnvelopeV3.from_mapping(value)


def _paragraph(index, chars):
    seed = "第%d段说明系统设计中的模块划分与接口约定。" % index
    return (seed * (chars // len(seed) + 1))[:chars]


LONG_CHAPTER = [
    (["第1章 绪论"], ["本文研究校园互助平台。"]),
    (["第3章 系统设计"], [_paragraph(index, 500) for index in range(30)]),
    (["第5章 结论"], ["本文完成了平台实现。"]),
]
SHORT_CHAPTER = [
    (["第1章 绪论"], ["本文研究校园互助平台。"]),
    (["第3章 系统设计", "3.1 总体架构"], [_paragraph(0, 600), "采用三层架构。"]),
    (["第3章 系统设计", "3.2 数据库设计"], ["共设计六张表。"]),
    (["第5章 结论"], ["本文完成了平台实现。"]),
]


# ── 章节树 ────────────────────────────────────────────────────────


def test_tree_follows_the_heading_paths_and_keeps_repeated_headings_apart():
    units = _units([
        (["第2章 需求", "小结"], ["甲"]),
        (["第3章 设计", "3.1 架构"], ["乙乙", "丙丙丙"]),
        (["第3章 设计", "小结"], ["丁"]),
    ])
    # 契约要求标题路径非空；拼树本身对没有标题的段落也不出错，归到一个占位节点下。
    untitled = {"section_path": [], "section_ordinal": 9, "unit_ordinal": 0,
                "normalized_text": "没有标题的段落", "evidence_unit_id": "x"}
    tree = section_tree.build_tree([*units, untitled])

    assert [node["title"] for node in tree] == ["第2章 需求", "第3章 设计", section_tree.UNTITLED]
    design = tree[1]
    assert (design["chars"], design["paragraphs"]) == (6, 3)
    assert [child["title"] for child in design["children"]] == ["3.1 架构", "小结"]
    # 两章各有一个“小结”：是两个节点，不会并在一起。
    assert tree[0]["children"][0]["unit_ids"] != design["children"][1]["unit_ids"]
    assert design["children"][0]["unit_ids"] == [units[1]["evidence_unit_id"], units[2]["evidence_unit_id"]]


def test_outline_lists_every_node_in_document_order_and_is_capped():
    rows = section_tree.outline(_units(SHORT_CHAPTER))
    assert [(row["title"], row["level"]) for row in rows] == [
        ("第1章 绪论", 1), ("第3章 系统设计", 1), ("3.1 总体架构", 2), ("3.2 数据库设计", 2), ("第5章 结论", 1),
    ]
    assert rows[1] == {"title": "第3章 系统设计", "level": 1, "chars": 600 + 7 + 7, "paragraphs": 3}

    many = _units([(["第%d章" % chapter, "%d.%d" % (chapter, part)], ["正文"])
                   for chapter in range(1, 31) for part in range(1, 4)])
    capped = section_tree.outline(many)
    # 放不下时先去掉最深的一层，章一级保留完整。
    assert len(capped) == 30 and {row["level"] for row in capped} == {1}


def test_stable_alias_depends_only_on_the_position_in_the_paper():
    units = _units(SHORT_CHAPTER)
    assert [section_tree.stable_alias(unit) for unit in units] == ["E0.0", "E1.0", "E1.1", "E2.0", "E3.0"]


# ── 阶段 1：按范围原样取证，“整章”名副其实 ─────────────────────────


def test_a_chapter_that_fits_is_sent_verbatim_in_document_order(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    units = _units(SHORT_CHAPTER)
    envelope = build_v4_envelope(_thesis(units, _scoped_rule("section", ["系统设计"])),
                                 top_k=1, judge_scope=True, **BUDGET)
    view = json.loads(core_view.build_core_request(envelope).user)

    # 600 字的段落超过旧的“每段 400 字”上限，现在原样给出，没有省略号。
    assert [item["text"] for item in view["evidence"]] == [unit["normalized_text"] for unit in units[1:4]]
    assert view["scope"]["sections"] == [
        {"role": "系统设计", "headings": ["第3章 系统设计"], "coverage": "complete"}
    ]


def test_a_chapter_over_the_character_budget_is_cut_and_marked_partial(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    monkeypatch.setattr(settings, "SCORING_SCOPE_EVIDENCE_CHAR_BUDGET", 2000)
    units = _units(LONG_CHAPTER)
    envelope = build_v4_envelope(
        _thesis(units, _scoped_rule("section", ["系统设计"])),
        top_k=12, judge_scope=True, scope_char_budget=2000, **BUDGET)
    value = envelope.to_mapping()
    request = core_view.build_core_request(envelope)
    view = json.loads(request.user)

    assert sum(len(unit["normalized_text"]) for unit in value["evidence_units"]) <= 2000
    # 选中的每一段都原样出现；视图里没有再丢段，所以“部分”与实际一致。
    assert len(view["evidence"]) == len(value["evidence_units"]) == 4
    assert all("…" not in item["text"] for item in view["evidence"])
    assert view["scope"]["sections"][0]["coverage"] == "partial"
    originals = {unit["normalized_text"] for unit in value["evidence_units"]}
    assert {item["text"] for item in view["evidence"]} == originals


def test_a_truncated_paragraph_makes_its_chapter_partial(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    monkeypatch.setattr(settings, "SCORING_SCOPE_EVIDENCE_CHAR_BUDGET", 500)
    units = _units([(["第3章 系统设计"], [_paragraph(0, 1500)])])
    envelope = build_v4_envelope(_thesis(units, _scoped_rule("section", ["系统设计"])),
                                 top_k=3, judge_scope=True, scope_char_budget=500, **BUDGET)
    value = envelope.to_mapping()
    view = json.loads(core_view.build_core_request(envelope).user)

    # 选证身份里这一章“选全了”，但唯一的一段被截短了：不能告诉模型这是整章。
    assert value["evidence_selection_identity"]["section_coverage"][0]["complete"] is True
    assert "…" in view["evidence"][0]["text"]
    assert view["scope"]["sections"][0]["coverage"] == "partial"


def test_scoped_requests_may_use_a_larger_context_window():
    units = _units(LONG_CHAPTER)
    rule = _scoped_rule("section", ["系统设计"])
    small = build_v4_envelope(_thesis(units, rule), top_k=12, judge_scope=True, **BUDGET).to_mapping()
    large = build_v4_envelope(_thesis(units, rule), top_k=12, judge_scope=True,
                              scope_context_window_tokens=14000, **BUDGET).to_mapping()

    assert small["token_budget_identity"]["context_window_tokens"] == BUDGET["context_window_tokens"]
    assert large["token_budget_identity"]["context_window_tokens"] == 14000
    assert len(large["evidence_units"]) > len(small["evidence_units"])
    # 全文范围的规则不用范围窗口。
    plain = build_v4_envelope(_thesis(units, {"name": "语言表达", "rule_text": "存在口语化表达"}),
                              top_k=3, judge_scope=True, scope_context_window_tokens=14000, **BUDGET).to_mapping()
    assert plain["token_budget_identity"]["context_window_tokens"] == BUDGET["context_window_tokens"]


def test_the_outline_is_shown_with_the_paper_and_frozen_by_the_executor(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    units = _units(SHORT_CHAPTER)
    value = _thesis(units, _scoped_rule("section", ["系统设计"])).to_mapping()
    value["profile_prompt_extensions"]["section_outline"] = section_tree.section_outline(units)
    envelope = build_v4_envelope(PromptEnvelopeV3.from_mapping(value), top_k=1, judge_scope=True, **BUDGET)
    request = core_view.build_core_request(envelope)

    assert json.loads(request.user)["paper"]["outline"] == section_tree.outline(units)
    assert "paper.outline" in request.system and "不能作为证据" in request.system

    # 执行器：有规则要看章节时，目录骨架与章节角色一起写进信封扩展。
    rule = scoped_run._scoped_rule("section", ["总体方案"])
    response = scoped_run._semantic_response(rule["rule_code"], status="not_triggered")
    _result, runtime, _request = scoped_run._run(rule, {rule["rule_code"]: response})
    extensions = runtime.envelopes[0]["profile_prompt_extensions"]
    assert extensions["section_outline"]["version"] == section_tree.SECTION_TREE_VERSION
    assert extensions["section_outline"]["rows"]


# ── 阶段 2：稳定前缀 ──────────────────────────────────────────────


def _request(units, rule, **options):
    envelope = build_v4_envelope(_thesis(units, rule), top_k=3, **{**BUDGET, **options})
    return core_view.build_core_request(envelope)


def test_rules_reading_the_same_chapter_share_one_prefix(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_JUDGE_SCOPE_ENABLED", True)
    monkeypatch.setattr(settings, "SCORING_STABLE_PREFIX_ENABLED", True)
    units = _units(SHORT_CHAPTER)
    first = _request(units, _scoped_rule("section", ["系统设计"], name="架构说明", rule_text="没有说明总体架构"),
                     judge_scope=True)
    second = _request(units, _scoped_rule("section", ["系统设计"], name="数据库设计", rule_text="表结构设计缺少字段说明"),
                      judge_scope=True)

    assert first.system == second.system
    assert list(json.loads(first.user)) == ["evidence", "criterion", "scope", "rule"]
    cut = first.user.index('"criterion"')
    # 到证据结束为止逐字相同；之后才是随规则变化的内容。
    assert cut > 600 and first.user[:cut] == second.user[:cut]
    assert first.user != second.user
    assert list(first.alias_to_unit) == ["E1.0", "E1.1", "E2.0"]
    assert "E3.2" in first.system and core_view.provider_prompt_suffix() == "+rule-scope@2+stable-prefix@1+insufficient@1"


def test_stable_prefix_lists_ranked_evidence_in_document_order(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_STABLE_PREFIX_ENABLED", True)
    units = _units(SHORT_CHAPTER)
    request = _request(units, {"name": "结论呼应", "rule_text": "结论没有回应绪论提出的问题"})
    refs = [item["ref"] for item in json.loads(request.user)["evidence"]]

    assert refs == sorted(refs, key=lambda ref: tuple(int(part) for part in ref[1:].split(".")))
    assert set(request.alias_to_unit.values()) <= {unit["evidence_unit_id"] for unit in units}
    assert request.schema["properties"]["occurrences"]["items"]["properties"]["evidence"]["items"][
        "properties"]["evidence_ref"]["enum"] == refs


def test_both_switches_off_keep_the_previous_request():
    units = _units(SHORT_CHAPTER)
    request = _request(units, {"name": "结论呼应", "rule_text": "结论没有回应绪论提出的问题"})
    view = json.loads(request.user)

    assert list(view) == ["criterion", "evidence", "rule"]
    assert [item["ref"] for item in view["evidence"]] == ["E1", "E2", "E3"]
    assert request.system == core_view._SINGLE_INSTRUCTIONS


# ── 阶段 2：缓存命中计量 ──────────────────────────────────────────


def test_cached_prompt_tokens_are_read_from_each_protocol_and_summed():
    from backend.app.services.llm.anthropic_messages_adapter import observed_usage_from_messages
    from backend.app.services.llm.openai_adapter import _observed_usage_from_responses
    from backend.app.services.llm.openai_compatible_adapter import _observed_usage_from_chat
    from backend.app.services.llm.usage import UsageMeter

    chat = _observed_usage_from_chat({
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920,
                  "prompt_tokens_details": {"cached_tokens": 640}},
    })
    deepseek = _observed_usage_from_chat({
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920,
                  "prompt_cache_hit_tokens": 512},
    })
    silent = _observed_usage_from_chat({
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920},
    })
    responses = _observed_usage_from_responses({
        "status": "completed", "output": [],
        "usage": {"input_tokens": 900, "output_tokens": 20, "total_tokens": 920,
                  "input_tokens_details": {"cached_tokens": 300}},
    })
    claude = observed_usage_from_messages({
        "content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "cache_read_input_tokens": 800, "output_tokens": 20},
    })
    assert [item["cached_tokens"] for item in (chat, deepseek, silent, responses, claude)] == [640, 512, 0, 300, 800]

    meter = UsageMeter()
    for usage in (chat, deepseek, silent):
        meter.record_success(usage)
    assert meter.snapshot()["cached_tokens"] == 1152

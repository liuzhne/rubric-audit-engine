"""章节角色词表与范围校验（docs/评分细则适用范围方案.md 第 3 节）。

起草时还没有论文，模型写不出真实标题，所以范围用“章节角色”表示：每个 Profile 一张固定词表，
严格 Schema 用枚举限定。评分时再把论文标题对到角色，``headings`` 是标题里的同义说法。

词表改动（增删角色、改名、改同义词）要升 ``VOCABULARY_VERSION``：它写进每条细则的范围记录，
评分时也进入证据选择身份。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


VOCABULARY_VERSION = "section-roles@1"

DOCUMENT = "document"
SECTION = "section"
CROSS_SECTION = "cross_section"
SCOPE_TYPES = (DOCUMENT, SECTION, CROSS_SECTION)
MAX_CROSS_SECTIONS = 3
# ai_draft：起草细则时模型返回；ai_identified：范围识别任务补上（不经过起草的规则）。
SCOPE_SOURCES = ("ai_draft", "ai_identified")

_TYPE_TEXT = {
    DOCUMENT: "全文：规则看整篇，或说不清只看哪一章",
    SECTION: "章节内：规则只评价某一章的内容",
    CROSS_SECTION: "跨章节：规则要对照两到三章之间是否一致、呼应",
}


@dataclass(frozen=True)
class SectionRole:
    """一个章节角色：给模型看的说明，以及论文标题里的同义说法。"""

    name: str
    description: str
    headings: tuple[str, ...]


_THESIS = (
    SectionRole("摘要", "摘要与关键词", ("摘要", "abstract")),
    SectionRole("绪论", "研究背景、问题、目标与意义", ("绪论", "引言", "前言", "导论", "研究背景", "introduction")),
    SectionRole("文献综述", "国内外研究现状、相关工作与理论基础",
                ("文献综述", "相关工作", "研究现状", "国内外研究", "理论基础", "相关理论", "相关技术", "related work")),
    SectionRole("研究方法", "研究设计、方法、数据与工具", ("研究方法", "研究设计", "研究方案", "研究思路", "methodology")),
    SectionRole("需求分析", "需求、可行性与问题分析", ("需求分析", "可行性分析", "需求")),
    SectionRole("系统设计", "总体设计、详细设计、架构与数据库设计",
                ("系统设计", "总体设计", "详细设计", "方案设计", "架构设计", "数据库设计", "概要设计")),
    SectionRole("系统实现", "功能与模块的实现", ("系统实现", "功能实现", "模块实现", "实现")),
    SectionRole("测试与实验", "测试、实验、结果与分析",
                ("系统测试", "实验结果", "结果与分析", "结果分析", "实证分析", "数据分析", "实验", "测试")),
    SectionRole("结论", "结论、总结与展望", ("结论", "总结", "结语", "结束语", "展望", "conclusion")),
    SectionRole("参考文献", "参考文献列表", ("参考文献", "references", "bibliography")),
    SectionRole("致谢", "致谢", ("致谢", "谢辞", "acknowledg")),
    SectionRole("附录", "附录与附件", ("附录", "appendix")),
)

_GENERAL_DOCUMENT = (
    SectionRole("摘要", "摘要、概要或内容提要", ("摘要", "概要", "内容提要", "abstract", "summary")),
    SectionRole("引言", "引言、前言与写作目的", ("引言", "前言", "导言", "概述", "introduction")),
    SectionRole("背景", "背景、现状与问题提出", ("背景", "现状", "问题提出", "background")),
    SectionRole("方法", "方法、过程与做法", ("方法", "研究方法", "实施过程", "做法", "method")),
    SectionRole("结果", "结果、发现与数据", ("结果", "发现", "成效", "results")),
    SectionRole("讨论", "讨论、分析与建议", ("讨论", "分析", "建议", "对策", "discussion")),
    SectionRole("结论", "结论与总结", ("结论", "总结", "结语", "conclusion")),
    SectionRole("参考文献", "参考文献列表", ("参考文献", "references", "bibliography")),
    SectionRole("附录", "附录与附件", ("附录", "附件", "appendix")),
)

_TECHNICAL_PROPOSAL = (
    SectionRole("项目概述", "项目背景、目标与范围", ("项目概述", "项目简介", "项目背景", "项目概况", "概述")),
    SectionRole("需求分析", "需求、现状与问题分析", ("需求分析", "现状分析", "问题分析", "需求")),
    SectionRole("总体方案", "总体方案、技术路线与架构",
                ("总体方案", "技术方案", "解决方案", "总体设计", "技术路线", "系统架构", "架构")),
    SectionRole("详细设计", "模块、功能与接口设计", ("详细设计", "模块设计", "功能设计", "接口设计")),
    SectionRole("实施计划", "实施步骤、进度与里程碑",
                ("实施计划", "实施方案", "进度安排", "进度计划", "里程碑", "工作计划")),
    SectionRole("风险分析", "风险与保障措施", ("风险分析", "风险", "保障措施")),
    SectionRole("经费预算", "预算、经费与成本", ("经费预算", "预算", "经费", "成本", "报价")),
    SectionRole("团队与组织", "团队、人员与组织分工", ("团队与组织", "项目团队", "人员", "组织", "团队")),
    SectionRole("预期成果", "预期成果、效益与验收指标",
                ("预期成果", "预期效益", "考核指标", "验收", "成果", "效益")),
    SectionRole("附录", "附录与附件", ("附录", "附件", "appendix")),
)

_VOCABULARIES = {
    "thesis": _THESIS,
    "general_document": _GENERAL_DOCUMENT,
    "technical_proposal": _TECHNICAL_PROPOSAL,
}


class ScopeError(ValueError):
    """范围不合格：类型未知、章节数不对，或角色不在本 Profile 的词表里。"""


def vocabulary(profile_key) -> tuple[SectionRole, ...]:
    """Profile 的章节角色词表；未知 Profile 按毕业论文（与起草、评分的缺省 Profile 一致）。"""

    return _VOCABULARIES.get(str(profile_key or "thesis"), _THESIS)


def role_names(profile_key) -> list[str]:
    """词表里的角色名，按词表顺序。"""

    return [role.name for role in vocabulary(profile_key)]


def vocabulary_payload(profile_key) -> dict:
    """写进模型输入的词表说明：范围类型、可选角色及其含义。"""

    return {
        "version": VOCABULARY_VERSION,
        "types": dict(_TYPE_TEXT),
        "sections": [{"name": role.name, "covers": role.description} for role in vocabulary(profile_key)],
        "rules": "document 时 sections 为空数组；section 时恰好 1 个；cross_section 时 2 到 3 个且互不相同。",
    }


def scope_schema(profile_key) -> dict:
    """范围对象的严格 JSON Schema：类型与角色都用枚举限定。"""

    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(SCOPE_TYPES)},
            "sections": {
                "type": "array",
                "maxItems": MAX_CROSS_SECTIONS,
                "items": {"type": "string", "enum": role_names(profile_key)},
            },
        },
        "required": ["type", "sections"],
        "additionalProperties": False,
    }


def normalize_scope(value, profile_key) -> dict:
    """校验模型返回的范围，返回 ``{"type", "sections"}``；不合格抛 ``ScopeError``。"""

    if not isinstance(value, Mapping):
        raise ScopeError("适用范围必须是对象")
    scope_type = str(value.get("type") or "").strip()
    if scope_type not in SCOPE_TYPES:
        raise ScopeError("适用范围的类型无效")
    raw_sections = value.get("sections")
    if raw_sections is None:
        raw_sections = []
    if not isinstance(raw_sections, list):
        raise ScopeError("适用范围的章节必须是数组")
    sections = [str(item).strip() for item in raw_sections]
    allowed = set(role_names(profile_key))
    if any(item not in allowed for item in sections) or len(set(sections)) != len(sections):
        raise ScopeError("适用范围引用了词表之外或重复的章节")
    if scope_type == DOCUMENT and sections:
        raise ScopeError("全文范围不能指定章节")
    if scope_type == SECTION and len(sections) != 1:
        raise ScopeError("章节内范围必须恰好指定 1 个章节")
    if scope_type == CROSS_SECTION and not 2 <= len(sections) <= MAX_CROSS_SECTIONS:
        raise ScopeError(f"跨章节范围必须指定 2 到 {MAX_CROSS_SECTIONS} 个章节")
    return {"type": scope_type, "sections": sections}


def stored_scope(value, profile_key, *, source) -> dict:
    """写进 ``atomic_rules.scope`` 的记录：范围本身加来源与词表版本。"""

    if source not in SCOPE_SOURCES:
        raise ScopeError("适用范围的来源无效")
    return {**normalize_scope(value, profile_key), "source": source, "vocabulary": VOCABULARY_VERSION}


def read_scope(stored) -> dict | None:
    """读库里的范围记录；空或形状不对一律返回 None（按全文处理，方案 Q5）。"""

    if not isinstance(stored, Mapping):
        return None
    scope_type = stored.get("type")
    sections = stored.get("sections")
    if scope_type not in SCOPE_TYPES or not isinstance(sections, list):
        return None
    return {
        "type": scope_type,
        "sections": [str(item) for item in sections],
        "source": stored.get("source"),
        "vocabulary": stored.get("vocabulary"),
    }


def describe_scope(stored) -> str:
    """给页面和日志用的中文说明。"""

    scope = read_scope(stored)
    if scope is None or scope["type"] == DOCUMENT:
        return "全文"
    label = "章节内" if scope["type"] == SECTION else "跨章节"
    return f"{label}：{'、'.join(scope['sections'])}"

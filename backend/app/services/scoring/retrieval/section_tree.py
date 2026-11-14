"""论文章节树（docs/论文章节树与稳定前缀取证方案.md 阶段 1）。

论文快照里每个证据单元带标题路径 ``section_path`` 与章节序号、段落序号，层级信息已经齐全。
这里只在读取时用代码把它拼成树，不调用模型、不另外存储：树的形状由原文唯一决定，随时可以重算。

- ``build_tree``：按原文顺序拼树。同一父节点下，标题相同且相邻的章节并成一个节点；中间隔了别的
  章节再出现的同名标题（例如每章各有一个“小结”）是不同节点。
- ``outline``：目录骨架——每个节点的标题、层级、字数、段落数，放在请求开头给同一篇论文的所有
  规则共用。只用来了解全文结构，不是证据。
- ``stable_alias``：按章节序号与段落序号给证据起别名，同一段在不同请求里的写法相同。
"""

from __future__ import annotations


SECTION_TREE_VERSION = "section-tree@1"
MAX_OUTLINE_ROWS = 60
MAX_OUTLINE_LEVEL = 3
MAX_OUTLINE_TITLE_CHARS = 30
UNTITLED = "（无标题）"


def unit_order(unit) -> tuple:
    return (int(unit.get("section_ordinal") or 0), int(unit.get("unit_ordinal") or 0))


def stable_alias(unit) -> str:
    """``E<章节序号>.<段落序号>``：只由段落在论文里的位置决定。"""

    return "E%d.%d" % unit_order(unit)


def _node(title, level):
    return {"title": title, "level": level, "chars": 0, "paragraphs": 0, "unit_ids": [], "children": []}


def build_tree(units) -> list[dict]:
    """顶层节点列表；每个节点带 ``children``，``chars`` / ``paragraphs`` 含全部下级。"""

    roots: list[dict] = []
    for unit in sorted(units or (), key=unit_order):
        path = [str(item) for item in unit.get("section_path") or ()] or [UNTITLED]
        chain = []
        siblings = roots
        for level, title in enumerate(path, start=1):
            # 只和同一父节点下的最后一个节点合并：按原文顺序走，最后一个就是紧挨着的那个，
            # 中间隔了别的章节再出现的同名标题会另起一个节点。
            if siblings and siblings[-1]["title"] == title:
                node = siblings[-1]
            else:
                node = _node(title, level)
                siblings.append(node)
            chain.append(node)
            siblings = node["children"]
        size = len(unit.get("normalized_text") or "")
        for node in chain:
            node["chars"] += size
            node["paragraphs"] += 1
        chain[-1]["unit_ids"].append(unit.get("evidence_unit_id"))
    return roots


def _walk(nodes):
    for node in nodes:
        yield node
        yield from _walk(node["children"])


def outline(units) -> list[dict]:
    """目录骨架：``[{title, level, chars, paragraphs}]``，按原文顺序。

    行数超过上限时先去掉最深的一层（它们的字数已经计入上级），直到放得下。
    """

    rows = [node for node in _walk(build_tree(units)) if node["level"] <= MAX_OUTLINE_LEVEL]
    level = MAX_OUTLINE_LEVEL
    while len(rows) > MAX_OUTLINE_ROWS and level > 1:
        level -= 1
        rows = [node for node in rows if node["level"] <= level]
    return [
        {
            "title": node["title"][:MAX_OUTLINE_TITLE_CHARS],
            "level": node["level"],
            "chars": node["chars"],
            "paragraphs": node["paragraphs"],
        }
        for node in rows[:MAX_OUTLINE_ROWS]
    ]


def section_outline(units) -> dict:
    """写进信封扩展 ``section_outline`` 的记录（进入判定身份，同一篇论文的规则共用）。"""

    return {"version": SECTION_TREE_VERSION, "rows": outline(units)}


__all__ = [
    "MAX_OUTLINE_ROWS",
    "SECTION_TREE_VERSION",
    "build_tree",
    "outline",
    "section_outline",
    "stable_alias",
    "unit_order",
]

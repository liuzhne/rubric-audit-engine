"""选择卡片：让用户在几个对象里点一个（docs/评分助手选择卡片方案.md）。

一种卡片 `choice` 承担所有“在几个对象里挑一个”：Agent 的回复用 `choose` 标明用途与候选编号，规则模式的
查询图直接给候选。选项（对象 ID 与显示文字）一律来自工具结果，不经模型转述；点了之后做什么按用途登记
（处理函数在 `runner`，那里才有流程与查询）。卡片不绑定角色：谁在用助手，就是谁在选。

用途在这里登记一次：对象种类、给模型看的说明、当前能不能用。加一种用途只改这张表、提示词里的清单与
`runner` 的处理函数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from backend.app.services.assistant.context import EVENT_INTENT
from backend.app.services.assistant.context import RunContext

MIN_OPTIONS = 2
MAX_OPTIONS = 10
#: 规则模式选材料：一个任务最多 30 份，全部列出（与原来的 choose_paper 一致）。
MAX_PAPER_OPTIONS = 30
QUESTION_CHARS = 200

#: 上传之后任务就锁定了评分标准（v2 界面改造方案 §4.3），这些等待中不能换。
RUBRIC_LOCKED_KINDS = frozenset({"upload_papers", "confirm_start", "confirm_remove", "wait_job"})


@dataclass(frozen=True)
class Purpose:
    key: str
    kind: str  # 引用编号的种类：r 评分标准、t 任务、p 材料
    title: str  # 卡片标题与历史里的叫法
    describe: str  # 给模型看：选来做什么
    unavailable: Callable[[RunContext], str | None] = lambda _ctx: None


def _rubric_unavailable(ctx: RunContext) -> str | None:
    if (ctx.conversation.pending or {}).get("kind") in RUBRIC_LOCKED_KINDS:
        return "文件已经上传，这次评分锁定了评分标准，现在不能换；要换标准请新开一次评分"
    return None


PURPOSES: dict[str, Purpose] = {
    purpose.key: purpose for purpose in (
        Purpose("use_rubric", "r", "选择评分标准",
                "用哪份评分标准：正在等选文件时就换成它，没有在评分时就用它开始评分",
                _rubric_unavailable),
        Purpose("focus_task", "t", "选择任务", "说的是哪个任务：选中后接着回答用户刚才的问题"),
        Purpose("focus_paper", "p", "选择材料", "说的是哪一份材料：选中后接着回答用户刚才的问题"),
    )
}


def unavailable(ctx: RunContext, purpose: str) -> str | None:
    found = PURPOSES.get(purpose)
    return "没有这种用途" if found is None else found.unavailable(ctx)


def card(purpose: str, options: list[dict], *, question: str | None = None, next_query: dict | None = None,
         limit: int = MAX_OPTIONS) -> dict:
    """生成一张选择卡片。`options`：[{id, label, current?, batch_id?}]，顺序就是显示顺序。

    `question` 是用户刚才那句话（Agent 选完接着回答它）；`next_query` 是规则模式选完接着跑的查询
    （`{"intent", "ordinal", "name", "text"}`），两者至多一个。
    """

    built = {
        "type": "choice", "status": "proposed", "purpose": purpose, "title": PURPOSES[purpose].title,
        "options": [{key: value for key, value in option.items() if value not in (None, False)}
                    for option in options[:limit]],
    }
    if question:
        built["question"] = str(question)[:QUESTION_CHARS]
    if next_query:
        built["next_query"] = {key: value for key, value in next_query.items() if value not in (None, "")}
    return built


def current_id(ctx: RunContext, purpose: str) -> str | None:
    """正在用的那个对象：评分标准看等待中的选文件卡片，任务与材料看焦点。"""

    focus = ctx.conversation.focus or {}
    if purpose == "focus_task":
        return focus.get("batch_id")
    if purpose == "focus_paper":
        return focus.get("paper_id")
    pending = ctx.conversation.pending or {}
    if purpose == "use_rubric" and pending.get("kind") == "pick_files":
        _message, picked = ctx.find_card(pending.get("card_id"))
        return (picked or {}).get("rubric_id")
    return None


def option(card_value: dict, option_id: str | None) -> dict | None:
    return next((item for item in card_value.get("options") or [] if item.get("id") == option_id), None)


def chosen_label(card_value: dict) -> str | None:
    picked = option(card_value, (card_value.get("result") or {}).get("id"))
    return picked.get("label") if picked else None


# --- 打字作答的兜底（方案 §6.5） --------------------------------------------------------

_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_ORDINAL = re.compile(r"^(?:选|用|要|就)?第?([0-9]{1,2}|[一二两三四五六七八九十])(?:个|份|项|号|条|种)?(?:吧|的|那个|那份)?$")
_TRIM = "。！!，,？? 　"
_NUMBERED = re.compile(r"^([0-9]{1,2})\. ")


def ordinal(text: str) -> int | None:
    found = _ORDINAL.match(str(text or "").strip(_TRIM))
    if not found:
        return None
    value = found.group(1)
    return int(value) if value.isdigit() else _DIGITS[value]


def _core(label: str) -> str:
    """显示文字里的名称：“《开题报告》v1.0（通用文档）”→“开题报告”。"""

    found = re.search(r"《(.+?)》", label)
    return (found.group(1) if found else _NUMBERED.sub("", label)).strip()


def match_typed(card_value: dict, text: str) -> dict | None:
    """用户没点按钮、打字回答：序号按卡片上的顺序，名称要能唯一对上。对不上返回 None（交给模型）。"""

    options = card_value.get("options") or []
    number = ordinal(text)
    if number is not None:
        # 选项自带序号（材料按上传顺序编号，“3. 张三”）时按那个序号，否则按卡片上的位置。
        numbered = {int(found.group(1)): item for item in options
                    if (found := _NUMBERED.match(item.get("label") or ""))}
        if numbered:
            return numbered.get(number)
        return options[number - 1] if 1 <= number <= len(options) else None
    said = str(text or "").strip(_TRIM)
    for prefix in ("就用", "选", "用", "要"):
        if said.startswith(prefix) and len(said) > len(prefix):
            said = said[len(prefix):]
            break
    said = said.strip("《》 ")
    for suffix in ("那个", "那份", "这个", "这份", "吧"):
        if said.endswith(suffix) and len(said) > len(suffix):
            said = said[: -len(suffix)]
    if len(said) < 2:
        return None
    hits = [item for item in options if said in _core(item.get("label") or "") or _core(item.get("label") or "") == said]
    return hits[0] if len(hits) == 1 else None


def latest_open(ctx: RunContext) -> tuple[object, dict] | tuple[None, None]:
    """最新一条助手消息里还没选的选择卡片。更早的不算：用户的话多半已经不是在回答它了。"""

    for message in reversed(ctx.conversation.messages):
        if message.role != "assistant" or message.intent == EVENT_INTENT:
            continue
        for item in message.cards or []:
            if item.get("type") == "choice" and item.get("status") == "proposed":
                return message, item
        return None, None
    return None, None


__all__ = ["MAX_OPTIONS", "MAX_PAPER_OPTIONS", "MIN_OPTIONS", "PURPOSES", "Purpose", "RUBRIC_LOCKED_KINDS", "card", "chosen_label",
           "current_id", "latest_open", "match_typed", "option", "ordinal", "unavailable"]

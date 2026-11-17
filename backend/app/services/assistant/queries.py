"""评分助手的查询图（LangGraph，无状态，方案 §7.1）。

查询不保存流程状态：流程停在“等评分结束”时，用户照样可以随时查询，不会打断
或改变流程。需要用户选择时（选任务、选论文）给出选择卡片，所需参数放在卡片里，
选完由运行器再发起一次查询。

分数、扣分点、证据由前端从接口现取渲染，这里只放引用编号与模板文案（B8）。
"""

from __future__ import annotations

from typing import TypedDict

from fastapi import HTTPException
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from backend.app.services.assistant import choices
from backend.app.services.assistant import tools
from backend.app.services.assistant.context import RunContext
from backend.app.services.assistant.views import overview_text
from backend.app.services.assistant.views import paper_label
from backend.app.services.assistant.views import papers_in_upload_order
from backend.app.services.assistant.views import quick_actions_card
from backend.app.services.assistant.views import resolve_paper
from backend.app.services.assistant.views import score_overview

ACTIVE_JOB_STATUSES = {"queued", "running", "cancel_requested"}
RETRYABLE_ITEM_STATUSES = {"failed", "canceled", "running"}

QUERY_INTENTS = ("query_progress", "query_paper", "query_review", "query_overview", "retry_failed", "cancel_job")


class QueryState(TypedDict, total=False):
    intent: str
    ordinal: int | None
    name: str | None
    text: str
    batch_id: str | None
    paper_id: str | None
    next: str


def _ctx(runtime: Runtime[RunContext]) -> RunContext:
    return runtime.context


def resolve_batch(state: QueryState, runtime: Runtime[RunContext]):
    ctx = _ctx(runtime)
    if state.get("batch_id"):
        return {"next": state["intent"]}
    focus_batch = (ctx.conversation.focus or {}).get("batch_id")
    if focus_batch:
        try:
            tools.call(ctx, "batch_summary", batch_id=focus_batch)
            return {"batch_id": focus_batch, "next": state["intent"]}
        except HTTPException:
            # 焦点里的任务已删除或无权访问：不沿用旧信息回答（方案 §9.3）。
            ctx.set_focus(batch_id=None, batch_name=None, job_id=None, paper_id=None)
            ctx.say("之前的评分任务现在不可用了（可能已删除或无权访问），换一个吧。")
    batches = tools.call(ctx, "list_recent_batches", limit=8)
    if not batches:
        ctx.say("还没有评分任务。要开始评分吗？", [quick_actions_card(["start_grading"])])
        return {"next": END}
    if len(batches) == 1:
        ctx.set_focus(batch_id=batches[0]["id"], batch_name=batches[0]["name"])
        return {"batch_id": batches[0]["id"], "next": state["intent"]}
    current = (ctx.conversation.focus or {}).get("batch_id")
    ctx.say("要查哪个评分任务？", [choices.card(
        "focus_task",
        [{"id": batch["id"], "label": batch["name"], "current": batch["id"] == current} for batch in batches],
        next_query={"intent": state["intent"], "ordinal": state.get("ordinal"), "name": state.get("name"),
                    "text": (state.get("text") or "")[:200]},
    )])
    return {"next": END}


def query_progress(state: QueryState, runtime: Runtime[RunContext]):
    ctx = _ctx(runtime)
    batch_id = state["batch_id"]
    job = tools.call(ctx, "latest_job", batch_id=batch_id)
    if job is None:
        # 没有作业时进度页只会显示原始报错（10-11 生产验收时发现），工作区不切过去。
        ctx.say("这个任务还没有开始评分。")
        return {"next": END}
    ctx.show_workspace("/tasks/%s/run" % batch_id)
    finished = job["succeeded_count"] + job["skipped_count"] + job["failed_count"] + job["canceled_count"]
    if job["status"] in ACTIVE_JOB_STATUSES:
        text = "正在评分：已完成 %d/%d 份。" % (finished, job["total_items"])
    else:
        text = "评分已结束：完成 %d 份，失败 %d 份。" % (job["succeeded_count"] + job["skipped_count"], job["failed_count"])
    ctx.say(text, [{"type": "job_progress", "status": "info", "batch_id": batch_id, "job_id": job["id"]}],
            intent="query_progress")
    return {"next": END}


def _show_paper(ctx: RunContext, batch_id: str, paper: dict, ordinal: int, profile_key: str):
    ctx.set_focus(paper_id=paper["paper_id"])
    ctx.show_workspace("/batches/%s/grade?paper=%s" % (batch_id, paper["paper_id"]))
    name = paper_label(paper, profile_key)
    if not paper.get("latest_run_id"):
        ctx.say("第 %d 份《%s》还没有评分结果。" % (ordinal, name))
        return
    score = "未出总分" if paper.get("latest_final_score") is None else "总分 %s" % paper["latest_final_score"]
    review = "，有评分项需要人工复核" if paper.get("latest_need_manual_review") else ""
    ctx.say(
        "第 %d 份《%s》：%s%s。各项得分、扣分点与证据如下，右侧可以对照原文。" % (ordinal, name, score, review),
        [{"type": "paper_result", "batch_id": batch_id, "paper_id": paper["paper_id"], "run_id": paper["latest_run_id"]}],
        intent="query_paper",
    )


def query_paper(state: QueryState, runtime: Runtime[RunContext]):
    ctx = _ctx(runtime)
    batch_id = state["batch_id"]
    summary = tools.call(ctx, "batch_summary", batch_id=batch_id)
    papers = papers_in_upload_order(summary["papers"])
    profile_key = summary.get("business_profile_key") or "thesis"
    if not papers:
        ctx.say("这个任务里还没有文件。")
        return {"next": END}
    if state.get("paper_id"):
        match = next((paper for paper in papers if paper["paper_id"] == state["paper_id"]), None)
        candidates = papers
    else:
        match, candidates = resolve_paper(papers, ordinal=state.get("ordinal"), name=state.get("name"),
                                          text=state.get("text") or "", profile_key=profile_key)
    if match is not None:
        _show_paper(ctx, batch_id, match, papers.index(match) + 1, profile_key)
        return {"next": END}
    prompt = "这个任务只有 %d 份文件，要查哪一份？" % len(papers) if state.get("ordinal") else "要查哪一份？"
    ctx.say(prompt, [choices.card(
        "focus_paper",
        [{"id": paper["paper_id"], "batch_id": batch_id,
          "label": "%d. %s" % (papers.index(paper) + 1, paper_label(paper, profile_key))}
         for paper in candidates],
        next_query={"intent": "query_paper"}, limit=choices.MAX_PAPER_OPTIONS,
    )])
    return {"next": END}


def query_review(state: QueryState, runtime: Runtime[RunContext]):
    ctx = _ctx(runtime)
    batch_id = state["batch_id"]
    summary = tools.call(ctx, "batch_summary", batch_id=batch_id)
    flagged = [paper for paper in summary["papers"] if paper.get("latest_need_manual_review")]
    ctx.show_workspace("/batches/%s/grade" % batch_id)
    if not flagged:
        ctx.say("没有需要人工复核的材料。")
        return {"next": END}
    ctx.say("有 %d 份材料需要人工复核。点开某一份可以看到具体是哪几项、为什么；复核请在右侧的评分工作区完成。" % len(flagged),
            [{"type": "review_list", "batch_id": batch_id}], intent="query_review")
    return {"next": END}


def query_overview(state: QueryState, runtime: Runtime[RunContext]):
    ctx = _ctx(runtime)
    batch_id = state["batch_id"]
    summary = tools.call(ctx, "batch_summary", batch_id=batch_id)
    ctx.show_workspace("/batches/%s/grade" % batch_id)
    ctx.say(overview_text(score_overview(summary["papers"])), [{"type": "overview", "batch_id": batch_id}],
            intent="query_overview")
    return {"next": END}


def retry_failed(state: QueryState, runtime: Runtime[RunContext]):
    """重试是写动作：这里只给确认卡片，用户确认后由运行器执行（方案 §7.6）。"""

    ctx = _ctx(runtime)
    batch_id = state["batch_id"]
    job = tools.call(ctx, "latest_job", batch_id=batch_id)
    retryable = job is not None and job["status"] in {"completed_with_errors", "failed", "canceled"} and any(
        item["status"] in RETRYABLE_ITEM_STATUSES for item in job["items"]
    )
    if not retryable:
        ctx.say("没有可以重试的失败项。")
        return {"next": END}
    count = sum(1 for item in job["items"] if item["status"] in RETRYABLE_ITEM_STATUSES)
    ctx.say("有 %d 份没有评完。重试它们吗？已成功的结果会保留。" % count, [{
        "type": "confirm_retry", "status": "proposed", "batch_id": batch_id, "job_id": job["id"], "count": count,
    }])
    return {"next": END}


def cancel_job(state: QueryState, runtime: Runtime[RunContext]):
    ctx = _ctx(runtime)
    batch_id = state["batch_id"]
    job = tools.call(ctx, "latest_job", batch_id=batch_id)
    if job is None or job["status"] not in ACTIVE_JOB_STATUSES:
        ctx.say("当前没有进行中的评分。")
        return {"next": END}
    finished = job["succeeded_count"] + job["skipped_count"] + job["failed_count"] + job["canceled_count"]
    ctx.say("正在评分（%d/%d）。取消剩余的吗？已完成的结果会保留。" % (finished, job["total_items"]), [{
        "type": "confirm_cancel", "status": "proposed", "batch_id": batch_id, "job_id": job["id"],
    }])
    return {"next": END}


# --- 列清单（docs/评分助手意图识别模型优先方案.md 3.2）：不需要先选任务，不进查询图 --------------------

LIST_LIMIT = 8
RUBRICS_LINK = {"type": "link", "label": "打开评分标准页", "path": "/rubrics"}
TASKS_LINK = {"type": "link", "label": "打开评分任务页", "path": "/tasks"}


def _unpublished_rubric_count(ctx: RunContext) -> int:
    from backend.app.api import actions

    return sum(1 for rubric in actions.list_visible_rubrics(ctx.db, ctx.principal)
               if rubric.status not in {"published", "archived"})


def _numbered(lines: list[str]) -> str:
    return "\n".join("%d. %s" % (index, line) for index, line in enumerate(lines, start=1))


def list_rubrics(ctx: RunContext) -> None:
    """我能用的评分标准：已发布的逐条列出，两份以上给选择卡片（点一份就用它开始评分）。"""

    from backend.app.services.assistant.views import document_type_label

    rubrics = tools.call(ctx, "list_published_rubrics")
    unpublished = _unpublished_rubric_count(ctx)
    drafts = "；另有 %d 份还没发布，在评分标准页发布后才能用来评分" % unpublished if unpublished else ""
    if not rubrics:
        ctx.say("还没有已发布的评分标准%s。可以上传你的评分规则和模板，我来帮你导入。" % drafts,
                [quick_actions_card(["import_rubric"]), RUBRICS_LINK], intent="list_rubrics")
        return
    lines = ["%s（%s · 第 %s 版）" % (item["name"], document_type_label(item["business_profile_key"]), item["version"])
             for item in rubrics[:LIST_LIMIT]]
    more = "，下面是最新发布的 %d 份" % LIST_LIMIT if len(rubrics) > LIST_LIMIT else ""
    text = "你能用的评分标准有 %d 份%s%s：\n%s" % (len(rubrics), more, drafts, _numbered(lines))
    if len(rubrics) >= choices.MIN_OPTIONS and not choices.unavailable(ctx, "use_rubric"):
        current = choices.current_id(ctx, "use_rubric")
        ctx.say(text + "\n点一份就用它开始评分。", [choices.card(
            "use_rubric",
            [{"id": item["id"], "label": item["name"], "current": item["id"] == current} for item in rubrics],
        ), RUBRICS_LINK], intent="list_rubrics")
        return
    ctx.say(text, [quick_actions_card(["start_grading"]), RUBRICS_LINK], intent="list_rubrics")


def list_tasks(ctx: RunContext) -> None:
    """我最近的评分任务：逐条列出，两个以上给选择卡片（选中后可以接着问进度、扣分）。"""

    batches = tools.call(ctx, "list_recent_batches", limit=LIST_LIMIT)
    if not batches:
        ctx.say("还没有评分任务。要开始评分吗？", [quick_actions_card(["start_grading"])], intent="list_tasks")
        return
    text = "你最近的评分任务（最新的在前）：\n%s" % _numbered([batch["name"] for batch in batches])
    if len(batches) >= choices.MIN_OPTIONS:
        current = (ctx.conversation.focus or {}).get("batch_id")
        ctx.say(text + "\n选一个，就可以接着问它的进度和扣分。", [choices.card(
            "focus_task",
            [{"id": batch["id"], "label": batch["name"], "current": batch["id"] == current} for batch in batches],
        ), TASKS_LINK], intent="list_tasks")
        return
    ctx.set_focus(batch_id=batches[0]["id"], batch_name=batches[0]["name"])
    ctx.say(text + "\n可以接着问它的进度和扣分。", [quick_actions_card(["query_progress", "query_overview"]), TASKS_LINK],
            intent="list_tasks")


def build_query_graph():
    builder = StateGraph(QueryState, context_schema=RunContext)
    builder.add_node("resolve_batch", resolve_batch)
    for name, fn in (
        ("query_progress", query_progress), ("query_paper", query_paper), ("query_review", query_review),
        ("query_overview", query_overview), ("retry_failed", retry_failed), ("cancel_job", cancel_job),
    ):
        builder.add_node(name, fn)
        builder.add_edge(name, END)
    builder.add_edge(START, "resolve_batch")
    builder.add_conditional_edges("resolve_batch", lambda state: state["next"], [*QUERY_INTENTS, END])
    return builder.compile()

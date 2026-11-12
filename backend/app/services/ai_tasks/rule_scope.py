"""范围识别（``rule_scope``）：给不经过起草的规则补上适用范围。

每个评分项一个条目（规则多时再拆），只调用一次模型；输出没有 rules 数组时带修正提示
再执行一次，仍不合格就把这一项记为识别失败继续，其余评分项照常。全部条目结束后原地
写进当前草稿的规则（docs/评分细则适用范围方案.md 第 6 节）。
"""

from __future__ import annotations

from backend.app.db import models
from backend.app.services.ai_tasks.errors import AITaskItemError
from backend.app.services.ai_tasks.errors import AITaskProblem
from backend.app.services.ai_tasks.handlers import PreparedTask
from backend.app.services.ai_tasks.handlers import TaskHandler
from backend.app.services.ai_tasks.handlers import register_handler
from backend.app.services.ai_tasks.provider_errors import provider_item_error
from backend.app.services.llm.errors import ProviderJSONOutputError
from backend.app.services.rubric_scope import VOCABULARY_VERSION
from backend.app.services.rubric_scope.identification import SCOPE_PROMPT_VERSION
from backend.app.services.rubric_scope.identification import ScopeIdentificationError
from backend.app.services.rubric_scope.identification import apply_identified_scopes
from backend.app.services.rubric_scope.identification import build_items
from backend.app.services.rubric_scope.identification import identify_once
from backend.app.services.rubric_scope.identification import pending_rules
from backend.app.services.rubrics.draft_graph import read_execution_draft


KIND = "rule_scope"
_PROVIDER_HINTS = {
    "authentication_failed": "AI 连接鉴权失败，请检查 API Key 与地域是否匹配。",
    "permission_denied": "AI 连接没有调用权限，请检查模型授权。",
    "quota_exhausted": "AI 连接的额度已用完（余额不足或配额耗尽），请充值或更换连接后重试。",
    "request_timeout": "AI 请求超时，请稍后重试失败的评分项。",
    "model_or_endpoint_not_found": "AI 模型或接口地址不存在，请检查连接配置。",
    "invalid_request": "AI 接口拒绝了请求参数，请检查模型与协议兼容性。",
    "rate_limited": "AI 调用受到限流，稍后自动继续。",
}


def _prepare(session, rubric_id, _params, _principal):
    rubric = session.get(models.Rubric, rubric_id)
    if rubric is None or rubric.status != "draft":
        raise AITaskProblem(
            409, "RUBRIC_NOT_DRAFT", "评分标准已提交审核或已发布，不能识别适用范围。", "请先新建草稿后再识别。"
        )
    version = (read_execution_draft(session=session, rubric_id=rubric_id).get("active_compilation") or {}).get(
        "version"
    ) or {}
    if not version.get("id"):
        raise AITaskProblem(409, "RUBRIC_DRAFT_MISSING", "没有可识别的执行草稿。", "请刷新页面后重试。")
    profile_key = version.get("business_profile_key") or "thesis"
    rules = pending_rules(session, version["id"])
    criteria = {
        criterion.id: criterion
        for criterion in session.query(models.RubricCriterion).filter(
            models.RubricCriterion.rubric_id == rubric_id
        )
    }
    snapshot = {
        "version_id": version["id"],
        "business_profile_key": profile_key,
        "vocabulary": VOCABULARY_VERSION,
        "rule_codes": [rule.rule_code for rule in rules],
    }
    scope = {"version_id": version["id"]}
    if not rules:
        return PreparedTask(
            scope=scope, input_snapshot=snapshot, items=[],
            result={"status": "nothing_to_identify", "identified": 0, "scopes": {}, "failed": [], "discarded": 0,
                    "missing": []},
        )
    return PreparedTask(scope=scope, input_snapshot=snapshot, items=build_items(criteria, rules))


def _output_error(exc: ProviderJSONOutputError) -> AITaskItemError | None:
    """模型答了但不能用：截断与拒答不会因为再问一次而变好；其它按“缺 rules”处理。"""

    if exc.reason == "output_truncated":
        return AITaskItemError(
            "SCOPE_OUTPUT_TRUNCATED",
            "模型输出达到长度上限，范围识别不完整；请提高连接的输出 Token 上限后重试失败的评分项。",
        )
    if exc.reason == "refused":
        return AITaskItemError("SCOPE_REFUSED", "模型拒绝识别这项内容（安全策略），可调整后重试。")
    if exc.reason == "error_envelope":
        return AITaskItemError(
            "AI_PROVIDER_ERROR", "AI 接口在成功状态中返回错误，请测试当前连接后重试。", disposition="retry"
        )
    return None


def _run_item(view, item_input, scorer):
    repair = bool(item_input.get("repair_code"))
    profile_key = view.input_snapshot.get("business_profile_key") or "thesis"
    code = item_input["criterion"]["code"]
    try:
        scopes, discarded = identify_once(scorer, item_input, profile_key, repair=repair)
    except ProviderJSONOutputError as exc:
        error = _output_error(exc)
        if error is not None:
            raise error from exc
        scopes, discarded = None, 0
    except Exception as exc:
        raise provider_item_error(
            exc, code="AI_PROVIDER_ERROR", message="AI 服务暂时不可用，请稍后重试失败的评分项。",
            hints=_PROVIDER_HINTS,
        ) from exc
    # 没有 rules 数组，或给出的项全部不合格（编号不认识、范围不合格），都按输出不合格处理。
    if scopes is None or (not scopes and discarded):
        if not repair:
            raise AITaskItemError(
                "SCOPE_OUTPUT_INVALID", "模型输出缺少 rules 数组，或给出的范围全部不合格。", disposition="repair"
            )
        # 修正后仍不合格：这一项的规则保持没有范围（按全文），其余评分项照常。
        return {"code": code, "scopes": {}, "discarded": discarded, "failed": True}
    return {"code": code, "scopes": scopes, "discarded": discarded, "failed": False}


def _merge(view, outputs):
    scopes = {}
    for output in outputs:
        scopes.update(output.get("scopes") or {})
    return {
        "status": "identified",
        "prompt_version": SCOPE_PROMPT_VERSION,
        # 写入时只补仍然没有范围的规则；期间重新起草写上范围的不覆盖，实际写入可能更少。
        "identified": len(scopes),
        "vocabulary": view.input_snapshot.get("vocabulary"),
        "scopes": scopes,
        "failed": sorted({output["code"] for output in outputs if output.get("failed")}),
        "discarded": sum(int(output.get("discarded") or 0) for output in outputs),
        "missing": sorted(set(view.input_snapshot.get("rule_codes") or []) - set(scopes)),
    }


def _on_task_success(session, view, result):
    snapshot = view.input_snapshot
    try:
        apply_identified_scopes(
            session,
            rubric_id=view.rubric_id,
            version_id=snapshot["version_id"],
            profile_key=snapshot.get("business_profile_key") or "thesis",
            scopes=result.get("scopes") or {},
            actor_id=view.owner_id,
        )
    except ScopeIdentificationError as exc:
        raise AITaskItemError(exc.code, exc.message) from exc


RULE_SCOPE_HANDLER = register_handler(
    TaskHandler(
        kind=KIND,
        prompt_version=SCOPE_PROMPT_VERSION,
        prepare=_prepare,
        run_item=_run_item,
        merge=_merge,
        on_task_success=_on_task_success,
        label="适用范围识别",
        extra={
            "requires_real_model": True,
            "missing_model_problem": (
                "AI_CONNECTION_MISSING",
                "当前没有可用于识别适用范围的真实 AI 连接。",
                "请配置真实 AI 连接后重试。",
            ),
        },
    )
)


__all__ = ["KIND", "RULE_SCOPE_HANDLER"]

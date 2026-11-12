"""Transport-neutral projection of task failures onto stable Core issues."""

from __future__ import annotations

import re


_SAFE_EXCEPTION_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,79}$")
_PROJECT_MARKER = "backend/app/"

#: 全部校验标识。与适配器的修正提示清单（``llm/core_adapter.OUTPUT_REPAIR_HINTS``）一一对应，
#: 由测试保证；内核不导入适配器，所以在这里再列一遍。异常带来的 ``check`` 不在其中时一律
#: 当作 ``fields``：既不会把任意文字写进日志，也不会拿一个没有提示的标识去重问。
OUTPUT_REPAIR_CHECKS = frozenset({
    "json",
    "fields",
    "status",
    "non_triggered_has_decision",
    "triggered_without_evidence",
    "evidence",
    "level",
    "finding_code",
    "rule_code",
})


def safe_output_check(value) -> str:
    return value if isinstance(value, str) and value in OUTPUT_REPAIR_CHECKS else "fields"


#: 材料级统一报告的「模型回答不合格」根因码。
OUTPUT_INVALID_CODE = "PROVIDER_OUTPUT_INVALID"
#: 执行器校验返回的、同样表示「模型答了但回答不合格」的错误码 -> 校验标识。错误码本身
#: 不改（已保存的结果、人工复核任务与回放校验都在用）；校验标识决定重问时用哪条固定提示。
OUTPUT_INVALID_CHECKS = {
    "RULE_DECISION_INVALID": "fields",
    "REQUIRED_EVIDENCE_INVALID": "evidence",
    "BAND_SELECTION_INVALID": "level",
    "FINDING_CODE_NOT_AUTHORIZED": "finding_code",
    "UNKNOWN_RULE": "rule_code",
}


def failure_site(exc: BaseException) -> str:
    """异常出在本项目的哪一行：``模块路径:行号``，取调用栈里最靠内的本项目代码帧。

    只有位置，没有异常消息和变量值，所以不会带出材料内容；找不到时返回 ``-``。
    """

    site = "-"
    traceback = getattr(exc, "__traceback__", None)
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        index = filename.rfind(_PROJECT_MARKER)
        if index >= 0:
            module = filename[index + len(_PROJECT_MARKER):]
            if module.endswith(".py"):
                module = module[:-3]
            site = "%s:%d" % (module, traceback.tb_lineno)
        traceback = traceback.tb_next
    return site


def project_rule_execution_failure(exc: Exception) -> tuple[str, str]:
    """Use duck-typed safe error contracts without importing adapters."""

    direct_code = getattr(exc, "code", None)
    if isinstance(direct_code, str) and direct_code.strip():
        normalized = direct_code.strip().upper()
        if normalized == "TOKEN_BUDGET_EXCEEDED":
            return normalized, "paper input-token budget exhausted; request not sent"
        if normalized == "PROVIDER_THINKING_NOT_DISABLED":
            # 运行身份说不思考，模型却在思考：判定不采用；之前已发现时这次根本没发。
            return normalized, (
                "provider reasoned although the run identity disables thinking; "
                + ("decision discarded" if getattr(exc, "sent", False) else "request not sent")
            )
        if normalized == "PROVIDER_CIRCUIT_OPEN":
            # Not an input problem: the request was never sent because earlier
            # provider failures opened the connection's circuit breaker.
            return normalized, "semantic provider request deferred by open circuit"
        return normalized, "rule input could not be prepared safely"
    provider_error = getattr(exc, "error", None)
    provider_code = getattr(provider_error, "code", None)
    if isinstance(provider_code, str) and provider_code.strip():
        normalized = provider_code.strip().upper()
        if not normalized.startswith("PROVIDER_"):
            normalized = "PROVIDER_" + normalized
        return normalized, "semantic provider request failed"
    # Every adapter's JSON-output error carries a safe ``reason``.  Truncation
    # is actionable (raise max output tokens / disable thinking), unlike the
    # generic ValueError it would otherwise be reported as.
    if getattr(exc, "reason", None) == "output_truncated":
        return (
            "PROVIDER_OUTPUT_TRUNCATED",
            "semantic provider output hit the max output token limit",
        )
    # 模型答了但没通过回答校验（或不是可用的 JSON）：是模型输出问题，不是系统错误。
    reason = getattr(exc, "reason", None)
    if reason in ("output_invalid", "invalid_json", "empty_content"):
        check = safe_output_check(getattr(exc, "check", None)) if reason == "output_invalid" else "json"
        return (
            OUTPUT_INVALID_CODE,
            "semantic provider output failed validation (check=%s)" % check,
        )
    # Claude 的安全策略拒答（HTTP 200，stop_reason=refusal）：不是输入问题，也不是截断。
    if getattr(exc, "reason", None) == "refused":
        return (
            "PROVIDER_OUTPUT_REFUSED",
            "semantic provider declined to answer this rule",
        )
    exception_type = type(exc).__name__
    if not _SAFE_EXCEPTION_TYPE.fullmatch(exception_type):
        exception_type = "Exception"
    return (
        "RULE_EXECUTION_FAILED",
        f"rule execution failed (exception_type={exception_type}, where={failure_site(exc)})",
    )


__all__ = [
    "OUTPUT_INVALID_CHECKS",
    "OUTPUT_REPAIR_CHECKS",
    "safe_output_check",
    "OUTPUT_INVALID_CODE",
    "failure_site",
    "project_rule_execution_failure",
]

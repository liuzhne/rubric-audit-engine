"""每次模型调用一行诊断日志：只记受控字段，不记请求或响应正文。

`debug_logging` 记录原文，受保护部署禁用；这里的字段在生产也可以常开：
- 配置的模型、实际路由到的模型（`openrouter/free` 这类路由每次可能不同）；
- 上游服务商（聚合平台返回的 ``provider``）；
- HTTP 状态、错误类别和厂商错误码（如智谱的 1302/1305），用来区分并发超额、
  频率超额和服务繁忙；
- 应答的结束原因与 token 计数（输入、可见输出、推理）：运行身份说「思考关闭」，
  模型到底有没有思考、是不是撞上输出上限被截断，看这一行就知道。

唯一的例外是请求被拒（400/413/422 这类请求范围的错误）：错误码往往只有笼统的
INVALID_ARGUMENT，真正原因只在厂商的错误说明里，于是把它脱敏、截断后附在行尾。
限流、鉴权、5xx 等其他失败仍然不记。
"""

from __future__ import annotations

import json
import logging
import re

from backend.app.services.llm.errors import _default_message


logger = logging.getLogger("paper_grading.llm.calls")
_UNSAFE = re.compile(r"[^\w.:/@+-]")


def _field(value, limit: int = 80) -> str:
    """标识类字段只保留常见字符并截断，防止日志注入或夹带正文。"""

    if value is None or value == "":
        return "-"
    return _UNSAFE.sub("_", str(value))[:limit] or "-"


def _envelope_error(data):
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return None, None
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    return error.get("code"), metadata.get("provider_name")


def _token(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _optional_token(value) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else "-"


def log_call_succeeded(provider, model, data, *, elapsed_ms, attempt, usage=None, context=None):
    """HTTP 2xx。响应体里带 ``error`` 的（“成功状态中返回错误”）单独记成失败。

    ``context`` 是这次调用的思考设置（``thinking_capabilities.call_context``）：思考开关、强度、
    计划来源、命中的能力表记录、正式输出预算与实发上限。正式输出用量算不出来时记 ``-``。
    """

    routed = data.get("model") if isinstance(data, dict) else None
    upstream = data.get("provider") if isinstance(data, dict) else None
    if isinstance(data, dict) and data.get("error"):
        code, error_upstream = _envelope_error(data)
        logger.warning(
            "llm_call_error_envelope provider=%s model=%s routed_model=%s upstream=%s "
            "provider_code=%s elapsed_ms=%d attempt=%d",
            _field(provider), _field(model), _field(routed), _field(error_upstream or upstream),
            _field(code), elapsed_ms, attempt + 1,
        )
        return
    usage = usage if isinstance(usage, dict) else {}
    context = context if isinstance(context, dict) else {}
    logger.info(
        "llm_call provider=%s model=%s routed_model=%s upstream=%s elapsed_ms=%d attempt=%d "
        "finish_reason=%s prompt_tokens=%d completion_tokens=%d reasoning_tokens=%d reasoning=%s "
        "answer_tokens=%s answer=%s thinking=%s effort=%s thinking_source=%s entry=%s "
        "answer_budget=%s max_tokens=%s cached_tokens=%d",
        _field(provider), _field(model), _field(routed), _field(upstream), elapsed_ms, attempt + 1,
        _field(usage.get("finish_reason")), _token(usage.get("prompt_tokens")),
        _token(usage.get("completion_tokens")), _token(usage.get("reasoning_tokens")),
        _field(usage.get("reasoning_source") or "none"),
        _optional_token(usage.get("answer_tokens")), _field(usage.get("answer_source")),
        _field(context.get("thinking")), _field(context.get("effort")), _field(context.get("source")),
        _field(context.get("entry")), _optional_token(context.get("answer_budget")),
        _optional_token(context.get("max_tokens")), _token(usage.get("cached_tokens")),
    )


def log_thinking_mismatch(provider, model, *, usage, context):
    """能力表说这个模型能关思考（并且已经发了关闭写法），应答里却观测到了思考。

    AI 任务不经一致性守卫，这一行是发现能力表误标的主要途径（docs/模型思考能力表方案.md 第 5.4 节）。
    """

    logger.warning(
        "thinking_capability_mismatch provider=%s model=%s entry=%s thinking_source=%s "
        "reasoning_tokens=%d reasoning=%s",
        _field(provider), _field(model), _field(context.get("entry")), _field(context.get("source")),
        _token(usage.get("reasoning_tokens")), _field(usage.get("reasoning_source")),
    )


def log_output_limit_fallback(provider, model, *, field, sent, fallback, context):
    """厂商拒绝了按项目封顶放开的输出上限（模型最大输出更小），改用放开前的上限重发。

    出现这一行说明能力表缺这个模型的记录（或记录没写最大输出）：补上记录后不再多花一次请求。
    """

    context = context if isinstance(context, dict) else {}
    logger.warning(
        "thinking_output_limit_fallback provider=%s model=%s field=%s sent=%s fallback=%s "
        "thinking_source=%s entry=%s",
        _field(provider), _field(model), _field(field), _field(sent), _field(fallback),
        _field(context.get("source")), _field(context.get("entry")),
    )


_PROVIDER_MESSAGE_LIMIT = 300


def _rejection_message(projected) -> str | None:
    """请求被拒时厂商给的说明（已去换行、遮蔽密钥）；其他失败或没有说明时为 None。"""

    if projected.scope != "request" or projected.http_status is None:
        return None
    message = projected.message
    if not message or message == _default_message(projected.code):
        return None
    return message[:_PROVIDER_MESSAGE_LIMIT]


def log_call_failed(provider, model, projected, *, attempt, attempts, will_retry):
    message = _rejection_message(projected)
    logger.warning(
        "llm_call_failed provider=%s model=%s code=%s status=%s provider_code=%s provider_type=%s "
        "attempt=%d/%d retry=%s%s",
        _field(provider), _field(model), _field(projected.code), _field(projected.http_status),
        _field(projected.provider_error_code), _field(projected.provider_error_type),
        attempt + 1, attempts, "yes" if will_retry else "no",
        # JSON 字符串转义引号与控制字符，说明里的内容伪造不出新的日志字段。
        "" if message is None else " provider_message=" + json.dumps(message, ensure_ascii=False),
    )

"""管理员登记的模型思考能力：进程内缓存（docs/模型管理页与评分思考修正方案.md 3.2）。

``thinking_capabilities.lookup`` 每次都读 ``current()``，所以这里只能是内存里的元组，不能查库。
刷新发生在手里有数据库会话的入口：解析连接运行时（``ai_connections.resolve_connection_runtime``）与
平台模型（``llm.factory._platform_runtime``）。两处都在建适配器之前，所以新任务最迟 ``REFRESH_SECONDS``
秒后用上新记录；已经建好的适配器在构造时算好了计划，不受影响。写入接口提交后调用 ``invalidate()``，
同一进程的下一次解析立刻重读。读库失败时保留上一次的结果（读不出来不该让评分换一套思考写法）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 30

_lock = threading.Lock()
_state = {"loaded_at": None, "version": None, "capabilities": ()}


def content_hash(row) -> str:
    """一条记录里影响请求与匹配的内容的哈希（不含备注、审计字段）。"""

    payload = {
        "protocol": row.protocol, "host_family": row.host_family, "model_prefix": row.model_prefix,
        "can_disable": bool(row.can_disable), "off_request": row.off_request,
        "lowest_effort": row.lowest_effort, "effort_request": row.effort_request,
        "max_output_tokens": row.max_output_tokens,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_capability(row):
    from backend.app.services.llm.thinking_capabilities import ThinkingCapability

    return ThinkingCapability(
        protocols=(row.protocol,),
        families=(row.host_family,),
        model_prefix=row.model_prefix,
        can_disable=bool(row.can_disable),
        off_request=row.off_request,
        lowest_effort=row.lowest_effort or None,
        effort_request=row.effort_request,
        max_output_tokens=row.max_output_tokens,
        sources=(row.source_url,),
        verified_on=row.verified_on,
        note=row.note or "",
        override_hash=content_hash(row),
    )


def current() -> tuple:
    return _state["capabilities"]


def invalidate() -> None:
    with _lock:
        _state["loaded_at"] = None


def reset() -> None:
    """测试用：清空缓存（每个测试的数据库都是新的）。"""

    with _lock:
        _state.update(loaded_at=None, version=None, capabilities=())


def _version(session):
    from sqlalchemy import func
    from sqlalchemy import select

    from backend.app.db.models import ModelCapabilityOverride

    count, latest, revisions = session.execute(
        select(func.count(ModelCapabilityOverride.id), func.max(ModelCapabilityOverride.updated_at),
               func.coalesce(func.sum(ModelCapabilityOverride.revision), 0))
    ).one()
    return (int(count or 0), latest.isoformat() if latest else None, int(revisions or 0))


def refresh_if_stale(session, *, now: float | None = None) -> None:
    """距上次读取超过 ``REFRESH_SECONDS`` 秒（或被 ``invalidate``）时，按版本号决定是否重读。"""

    if session is None:
        return
    now = time.monotonic() if now is None else now
    loaded_at = _state["loaded_at"]
    if loaded_at is not None and now - loaded_at < REFRESH_SECONDS:
        return
    try:
        version = _version(session)
        if version != _state["version"] or loaded_at is None:
            from sqlalchemy import select

            from backend.app.db.models import ModelCapabilityOverride

            rows = session.scalars(
                select(ModelCapabilityOverride).where(ModelCapabilityOverride.enabled.is_(True))
            ).all()
            capabilities = tuple(to_capability(row) for row in rows)
        else:
            capabilities = _state["capabilities"]
    except Exception as exc:  # noqa: BLE001 - 迁移未跑或库暂时不可达：沿用上一次的结果
        logger.warning("capability_overrides_refresh_failed error=%s", type(exc).__name__)
        with _lock:
            _state["loaded_at"] = now
        return
    with _lock:
        _state.update(loaded_at=now, version=version, capabilities=capabilities)


__all__ = ["REFRESH_SECONDS", "content_hash", "current", "invalidate", "refresh_if_stale", "reset", "to_capability"]

"""模型管理：平台管理员登记模型的思考能力（docs/模型管理页与评分思考修正方案.md 3.2）。

全部接口要求平台管理员。列表合并代码能力表与管理员记录；写入后让本进程的能力缓存失效，
其它进程（worker）最迟 30 秒后读到。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.api.deps import CurrentPrincipal
from backend.app.api.deps import current_principal
from backend.app.api.deps import require_platform_admin
from backend.app.db.models import AIConnection
from backend.app.db.models import ModelCapabilityOverride
from backend.app.db.models import utcnow
from backend.app.db.session import get_db
from backend.app.schemas.model_capability import HOST_FAMILIES
from backend.app.schemas.model_capability import ModelCapabilityCreate
from backend.app.schemas.model_capability import ModelCapabilityEntry
from backend.app.schemas.model_capability import ModelCapabilityList
from backend.app.schemas.model_capability import ModelCapabilityUpdate
from backend.app.services.assistant.agent import evaluated as agent_evaluated
from backend.app.services.auth import audit
from backend.app.services.auth import auth_active
from backend.app.services.llm import capability_overrides
from backend.app.services.llm import hosts
from backend.app.services.llm import thinking_capabilities as thinking

router = APIRouter(prefix="/admin/model-capabilities", tags=["model-capabilities"])

#: 连接协议 → 能力表协议。
_PROTOCOL_OF = {"openai_compatible": thinking.CHAT, "openai_responses": thinking.RESPONSES,
                "anthropic_messages": thinking.MESSAGES}


def _active_connections(db: Session) -> list[tuple[str, str, str]]:
    rows = db.execute(
        select(AIConnection.provider_type, AIConnection.base_url, AIConnection.model_name)
        .where(AIConnection.status == "active")
    ).all()
    return [(_PROTOCOL_OF.get(provider), hosts.host_family(base_url), thinking.normalize_model_name(model))
            for provider, base_url, model in rows if provider in _PROTOCOL_OF]


def _hits(connections, protocols, families, prefix) -> int:
    return sum(1 for protocol, family, name in connections
               if protocol in protocols and family in families and thinking.prefix_matches(name, prefix))


def _code_entry(capability, *, overridden: bool, connections) -> ModelCapabilityEntry:
    return ModelCapabilityEntry(
        source="code", key=capability.key, protocol="+".join(capability.protocols),
        host_family="+".join(capability.families), model_prefix=capability.model_prefix,
        can_disable=capability.can_disable,
        off_request=dict(capability.off_request) if capability.off_request is not None else None,
        lowest_effort=capability.lowest_effort,
        effort_request=dict(capability.effort_request) if capability.effort_request is not None else None,
        max_output_tokens=capability.max_output_tokens, note=capability.note,
        sources=list(capability.sources), verified_on=capability.verified_on, overridden=overridden,
        active_connections=_hits(connections, capability.protocols, capability.families, capability.model_prefix),
    )


def _admin_entry(row: ModelCapabilityOverride, connections) -> ModelCapabilityEntry:
    return ModelCapabilityEntry(
        source="admin", id=row.id, key="%s%s" % (thinking.OVERRIDE_KEY_PREFIX, capability_overrides.content_hash(row)[:8]),
        protocol=row.protocol, host_family=row.host_family, model_prefix=row.model_prefix,
        can_disable=row.can_disable, off_request=row.off_request, lowest_effort=row.lowest_effort,
        effort_request=row.effort_request, max_output_tokens=row.max_output_tokens, enabled=row.enabled,
        note=row.note or "", sources=[row.source_url], verified_on=row.verified_on,
        active_connections=_hits(connections, (row.protocol,), (row.host_family,), row.model_prefix)
        if row.enabled else 0,
        revision=row.revision, updated_at=row.updated_at,
    )


@router.get("", response_model=ModelCapabilityList)
def list_model_capabilities(
    principal: CurrentPrincipal = Depends(current_principal),
    db: Session = Depends(get_db),
):
    require_platform_admin(principal)
    rows = db.scalars(select(ModelCapabilityOverride).order_by(ModelCapabilityOverride.updated_at.desc())).all()
    connections = _active_connections(db)
    overridden = {(row.protocol, row.host_family, row.model_prefix) for row in rows if row.enabled}
    items = [_admin_entry(row, connections) for row in rows]
    for capability in thinking.CAPABILITIES:
        covered = any((protocol, family, capability.model_prefix) in overridden
                      for protocol in capability.protocols for family in capability.families)
        items.append(_code_entry(capability, overridden=covered, connections=connections))
    return ModelCapabilityList(
        items=items,
        host_families=list(HOST_FAMILIES),
        agent_evaluated=sorted("%s · %s" % pair for pair in agent_evaluated.EVALUATED_MODELS),
    )


def _actor(principal: CurrentPrincipal) -> str | None:
    # 开发模式的占位用户不在 users 表里：只在真实会话下记操作人。
    return principal.user_id if auth_active() else None


def _metadata(row: ModelCapabilityOverride) -> dict:
    return {"id": row.id, "protocol": row.protocol, "host_family": row.host_family,
            "model_prefix": row.model_prefix, "can_disable": row.can_disable, "enabled": row.enabled,
            "content_hash": capability_overrides.content_hash(row)[:16], "revision": row.revision}


@router.post("", response_model=ModelCapabilityEntry, status_code=201)
def create_model_capability(
    payload: ModelCapabilityCreate,
    principal: CurrentPrincipal = Depends(current_principal),
    db: Session = Depends(get_db),
):
    require_platform_admin(principal)
    now = utcnow()
    row = ModelCapabilityOverride(
        protocol=payload.protocol, host_family=payload.host_family, model_prefix=payload.model_prefix,
        can_disable=payload.can_disable, off_request=payload.off_request, lowest_effort=payload.lowest_effort,
        effort_request=payload.effort_request, max_output_tokens=payload.max_output_tokens,
        enabled=payload.enabled, note=payload.note, source_url=payload.source_url,
        verified_on=payload.verified_on.isoformat(), revision=1,
        created_by=_actor(principal), updated_by=_actor(principal), created_at=now, updated_at=now,
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="同一协议、主机族与前缀已经有一条记录，请直接编辑它") from exc
    audit(db, "model_capability.created", actor_id=_actor(principal), metadata=_metadata(row))
    db.commit()
    db.refresh(row)
    capability_overrides.invalidate()
    return _admin_entry(row, _active_connections(db))


@router.patch("/{capability_id}", response_model=ModelCapabilityEntry)
def update_model_capability(
    capability_id: str,
    payload: ModelCapabilityUpdate,
    principal: CurrentPrincipal = Depends(current_principal),
    db: Session = Depends(get_db),
):
    require_platform_admin(principal)
    row = db.get(ModelCapabilityOverride, capability_id)
    if row is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    if row.revision != payload.revision:
        raise HTTPException(status_code=409, detail="这条记录刚被别人改过，请刷新后再改")
    for name in ("can_disable", "off_request", "lowest_effort", "effort_request", "max_output_tokens",
                 "enabled", "note", "source_url"):
        setattr(row, name, getattr(payload, name))
    row.verified_on = payload.verified_on.isoformat()
    row.revision += 1
    row.updated_by = _actor(principal)
    row.updated_at = utcnow()
    audit(db, "model_capability.updated", actor_id=_actor(principal), metadata=_metadata(row))
    db.commit()
    db.refresh(row)
    capability_overrides.invalidate()
    return _admin_entry(row, _active_connections(db))

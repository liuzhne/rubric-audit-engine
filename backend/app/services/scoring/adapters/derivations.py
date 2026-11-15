"""论文派生数据的存取：分节摘要与定位结果（docs/分节摘要定位方案.md 第 7 节）。

先查后生成；并发时以先写入的为准（唯一约束冲突后读回）。记录写入后不修改。
"""

from __future__ import annotations

from copy import deepcopy

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.db.models import DocumentDerivation
from backend.app.services.scoring.core.canonical import canonical_sha256
from backend.app.services.scoring.retrieval import section_summary


def _find(db, *, document_snapshot_id, kind, key_hash):
    return db.execute(
        select(DocumentDerivation).where(
            DocumentDerivation.document_snapshot_id == document_snapshot_id,
            DocumentDerivation.kind == kind,
            DocumentDerivation.key_hash == key_hash,
        )
    ).scalar_one_or_none()


def get_derivation(db, *, document_snapshot_id, kind, key_hash):
    """``{payload, payload_hash}``；没有返回 None。"""

    row = _find(db, document_snapshot_id=document_snapshot_id, kind=kind, key_hash=key_hash)
    if row is None:
        return None
    return {"payload": deepcopy(row.payload), "payload_hash": row.payload_hash}


def put_derivation(db, *, document_snapshot_id, kind, key_hash, payload, model_provider=None, model_name=None):
    """写入并返回 ``{payload, payload_hash}``；别的事务先写了同一条时返回它写的那一条。"""

    payload_hash = canonical_sha256(payload)
    try:
        with db.begin_nested():
            db.add(DocumentDerivation(
                document_snapshot_id=document_snapshot_id, kind=kind, key_hash=key_hash,
                payload=deepcopy(payload), payload_hash=payload_hash,
                model_provider=model_provider, model_name=model_name,
            ))
            db.flush()
    except IntegrityError:
        existing = get_derivation(db, document_snapshot_id=document_snapshot_id, kind=kind, key_hash=key_hash)
        if existing is None:
            raise
        return existing
    return {"payload": deepcopy(payload), "payload_hash": payload_hash}


def ensure_section_summaries(db, *, document_snapshot, generate, model_provider, model_name):
    """这份快照在这个模型下的分节摘要：有就复用，没有就调 ``generate(units)`` 生成并存下。

    ``generate`` 返回摘要记录（``section_summary.summarize_sections`` 的结果），或 None 表示
    当前评分器不能生成（Mock、没有 ``complete_json``）——这时不写入，返回 None。
    一个节点都没写出摘要的记录也不存：下次评分再试，而不是把一次失败冻结下来。
    """

    key_hash = section_summary.summary_key_hash(
        snapshot_hash=document_snapshot.snapshot_hash,
        model_provider=model_provider, model_name=model_name,
    )
    found = get_derivation(
        db, document_snapshot_id=document_snapshot.id, kind=section_summary.KIND, key_hash=key_hash
    )
    if found is not None:
        return found
    record = generate(list((document_snapshot.snapshot_payload or {}).get("evidence_units") or ()))
    if record is None or not any(node.get("summary") for node in record.get("nodes") or ()):
        return None
    return put_derivation(
        db, document_snapshot_id=document_snapshot.id, kind=section_summary.KIND, key_hash=key_hash,
        payload=record, model_provider=model_provider, model_name=model_name,
    )


__all__ = ["ensure_section_summaries", "get_derivation", "put_derivation"]

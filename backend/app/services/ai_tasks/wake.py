"""建任务之后叫醒执行者（路由与导入会话共用）。"""

from __future__ import annotations

import logging

from sqlalchemy.orm import sessionmaker

from backend.app.services.work_queue.sweep import ensure_sweep_chain
from backend.app.services.work_queue.wake import wake_for_capacity

logger = logging.getLogger("ai-tasks")


def session_factory(db):
    bind = db.get_bind()
    db.rollback()
    return sessionmaker(bind=bind, autocommit=False, autoflush=False)


def wake_task(db, task, *, reason: str) -> None:
    """按空闲名额叫醒（Vercel）；没发出去也不影响任务，巡检会补发。调用前必须已提交。"""

    source_key = next((item.source_key for item in task.items), None)
    token = "%s-%s" % (task.id, task.state_version)
    factory = session_factory(db)
    try:
        if source_key:
            with factory() as session:
                wake_for_capacity(session, source_key, reason=reason, token=token)
        ensure_sweep_chain(factory)
    except Exception:
        logger.exception("ai_task_wake_failed task_id=%s", task.id)

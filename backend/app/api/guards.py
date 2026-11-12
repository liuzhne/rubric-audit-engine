"""共享的可见性守卫：页面接口与评分助手的图调用同一组函数。

这些函数原先分散在各路由模块里（`_visible_batch`、`_visible_paper`、
`_job_or_404`……），助手的图要执行同样的读写，复制一份就会出现“改了一边漏了
另一边”。现在路由模块里的旧名字只是这里的别名，同一个函数对象
（对话评分助手方案 T4）。

只查归属与可见性，不查角色；角色由调用方 `require_organization_role` 负责。

评分任务及其下的文件、评分结果按个人隔离：教师只看得到自己建的任务，组织
管理员与平台管理员看得到本组织全部（docs/评分任务个人隔离方案.md）。看不到
的一律 404，不用 403——不泄露“这个对象存在”。
"""

from datetime import datetime
from datetime import timezone

from fastapi import HTTPException
from sqlalchemy import and_
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import CurrentPrincipal
from backend.app.db.models import EvaluationBatch
from backend.app.db.models import GradingBatch
from backend.app.db.models import Paper
from backend.app.db.models import Rubric
from backend.app.db.models import RubricImportSession
from backend.app.db.models import ScoringRun
from backend.app.services import ownership
from backend.app.services.auth import auth_active


def sees_whole_organization(principal: CurrentPrincipal) -> bool:
    """能看到本组织全部评分任务的身份：未开鉴权、平台管理员、组织管理员。"""
    return (
        not auth_active()
        or principal.platform_role == "platform_admin"
        or principal.organization_role == "org_admin"
    )


def personal_owner_id(principal: CurrentPrincipal) -> str | None:
    """需要按个人收窄时返回用户 id；能看到全组织的身份返回 None。

    评分任务、文件、评分结果、作业、复核与导出的每个入口都用它决定要不要加
    归属条件（docs/评分任务个人隔离方案.md）。新写的列表查询漏了它，教师就会
    看到同事的数据——数据库 RLS 只按组织，兜不住这一层。
    """
    return None if sees_whole_organization(principal) else principal.user_id


def _hidden(organization_id, owner_id, principal: CurrentPrincipal) -> bool:
    if principal.organization_id is not None and organization_id != principal.organization_id:
        return True
    owner = personal_owner_id(principal)
    return owner is not None and owner_id != owner


def batch_scope(principal: CurrentPrincipal) -> list:
    """评分任务列表的查询条件：组织 + （教师）本人。与 `visible_batch` 同一口径。"""
    clauses = []
    if principal.organization_id is not None:
        clauses.append(GradingBatch.organization_id == principal.organization_id)
    owner = personal_owner_id(principal)
    if owner is not None:
        clauses.append(GradingBatch.owner_id == owner)
    return clauses


def paper_scope(principal: CurrentPrincipal) -> list:
    clauses = []
    if principal.organization_id is not None:
        clauses.append(Paper.organization_id == principal.organization_id)
    owner = personal_owner_id(principal)
    if owner is not None:
        clauses.append(Paper.batch_id.in_(ownership.owned_batch_ids(owner)))
    return clauses


def run_scope(principal: CurrentPrincipal) -> list:
    clauses = []
    if principal.organization_id is not None:
        clauses.append(ScoringRun.organization_id == principal.organization_id)
    owner = personal_owner_id(principal)
    if owner is not None:
        clauses.append(ownership.run_owned_clause(owner))
    return clauses


def visible_batch(db: Session, batch_id: str, principal: CurrentPrincipal) -> GradingBatch:
    batch = db.get(GradingBatch, batch_id)
    if batch is None or _hidden(batch.organization_id, batch.owner_id, principal):
        raise HTTPException(status_code=404, detail="batch not found")
    return batch


def visible_paper(db: Session, paper_id: str, principal: CurrentPrincipal) -> Paper:
    paper = db.get(Paper, paper_id)
    if paper is None or _hidden(
        paper.organization_id, ownership.paper_owner_id(db, paper), principal
    ):
        raise HTTPException(status_code=404, detail="paper not found")
    return paper


def run_hidden(db: Session, run: ScoringRun, principal: CurrentPrincipal) -> bool:
    """评分结果对当前身份不可见；v1 与 v2 的结果守卫共用。"""
    return _hidden(run.organization_id, ownership.run_owner_id(db, run), principal)


def visible_run(db: Session, run_id: str, principal: CurrentPrincipal) -> ScoringRun:
    run = db.get(ScoringRun, run_id)
    if run is None or run_hidden(db, run, principal):
        raise HTTPException(status_code=404, detail="scoring run not found")
    return run


def visible_job(db: Session, job_id: str, principal: CurrentPrincipal):
    from backend.app.services.batch_scoring.jobs import get_batch_scoring_job

    job = get_batch_scoring_job(db, job_id)
    if job is None or _hidden(job.batch.organization_id, job.batch.owner_id, principal):
        raise HTTPException(status_code=404, detail="batch scoring job not found")
    return job


def visible_evaluation_batch(db: Session, batch_id: str, principal: CurrentPrincipal) -> EvaluationBatch:
    batch = db.get(EvaluationBatch, batch_id)
    if batch is None or _hidden(batch.organization_id, batch.owner_id, principal):
        raise HTTPException(status_code=404, detail="evaluation batch not found")
    return batch


def submission_hidden(db: Session, submission, principal: CurrentPrincipal) -> bool:
    return _hidden(
        submission.organization_id, ownership.submission_owner_id(db, submission), principal
    )


def rubric_visible_to(rubric: Rubric, principal: CurrentPrincipal) -> bool:
    """单份评分标准的可见性判定；列表过滤见 `visible_rubrics_filter`，两者同一口径。"""

    if not auth_active() or principal.platform_role == "platform_admin":
        return True
    return bool(
        rubric.visibility == "system"
        or rubric.visibility == "organization" and rubric.organization_id == principal.organization_id
        or rubric.visibility == "private" and rubric.owner_id == principal.user_id
    )


def visible_rubrics_filter(principal: CurrentPrincipal):
    """列表查询用的过滤条件；不需要过滤时返回 None。"""

    if not auth_active() or principal.platform_role == "platform_admin":
        return None
    return or_(
        Rubric.visibility == "system",
        and_(
            Rubric.visibility == "organization",
            Rubric.organization_id == principal.organization_id,
        ),
        and_(
            Rubric.visibility == "private",
            Rubric.owner_id == principal.user_id,
        ),
    )


def visible_rubric(db: Session, rubric_id: str, principal: CurrentPrincipal, *, loader=None) -> Rubric:
    rubric = loader(db, rubric_id) if loader is not None else db.get(Rubric, rubric_id)
    if rubric is None or not rubric_visible_to(rubric, principal):
        raise HTTPException(status_code=404, detail="rubric not found")
    return rubric


def visible_import_session(
    db: Session,
    session_id: str,
    principal: CurrentPrincipal,
    *,
    for_update: bool = False,
) -> RubricImportSession:
    query = select(RubricImportSession).where(RubricImportSession.id == session_id)
    if for_update:
        query = query.with_for_update()
    row = db.scalar(query)
    if row is None:
        raise HTTPException(status_code=404, detail="rubric import session not found")
    allowed = (
        not auth_active()
        or principal.platform_role == "platform_admin"
        or row.owner_id == principal.user_id
        or row.visibility == "organization"
        and row.organization_id == principal.organization_id
    )
    if not allowed:
        raise HTTPException(status_code=404, detail="rubric import session not found")
    if row.status == "draft" and row.expires_at <= datetime.now(timezone.utc).replace(tzinfo=None):
        row.status = "expired"
        row.state_version += 1
        db.commit()
        raise HTTPException(status_code=410, detail="rubric import session has expired")
    return row

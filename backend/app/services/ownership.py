"""评分任务的个人归属：查询条件与单个对象的判断（docs/评分任务个人隔离方案.md）。

教师只能看到自己建的评分任务，以及任务下的文件、评分结果、作业、复核与导出。
文件与评分结果各自有 `owner_id`，但这里**一律按所属任务的归属判断**：一个任务
下的东西要么全看得到，要么全看不到。各判各的会在三个字段不一致时出现任务可见
而某份文件不可见。

这里只有“归属于谁”的条件，不决定“谁需要被收窄”——那由
`api.guards.personal_owner_id` 按身份给出；能看全组织的身份拿到的是 `None`，
调用方据此不加条件。
"""

from sqlalchemy import or_
from sqlalchemy import select

from backend.app.db.models import EvaluationBatch
from backend.app.db.models import GradingBatch
from backend.app.db.models import Paper
from backend.app.db.models import ScoringRun
from backend.app.db.models import Submission


def owned_batch_ids(owner_id: str):
    return select(GradingBatch.id).where(GradingBatch.owner_id == owner_id)


def owned_paper_ids(owner_id: str):
    return select(Paper.id).where(Paper.batch_id.in_(owned_batch_ids(owner_id)))


def owned_submission_ids(owner_id: str):
    return select(Submission.id).where(
        Submission.evaluation_batch_id.in_(
            select(EvaluationBatch.id).where(EvaluationBatch.owner_id == owner_id)
        )
    )


def run_owned_clause(owner_id: str):
    """评分结果归属于某人：它的文件在此人的任务下，或它的提交在此人的 v2 批次下。"""
    return or_(
        ScoringRun.paper_id.in_(owned_paper_ids(owner_id)),
        ScoringRun.submission_id.in_(owned_submission_ids(owner_id)),
    )


def owned_run_ids(owner_id: str):
    return select(ScoringRun.id).where(run_owned_clause(owner_id))


def paper_owner_id(db, paper) -> str | None:
    batch = db.get(GradingBatch, paper.batch_id)
    return None if batch is None else batch.owner_id


def submission_owner_id(db, submission) -> str | None:
    batch = db.get(EvaluationBatch, submission.evaluation_batch_id)
    return None if batch is None else batch.owner_id


def run_owner_id(db, run) -> str | None:
    """评分结果所属任务的归属；两头都挂不上的旧行退回它自己的 `owner_id`。"""
    if run.paper_id is not None:
        paper = db.get(Paper, run.paper_id)
        return None if paper is None else paper_owner_id(db, paper)
    if run.submission_id is not None:
        submission = db.get(Submission, run.submission_id)
        return None if submission is None else submission_owner_id(db, submission)
    return run.owner_id

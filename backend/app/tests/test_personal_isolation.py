"""评分任务按个人隔离（docs/评分任务个人隔离方案.md）。

同一组织里的两位教师互相看不到对方的评分任务，以及任务下的文件、评分结果、
作业、复核与导出；组织管理员看得到全部。组织之间的隔离见
`test_organization_isolation.py`，这里只测同一组织内。
"""

from pathlib import Path
from types import SimpleNamespace

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy import text

from backend.app.api import guards
from backend.app.api.deps import CurrentPrincipal
from backend.app.core.config import settings
from backend.app.db import models
from backend.app.services import ownership
from backend.app.services.assistant import tools
from backend.app.services.submissions import review_tasks

ROOT = Path(__file__).resolve().parents[3]
ADMIN_PASSWORD = "bootstrap-admin-password"


def _enable_auth(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_PASSWORD", ADMIN_PASSWORD)
    monkeypatch.setattr(settings, "AUTH_SECRET", "d" * 48)
    monkeypatch.setattr(settings, "REGISTRATION_MODE", "invite_only")
    monkeypatch.setattr(settings, "AUTH_COOKIE_SECURE", False)


def _login_admin(client):
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}).status_code == 204
    return client.get("/api/organizations").json()[0]["id"]


def _member(client, username, role):
    """由平台管理员邀请进默认组织并完成注册；返回用户 id。"""
    organization_id = _login_admin(client)
    email = "%s@example.test" % username
    invitation = client.post(f"/api/organizations/{organization_id}/members", json={"email": email, "role": role})
    assert invitation.status_code == 201, invitation.text
    registered = client.post("/api/auth/register", json={
        "username": username, "email": email, "display_name": username,
        "password": "%s correct password" % username,
        "invitation_token": invitation.json()["invitation_token"],
    })
    assert registered.status_code == 201, registered.text
    with client.session_factory() as session:
        return session.scalar(select(models.User.id).where(models.User.username == username)), organization_id


def _login(client, username, organization_id):
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={
        "username": username, "password": "%s correct password" % username,
    }).status_code == 204
    assert client.post("/api/auth/organization-context", json={"organization_id": organization_id}).status_code == 200


def _task(client, *, owner_id, organization_id, name):
    """一个评分任务，带一份文件、一个评分结果、一个作业和一个人工复核任务。"""
    with client.session_factory() as session:
        rubric = models.Rubric(name=name + " rubric", version="v1", total_score=100,
                               owner_id=owner_id, organization_id=organization_id)
        session.add(rubric)
        session.flush()
        batch = models.GradingBatch(name=name, rubric_id=rubric.id, owner_id=owner_id,
                                    organization_id=organization_id)
        session.add(batch)
        session.flush()
        # 文件与结果上的 owner_id 故意留空：可见性只看所属任务的归属。
        paper = models.Paper(batch_id=batch.id, organization_id=organization_id,
                             file_name=name + ".docx", file_path=name + ".docx")
        session.add(paper)
        session.flush()
        run = models.ScoringRun(paper_id=paper.id, rubric_id=rubric.id, organization_id=organization_id)
        session.add(run)
        session.flush()
        job = models.BatchScoringJob(grading_batch_id=batch.id, status="failed", generation=1,
                                     observation_policy={}, observation_policy_hash="0" * 64)
        session.add(job)
        review = models.ManualReviewTask(
            organization_id=organization_id, scoring_run_id=run.id, criterion_code="C1",
            trigger_code="low_evidence", trigger_message="needs a human",
        )
        session.add(review)
        session.commit()
        return SimpleNamespace(batch=batch.id, paper=paper.id, run=run.id, job=job.id, review=review.id)


def _ids(response):
    assert response.status_code == 200, response.text
    return {row["id"] for row in response.json()}


def test_a_teacher_sees_only_their_own_tasks_and_everything_under_them(client, monkeypatch):
    _enable_auth(monkeypatch)
    alice, organization_id = _member(client, "alice", "teacher")
    bob, _ = _member(client, "bob", "teacher")
    hers = _task(client, owner_id=alice, organization_id=organization_id, name="alice task")
    his = _task(client, owner_id=bob, organization_id=organization_id, name="bob task")

    _login(client, "bob", organization_id)
    assert _ids(client.get("/api/batches")) == {his.batch}
    assert _ids(client.get("/api/papers")) == {his.paper}
    assert _ids(client.get("/api/papers", params={"batch_id": hers.batch})) == set()
    assert _ids(client.get("/api/scoring-runs")) == {his.run}
    assert _ids(client.get("/api/scoring-runs", params={"batch_id": hers.batch})) == set()
    assert _ids(client.get("/api/batch-scoring-jobs")) == {his.job}
    assert _ids(client.get("/api/v2/manual-review-tasks")) == {his.review}
    assert _ids(client.get("/api/v2/manual-review-tasks", params={"batch_id": hers.batch})) == set()
    assert client.get("/api/batches/overview").json()["total_batches"] == 1

    # 指名道姓也拿不到：一律 404，不用 403 泄露“它存在”。
    for path in (
        "/api/batches/%s" % hers.batch,
        "/api/batches/%s/summary" % hers.batch,
        "/api/batches/%s/progress" % hers.batch,
        "/api/batches/%s/export.xlsx" % hers.batch,
        "/api/batches/%s/export-history" % hers.batch,
        "/api/papers/%s" % hers.paper,
        "/api/papers/%s/parsed" % hers.paper,
        "/api/scoring-runs/%s" % hers.run,
        "/api/scoring-runs/%s/items" % hers.run,
        "/api/scoring-runs/%s/report" % hers.run,
        "/api/scoring-runs/%s/export.json" % hers.run,
        "/api/batch-scoring-jobs/%s" % hers.job,
        "/api/v2/manual-review-tasks/%s" % hers.review,
    ):
        assert client.get(path).status_code == 404, path
    assert client.patch("/api/batches/%s" % hers.batch, json={"name": "stolen"}).status_code == 404
    assert client.post("/api/batches/%s/start" % hers.batch).status_code == 404
    assert client.post("/api/papers/%s/score" % hers.paper).status_code == 404
    assert client.post("/api/v2/manual-review-tasks/%s/claim" % hers.review, json={"version": 1}).status_code == 404
    assert client.post("/api/v2/manual-review-tasks/%s/release" % hers.review, json={"version": 1}).status_code == 404
    assert client.post("/api/v2/manual-review-tasks/%s/resolve" % hers.review, json={
        "version": 1, "final_score": 1, "reason": "steal",
        "evidence": [{"evidence_unit_id": "u1", "quote": "x"}],
    }).status_code == 404

    # 自己的照常可见。
    assert client.get("/api/batches/%s/summary" % his.batch).status_code == 200
    assert client.get("/api/papers/%s" % his.paper).status_code == 200
    assert client.get("/api/scoring-runs/%s" % his.run).status_code == 200
    assert client.get("/api/v2/manual-review-tasks/%s" % his.review).status_code == 200


def test_an_organization_admin_sees_every_task_with_its_owner(client, monkeypatch):
    _enable_auth(monkeypatch)
    alice, organization_id = _member(client, "alice", "teacher")
    _member(client, "dean", "org_admin")
    hers = _task(client, owner_id=alice, organization_id=organization_id, name="alice task")
    orphan = _task(client, owner_id=None, organization_id=organization_id, name="legacy task")

    _login(client, "dean", organization_id)
    batches = {row["id"]: row for row in client.get("/api/batches").json()}
    assert set(batches) == {hers.batch, orphan.batch}
    assert batches[hers.batch]["owner_name"] == "alice"
    assert batches[orphan.batch]["owner_name"] is None
    assert client.get("/api/batches/%s" % hers.batch).json()["owner_name"] == "alice"
    assert _ids(client.get("/api/papers")) == {hers.paper, orphan.paper}
    assert _ids(client.get("/api/scoring-runs")) == {hers.run, orphan.run}
    assert _ids(client.get("/api/batch-scoring-jobs")) == {hers.job, orphan.job}
    assert _ids(client.get("/api/v2/manual-review-tasks")) == {hers.review, orphan.review}
    assert client.get("/api/scoring-runs/%s" % hers.run).status_code == 200

    # 没有归属的旧任务只有管理员看得到。
    _login(client, "alice", organization_id)
    assert _ids(client.get("/api/batches")) == {hers.batch}
    assert client.get("/api/batches/%s" % orphan.batch).status_code == 404
    assert client.get("/api/papers/%s" % orphan.paper).status_code == 404


def test_the_assistant_lists_only_the_tasks_the_teacher_owns(client, monkeypatch):
    _enable_auth(monkeypatch)
    alice, organization_id = _member(client, "alice", "teacher")
    bob, _ = _member(client, "bob", "teacher")
    hers = _task(client, owner_id=alice, organization_id=organization_id, name="alice task")
    _task(client, owner_id=bob, organization_id=organization_id, name="bob task")

    with client.session_factory() as session:
        ctx = SimpleNamespace(db=session, principal=CurrentPrincipal(alice, organization_id, "teacher", "user"))
        listed = tools._list_recent_batches(ctx, tools.RecentBatchesParams())
        assert [row["id"] for row in listed] == [hers.batch]
        # 服务层直接调用时同一口径：给了归属就收窄，不给（管理员）就是全组织。
        assert len(review_tasks.list_manual_tasks(session, organization_id=organization_id, owner_id=alice)) == 1
        assert len(review_tasks.list_manual_tasks(session, organization_id=organization_id)) == 2


def test_who_is_narrowed_to_their_own_tasks(monkeypatch):
    monkeypatch.setattr(guards, "auth_active", lambda: True)
    assert guards.personal_owner_id(CurrentPrincipal("u", "o", "teacher", "user")) == "u"
    assert guards.personal_owner_id(CurrentPrincipal("u", "o", "member", "user")) == "u"
    assert guards.personal_owner_id(CurrentPrincipal("u", "o", None, "user")) == "u"
    assert guards.personal_owner_id(CurrentPrincipal("u", "o", "org_admin", "user")) is None
    assert guards.personal_owner_id(CurrentPrincipal("u", "o", None, "platform_admin")) is None
    # 关闭鉴权的本地开发模式不收窄。
    monkeypatch.setattr(guards, "auth_active", lambda: False)
    assert guards.personal_owner_id(CurrentPrincipal("u", None, None, "developer")) is None


def test_a_v2_result_belongs_to_the_owner_of_its_evaluation_batch():
    rows = {
        (models.Submission, "s1"): SimpleNamespace(evaluation_batch_id="e1"),
        (models.EvaluationBatch, "e1"): SimpleNamespace(owner_id="alice"),
    }
    db = SimpleNamespace(get=lambda model, key: rows.get((model, key)))
    # 结果行自己的 owner_id 不算数：按所属批次判断。
    run = SimpleNamespace(paper_id=None, submission_id="s1", owner_id="someone-else")
    assert ownership.run_owner_id(db, run) == "alice"
    assert ownership.run_owner_id(db, SimpleNamespace(paper_id=None, submission_id="gone", owner_id="x")) is None
    assert ownership.run_owner_id(db, SimpleNamespace(paper_id=None, submission_id=None, owner_id="x")) == "x"


def test_0043_backfills_missing_owners_from_the_creator_and_indexes_them(monkeypatch, tmp_path):
    url = "sqlite+pysqlite:///%s" % (tmp_path / "owner.db")
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    config = Config()
    config.set_main_option("script_location", str(ROOT / "alembic"))
    command.upgrade(config, "0042_assistant_message_client_id")
    engine = create_engine(url)
    with engine.begin() as connection:
        # SQLite 默认不检查外键：评分标准与用户行不需要真的存在。
        columns = [row[1] for row in connection.execute(text("PRAGMA table_info(grading_batches)"))
                   if row[3] and row[4] is None and row[1] not in {"id", "name", "rubric_id"}]
        insert = text(
            "INSERT INTO grading_batches (id, name, rubric_id, owner_id, created_by%s) "
            "VALUES (:id, :id, 'r', :owner_id, :created_by%s)"
            % ("".join(", " + column for column in columns), "".join(", :" + column for column in columns))
        )
        filler = {column: (0 if "version" in column or "count" in column else "draft") for column in columns}
        now = {column: "2026-11-10 00:00:00" for column in columns if column.endswith("_at")}
        for batch_id, owner_id, created_by in (("kept", "alice", "bob"), ("filled", None, "bob"), ("orphan", None, None)):
            connection.execute(insert, {"id": batch_id, "owner_id": owner_id, "created_by": created_by, **filler, **now})
    command.upgrade(config, "0043_batch_owner_isolation")
    with engine.begin() as connection:
        owners = dict(connection.execute(text("SELECT id, owner_id FROM grading_batches")).all())
        indexes = {row[1] for row in connection.execute(text("PRAGMA index_list(grading_batches)"))}
        v2_indexes = {row[1] for row in connection.execute(text("PRAGMA index_list(evaluation_batches)"))}
    assert owners == {"kept": "alice", "filled": "bob", "orphan": None}
    assert "ix_grading_batches_owner_id" in indexes
    assert "ix_evaluation_batches_owner_id" in v2_indexes
    command.downgrade(config, "0042_assistant_message_client_id")
    with engine.begin() as connection:
        # 降级只删索引：回填的归属留着，清掉它会丢失“这个任务是谁的”。
        assert dict(connection.execute(text("SELECT id, owner_id FROM grading_batches")).all()) == owners
    engine.dispose()

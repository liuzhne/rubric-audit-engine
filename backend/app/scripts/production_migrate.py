"""生产数据库迁移的实例侧执行：.github/workflows/production-migrate.yml 经 SSH 调用。

数据库改成同机的 db 容器（不对外开端口）之后，GitHub 运行器连不到它。工作流把本提交的镜像
送到实例，用一次性容器（deploy/production/ops.sh）跑这里的子命令。切换前数据还在 Supabase 时，
走的也是这条路：实例连 Supabase 的 Session pooler。

连接串从环境变量读（``scripts/production_env.py --kind ops`` 生成，按实例上的 DATA_PLANE 选）：

- ``MIGRATION_DATABASE_URL``：owner，只用于 survey / pending / upgrade。
- ``RUNTIME_DATABASE_URL``：运行角色 ``pgs_app``，verify 只用它。owner 什么都读得到，证明不了应用
  能不能工作——0026/0027 的新表缺授权，就是只有用运行角色验证才暴露出来（见 0029）。

    python -m backend.app.scripts.production_migrate survey|pending|upgrade|verify|check

``check`` 是部署任务（ci.yml 的 deploy-lightsail-production）切换版本前的门：数据库的版本与本镜像
迁移目录的 head 不一致就返回 1，部署停在切换之前，线上继续跑旧版本。它只读一行版本号，用运行
角色（``alembic_version`` 对 ``pgs_app`` 只给了 SELECT），部署任务因此不需要 owner 连接串。

只打印版本、角色名、表名与行数，不打印行内容（生产库含学生 PII），也不打印连接串。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy import inspect
from sqlalchemy import select
from sqlalchemy import table
from sqlalchemy import text

from backend.app.db.session import engine_options
from backend.app.services.deployment.postgres_verifier import RUNTIME_ACCESS_TABLES


COMMANDS = ("survey", "pending", "upgrade", "verify", "check")
#: 每个子命令用哪一个连接串。
CREDENTIAL = {
    "survey": "MIGRATION_DATABASE_URL",
    "pending": "MIGRATION_DATABASE_URL",
    "upgrade": "MIGRATION_DATABASE_URL",
    "verify": "RUNTIME_DATABASE_URL",
    "check": "RUNTIME_DATABASE_URL",
}
#: 迁移目录按本文件定位：镜像里是 /app/alembic，与当前目录无关。
ALEMBIC_DIRECTORY = pathlib.Path(__file__).resolve().parents[3] / "alembic"


def _url(raw: str) -> str:
    for prefix in ("postgres://", "postgresql://"):
        if raw.startswith(prefix):
            return "postgresql+psycopg://" + raw[len(prefix) :]
    return raw


def _engine(raw: str):
    url = _url(raw)
    return create_engine(url, **engine_options(url))


def _run(command, url, runner=subprocess.run) -> int:
    # 子进程只拿到这一步该用的连接串：alembic 与 verify_postgres_ops 都读 DATABASE_URL。
    env = dict(os.environ, DATABASE_URL=_url(url))
    for name in CREDENTIAL.values():
        env.pop(name, None)
    return runner(command, env=env, check=False).returncode


def survey(url) -> int:
    engine = _engine(url)
    try:
        with engine.connect() as connection:
            head = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            print("current alembic head:", head)
            print("current_user:", connection.execute(text("SELECT current_user")).scalar_one())
            for name in sorted(inspect(connection).get_table_names(schema="public")):
                count = connection.execute(select(func.count()).select_from(table(name))).scalar_one()
                if count:
                    print("  %-34s %s" % (name, count))
            connection.rollback()
    finally:
        engine.dispose()
    return 0


def verify(url, runner=subprocess.run) -> int:
    code = _run([sys.executable, "-m", "backend.app.scripts.verify_postgres_ops"], url, runner)
    if code:
        return code
    engine = _engine(url)
    try:
        # 运行角色真的能读每一张需要授权的表——迁移成功不等于应用可用。新建表要加进
        # postgres_verifier.RUNTIME_ACCESS_TABLES，否则这道验证会随时间失效。
        with engine.connect() as connection:
            for name in RUNTIME_ACCESS_TABLES:
                connection.execute(select(func.count()).select_from(table(name))).scalar_one()
                print("runtime role can read:", name)
            connection.rollback()
    finally:
        engine.dispose()
    return 0


def code_head() -> str:
    """本镜像（本提交）迁移目录的 head。"""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIRECTORY))
    return ScriptDirectory.from_config(config).get_current_head()


def check(url, expected=None) -> int:
    """数据库版本与本镜像的 head 一致才返回 0。只读，不改任何东西。"""
    expected = expected or code_head()
    engine = _engine(url)
    try:
        with engine.connect() as connection:
            if inspect(connection).has_table("alembic_version"):
                found = sorted(connection.execute(text("SELECT version_num FROM alembic_version")).scalars())
            else:
                found = []
            connection.rollback()
    finally:
        engine.dispose()
    if found == [expected]:
        print("数据库版本与本次部署一致：%s" % expected)
        return 0
    print(
        "数据库版本与本次部署不一致：数据库为 %s，本次部署需要 %s。\n"
        "没有切换版本，线上仍是原来的版本。先运行 pgs-production-migrate"
        "（分支选 main，expected_head=%s，先 dry_run 再正式执行），然后重新运行这个部署任务。"
        % ("、".join(found) or "空（没有 alembic_version）", expected, expected),
        file=sys.stderr,
    )
    return 1


def main(argv=None, environ=None, runner=subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(argv)
    environ = os.environ if environ is None else environ
    name = CREDENTIAL[args.command]
    url = environ.get(name)
    if not url:
        print("缺少环境变量 %s" % name, file=sys.stderr)
        return 2
    try:
        if args.command == "survey":
            return survey(url)
        if args.command == "pending":
            return _run(["alembic", "history", "--indicate-current"], url, runner)
        if args.command == "check":
            return check(url)
        if args.command == "upgrade":
            return _run(["alembic", "upgrade", "head"], url, runner)
        return verify(url, runner)
    except Exception as exc:
        # 驱动的错误文本可能回显连接串片段：只给类型。
        print("%s 失败（%s）；为保护凭据，未输出原始错误。" % (args.command, type(exc).__name__), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Model-made reading aids derived from a document snapshot.

Revision ID: 0046_document_derivations
Revises: 0045_cached_prompt_tokens
Create Date: 2026-11-15

Whole-document rules only see the paragraphs a keyword ranking picks.  The
next step (docs/分节摘要定位方案.md) lets a model read a per-section summary of
the paper and point at the sections a rule needs, then judges on the original
text of those sections.  Summaries and locate results are written by a model,
so they cannot live in the immutable, content-addressed document snapshot;
they are frozen here and their hash enters the decision identity.

One row per (snapshot, kind, generation key).  ``key_hash`` covers everything
the result depends on (snapshot hash, tree and prompt versions, model), so a
new prompt or model writes a new row and old rows stay untouched.

One new table, so the 0023 rule applies: grant ``pgs_app`` and enable RLS in
the migration itself.  Production runs as ``pgs_app`` without DDL rights;
forgetting the grant does not fail the migration, it makes scoring fail with
"permission denied" right after a successful deploy.

The table is a cache: dropping it never changes a stored score, a later run
simply generates the rows again (with a different decision identity).  The
downgrade therefore drops it without checking for rows, like 0033's ledger.
"""

from alembic import op
import sqlalchemy as sa


revision = "0046_document_derivations"
down_revision = "0045_cached_prompt_tokens"
branch_labels = None
depends_on = None


TABLE = "document_derivations"


def _hex_check(column):
    """64 位小写十六进制；SQLite 与 PostgreSQL 都能执行（与模型里的写法相同）。"""

    stripped = column
    for character in "0123456789abcdef":
        stripped = "replace(%s, '%s', '')" % (stripped, character)
    return "length(%s) = 64 AND length(%s) = 0" % (column, stripped)


def _configure_postgres_runtime_access():
    connection = op.get_bind()
    if connection.dialect.name != "postgresql":
        return
    role_exists = connection.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pgs_app')")
    ).scalar()
    if not role_exists:
        return
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE %s TO pgs_app" % TABLE)
    op.execute("ALTER TABLE %s ENABLE ROW LEVEL SECURITY" % TABLE)
    op.execute(
        "CREATE POLICY pgs_app_dml ON %s FOR ALL TO pgs_app "
        "USING (true) WITH CHECK (true)" % TABLE
    )


def upgrade():
    op.create_table(
        TABLE,
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("model_provider", sa.String(length=100), nullable=True),
        sa.Column("model_name", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "kind IN ('section_summary', 'section_locate')", name="ck_document_derivations_kind"
        ),
        sa.CheckConstraint(_hex_check("key_hash"), name="ck_document_derivations_key_hash"),
        sa.CheckConstraint(_hex_check("payload_hash"), name="ck_document_derivations_payload_hash"),
        sa.ForeignKeyConstraint(
            ["document_snapshot_id"], ["document_snapshots.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_snapshot_id", "kind", "key_hash", name="uq_document_derivations_snapshot_kind_key"
        ),
    )
    _configure_postgres_runtime_access()


def downgrade():
    op.drop_table(TABLE)

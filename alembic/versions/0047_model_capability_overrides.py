"""Model thinking capabilities registered by platform admins.

Revision ID: 0047_model_capability_overrides
Revises: 0046_document_derivations
Create Date: 2026-11-17

The thinking capability table (``llm/thinking_capabilities.py``) decides how
each model's thinking is switched off or turned down.  Until now every new
model needed a code change and a release; a model missing from the table is
assumed to be able to turn thinking off, and when it cannot, every paper of a
scoring job stops with ``PROVIDER_THINKING_NOT_DISABLED``.

Platform admins can now register a record on the "模型管理" page
(docs/模型管理页与评分思考修正方案.md 3.2).  The code table stays the default;
enabled rows here take part in the same longest-prefix match and win a tie.
Rows are global, not per organization: whether a model can turn thinking off
is a vendor fact.

One new table, so the 0023 rule applies: grant ``pgs_app`` and enable RLS in
the migration itself.  The downgrade refuses while any row exists: dropping
the table would silently change how those models are called.
"""

from alembic import op
import sqlalchemy as sa


revision = "0047_model_capability_overrides"
down_revision = "0046_document_derivations"
branch_labels = None
depends_on = None


TABLE = "model_capability_overrides"


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
        sa.Column("protocol", sa.String(length=16), nullable=False),
        sa.Column("host_family", sa.String(length=32), nullable=False),
        sa.Column("model_prefix", sa.String(length=120), nullable=False),
        sa.Column("can_disable", sa.Boolean(), nullable=False),
        sa.Column("off_request", sa.JSON(), nullable=True),
        sa.Column("lowest_effort", sa.String(length=32), nullable=True),
        sa.Column("effort_request", sa.JSON(), nullable=True),
        sa.Column("max_output_tokens", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=False),
        sa.Column("source_url", sa.String(length=500), nullable=False),
        sa.Column("verified_on", sa.String(length=10), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column("updated_by", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "protocol IN ('chat', 'responses', 'messages')", name="ck_model_capability_overrides_protocol"
        ),
        sa.CheckConstraint("revision >= 1", name="ck_model_capability_overrides_revision"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("protocol", "host_family", "model_prefix", name="uq_model_capability_overrides_key"),
    )
    _configure_postgres_runtime_access()


def downgrade():
    connection = op.get_bind()
    count = connection.execute(sa.text("SELECT COUNT(*) FROM %s" % TABLE)).scalar()
    if count:
        raise RuntimeError(
            "refusing to downgrade 0047: %d model capability override(s) registered; "
            "dropping them would change how those models are called" % count
        )
    op.drop_table(TABLE)

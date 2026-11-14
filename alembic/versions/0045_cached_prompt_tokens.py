"""Record provider-reported cached prompt tokens.

Revision ID: 0045_cached_prompt_tokens
Revises: 0044_atomic_rule_scope
Create Date: 2026-11-15

Scoring requests are being reordered so that rules reading the same chapter
share one prompt prefix (docs/论文章节树与稳定前缀取证方案.md, stage 2).  Whether a
provider's prefix cache actually hits could not be told from the scoring side:
only the assistant recorded cached tokens.

One counter per scoring run and per usage-ledger row:

- ``cached_tokens``: input tokens the provider reported as served from its
  cache.  They are already part of ``prompt_tokens``.  Zero also means "the
  provider does not report it", so it is an observation, not proof of a miss.

Platform-model usage stays on ``scoring_runs`` (``ai_usage_ledger`` only
projects BYOK connections), so both tables get the column, as in 0040.

Only columns are added: the table-level ``pgs_app`` grants and RLS policies of
0023/0029 already cover them.  The downgrade refuses while any row carries a
non-zero value, because the observation cannot be rebuilt afterwards.
"""

from alembic import op
import sqlalchemy as sa


revision = "0045_cached_prompt_tokens"
down_revision = "0044_atomic_rule_scope"
branch_labels = None
depends_on = None


TABLES = ("scoring_runs", "ai_usage_ledger")
COLUMN = "cached_tokens"


def upgrade():
    for table in TABLES:
        op.add_column(
            table,
            sa.Column(COLUMN, sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade():
    connection = op.get_bind()
    for table in TABLES:
        observed = connection.execute(
            sa.text("SELECT count(*) FROM %s WHERE cached_tokens <> 0" % table)
        ).scalar_one()
        if observed:
            raise RuntimeError(
                "0045 downgrade would erase cached-token observations of %d %s row(s); "
                "archive them before downgrading" % (observed, table)
            )
    for table in TABLES:
        with op.batch_alter_table(table) as batch:
            batch.drop_column(COLUMN)

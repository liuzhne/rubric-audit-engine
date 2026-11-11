"""Batch owner isolation: backfill and index the owner of scoring tasks.

Revision ID: 0043_batch_owner_isolation
Revises: 0042_assistant_message_client_id
Create Date: 2026-11-10

Scoring tasks, their files and their results are now visible only to the
teacher who owns the task (docs/评分任务个人隔离方案.md); organization admins
still see the whole organization.  ``owner_id`` has been written on creation
for a long time but was never used for visibility, so two things are needed
before the guards start reading it:

* rows whose ``owner_id`` is NULL but whose ``created_by`` is known get the
  creator as owner — otherwise a teacher's older tasks would vanish from their
  own list.  Rows with neither stay NULL and are visible to admins only.
* every list query now filters on ``owner_id``, so both batch tables get an
  index on it.

No table is created: the ``pgs_app`` grants and RLS policies of the existing
tables already cover an index and an UPDATE.  The downgrade only drops the
indexes; the backfilled owners stay, because nothing read the column before
and clearing it would lose who a task belongs to.
"""

from alembic import op
import sqlalchemy as sa


revision = "0043_batch_owner_isolation"
down_revision = "0042_assistant_message_client_id"
branch_labels = None
depends_on = None

TABLES = ("grading_batches", "evaluation_batches")


def upgrade():
    for table in TABLES:
        op.execute(
            sa.text(
                "UPDATE %s SET owner_id = created_by "
                "WHERE owner_id IS NULL AND created_by IS NOT NULL" % table
            )
        )
        op.create_index("ix_%s_owner_id" % table, table, ["owner_id"])


def downgrade():
    for table in TABLES:
        op.drop_index("ix_%s_owner_id" % table, table_name=table)

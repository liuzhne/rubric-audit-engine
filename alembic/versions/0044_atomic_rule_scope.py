"""Rule applicable scope: ``atomic_rules.scope`` and the ``rule_scope`` AI task.

Revision ID: 0044_atomic_rule_scope
Revises: 0043_batch_owner_isolation
Create Date: 2026-11-11

Each rule's applicable scope is now returned by the model while drafting the
rule, or filled in by the new ``rule_scope`` AI task for rules that were never
drafted (docs/评分细则适用范围方案.md).  People no longer type it, so the old
free-text ``applies_to`` is not reused: the new nullable JSON column holds
``{type, sections, source, vocabulary}`` and NULL means "the whole document",
which is also how every existing rule is treated.

``ck_ai_tasks_kind`` gains ``rule_scope``.  No table is created, so the 0023
rule (grant + RLS for ``pgs_app``) does not apply; the existing grants and
policies cover the new column.

The downgrade refuses while any rule has a scope or any ``rule_scope`` task
exists: dropping them would silently move those rules back to whole-document
scoring and lose the task history.
"""

from alembic import op
import sqlalchemy as sa


revision = "0044_atomic_rule_scope"
down_revision = "0043_batch_owner_isolation"
branch_labels = None
depends_on = None


CONSTRAINT = "ck_ai_tasks_kind"
OLD_KINDS = "('rule_draft', 'unit_classification', 'rule_review', 'structure_suggestion')"
NEW_KINDS = "('rule_draft', 'unit_classification', 'rule_review', 'structure_suggestion', 'rule_scope')"


def _replace_kind_check(kinds: str) -> None:
    # SQLite cannot alter a CHECK in place; batch mode rebuilds the table and
    # keeps the other constraints and indexes.
    with op.batch_alter_table("ai_tasks") as batch:
        batch.drop_constraint(CONSTRAINT, type_="check")
        batch.create_check_constraint(CONSTRAINT, "kind IN %s" % kinds)


def upgrade() -> None:
    op.add_column("atomic_rules", sa.Column("scope", sa.JSON(), nullable=True))
    _replace_kind_check(NEW_KINDS)


def downgrade() -> None:
    connection = op.get_bind()
    scoped = connection.execute(
        sa.text("SELECT count(*) FROM atomic_rules WHERE scope IS NOT NULL")
    ).scalar_one()
    tasks = connection.execute(
        sa.text("SELECT count(*) FROM ai_tasks WHERE kind = 'rule_scope'")
    ).scalar_one()
    if scoped or tasks:
        raise RuntimeError(
            "0044 downgrade refused: %d atomic rule(s) carry an applicable scope and "
            "%d rule_scope AI task(s) exist. Dropping them would move those rules back "
            "to whole-document scoring and lose the task history." % (scoped, tasks)
        )
    _replace_kind_check(OLD_KINDS)
    with op.batch_alter_table("atomic_rules") as batch:
        batch.drop_column("scope")

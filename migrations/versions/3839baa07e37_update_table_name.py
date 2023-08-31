"""update table name

Revision ID: 3839baa07e37
Revises: faef5ed72a73
Create Date: 2023-08-31 16:27:36.947948

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "3839baa07e37"
down_revision = "faef5ed72a73"
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table("users", "user")


def downgrade():
    op.rename_table("user", "users")

"""update size of pretrained model description column

Revision ID: 168234a996ab
Revises: 6c8134aef069
Create Date: 2023-08-27 05:05:06.057757

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "168234a996ab"
down_revision = "6c8134aef069"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "pretrained_model",
        "description",
        existing_type=mysql.VARCHAR(length=500),
        type_=sa.String(length=1000),
        existing_nullable=False,
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "pretrained_model",
        "description",
        existing_type=sa.String(length=128),
        type_=mysql.VARCHAR(length=255),
        existing_nullable=False,
    )
    # ### end Alembic commands ###

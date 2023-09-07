"""add missing column of FinetuningModel

Revision ID: 6c8134aef069
Revises: eea5902b0e33
Create Date: 2023-08-27 04:27:04.179801

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = "6c8134aef069"
down_revision = "eea5902b0e33"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "finetuning_model",
        sa.Column(
            "fm_name", mysql.VARCHAR(length=50), nullable=False, comment="파인튜닝 모델의 이름"
        ),
    )


def downgrade():
    op.drop_column("finetuning_model", "fm_name")

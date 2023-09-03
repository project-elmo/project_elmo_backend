"""add setting table

Revision ID: b0a1bc00c080
Revises: e96512344a4c
Create Date: 2023-09-03 17:16:41.833325

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'b0a1bc00c080'
down_revision = 'e96512344a4c'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('setting',
    sa.Column('set_no', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model_path', sa.String(length=1024), nullable=True),
    sa.Column('result_path', sa.String(length=1024), nullable=True),
    sa.Column('is_gpu', sa.Boolean(), nullable=False),
    sa.PrimaryKeyConstraint('set_no')
    )
    op.alter_column('message', 'test_no',
               existing_type=mysql.INTEGER(display_width=11),
               comment='1 for user, otherwise 0',
               existing_nullable=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('message', 'test_no',
               existing_type=mysql.INTEGER(display_width=11),
               comment=None,
               existing_comment='1 for user, otherwise 0',
               existing_nullable=False)
    op.drop_table('setting')
    # ### end Alembic commands ###

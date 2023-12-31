"""Add new columns in trainint_parameter table

Revision ID: f0f5f640ba3f
Revises: e4e34deaa93a
Create Date: 2023-08-25 12:20:47.317434

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'f0f5f640ba3f'
down_revision = 'e4e34deaa93a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('training_parameter', sa.Column('model_name', sa.String(length=255), nullable=False))
    op.add_column('training_parameter', sa.Column('epochs', sa.Integer(), nullable=False))
    op.add_column('training_parameter', sa.Column('save_strategy', sa.String(length=50), nullable=False))
    op.add_column('training_parameter', sa.Column('logging_strategy', sa.String(length=50), nullable=False))
    op.add_column('training_parameter', sa.Column('evaluation_strategy', sa.String(length=50), nullable=False))
    op.add_column('training_parameter', sa.Column('learning_rate', sa.Float(), nullable=False))
    op.add_column('training_parameter', sa.Column('weight_decay', sa.Float(), nullable=False))
    op.add_column('training_parameter', sa.Column('batch_size', sa.Integer(), nullable=False))
    op.add_column('training_parameter', sa.Column('eval_steps', sa.Integer(), nullable=False))
    op.add_column('training_parameter', sa.Column('save_steps', sa.Integer(), nullable=False))
    op.add_column('training_parameter', sa.Column('save_total_limits', sa.Integer(), nullable=True))
    op.add_column('training_parameter', sa.Column('run_on_gpu', sa.Boolean(), nullable=False))
    op.add_column('training_parameter', sa.Column('load_best_at_the_end', sa.Boolean(), nullable=False))
    op.drop_column('training_parameter', 'parameters')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('training_parameter', sa.Column('parameters', mysql.VARCHAR(length=1000), nullable=False, comment='json 형식의 긴 텍스트로 저장. 예: {"batch_size": 32, "learning_rate": 1e2222, ...}'))
    op.drop_column('training_parameter', 'load_best_at_the_end')
    op.drop_column('training_parameter', 'run_on_gpu')
    op.drop_column('training_parameter', 'save_total_limits')
    op.drop_column('training_parameter', 'save_steps')
    op.drop_column('training_parameter', 'eval_steps')
    op.drop_column('training_parameter', 'batch_size')
    op.drop_column('training_parameter', 'weight_decay')
    op.drop_column('training_parameter', 'learning_rate')
    op.drop_column('training_parameter', 'evaluation_strategy')
    op.drop_column('training_parameter', 'logging_strategy')
    op.drop_column('training_parameter', 'save_strategy')
    op.drop_column('training_parameter', 'epochs')
    op.drop_column('training_parameter', 'model_name')
    # ### end Alembic commands ###

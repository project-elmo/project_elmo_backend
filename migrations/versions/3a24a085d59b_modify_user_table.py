"""Modify user table

Revision ID: 3a24a085d59b
Revises: ea146646c535
Create Date: 2023-08-23 17:24:45.886671

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "3a24a085d59b"
down_revision = "ea146646c535"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE `fastapi`.`users` \
            CHANGE COLUMN `id` `id` VARCHAR(50) NOT NULL , \
            CHANGE COLUMN `user_no` `user_no` INT(11) NOT NULL AUTO_INCREMENT ,\
            DROP PRIMARY KEY,\
            ADD PRIMARY KEY (`user_no`);\
            ;\
            "
    )
    # ### end Alembic commands ###


def downgrade():
    op.execute(
        "ALTER TABLE `fastapi`.`users` \
            CHANGE COLUMN `id` `id` BIGINT(20) NOT NULL AUTO_INCREMENT, \
            CHANGE COLUMN `user_no` `user_no` INT(11) NOT NULL,\
            DROP PRIMARY KEY,\
            ADD PRIMARY KEY (`id`);\
            ;\
            "
    )

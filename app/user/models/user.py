from sqlalchemy import Column, String, Integer, Boolean

from core.db import Base
from core.db.mixins import TimestampMixin
from sqlalchemy.orm import relationship


class User(Base, TimestampMixin):
    __tablename__ = "user"

    user_no = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String(50), nullable=False)
    password = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    nickname = Column(String(255), nullable=False, unique=True)
    is_admin = Column(Boolean, default=False)

    finetuning_models = relationship("FinetuningModel", backref="user")

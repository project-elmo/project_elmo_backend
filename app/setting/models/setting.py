from sqlalchemy import Column, String, Integer, Boolean

from core.db import Base


class ElmoSetting(Base):
    __tablename__ = "setting"
    set_no = Column(Integer, primary_key=True, autoincrement=True)
    model_path = Column(String(1024))
    result_path = Column(String(1024))
    is_gpu = Column(Boolean, nullable=False)

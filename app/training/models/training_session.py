from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from core.db import Base


class TrainingSession(Base):
    __tablename__ = "training_session"
    session_no = Column(
        Integer, primary_key=True, autoincrement=True, comment="AUTO_INCREMENT"
    )
    fm_no = Column(Integer, ForeignKey("finetuning_model.fm_no"), nullable=False)
    parent_session_no = Column(Integer, nullable=False, comment="부모 모델의 세션번호")
    start_time = Column(
        DateTime, default=func.now(), nullable=False, comment="YYYY-MM-DD HH:MM:SS"
    )
    end_time = Column(
        DateTime, default=func.now(), nullable=False, comment="YYYY-MM-DD HH:MM:SS"
    )
    ts_model_name = Column(
        String(50), nullable=False, comment="해당 세션으로 파인튜닝된 모델의 이름-기본값: epoch, loss로 표시"
    )
    uuid = Column(
        String(50), nullable=False, comment="저장 폴더명에 사용되는 uuid"
    )

    finetuning_model = relationship(
        "FinetuningModel", back_populates="training_sessions"
    )
    training_parameter = relationship(
        "TrainingParameter", uselist=False, backref="training_session"
    )

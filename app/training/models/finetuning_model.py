from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from core.db import Base


class FinetuningModel(Base):
    __tablename__ = "finetuning_model"
    fm_no = Column(
        Integer, primary_key=True, autoincrement=True, comment="AUTO_INCREMENT"
    )
    pm_no = Column(Integer, ForeignKey("pretrained_model.pm_no"), nullable=False)
    user_no = Column(Integer, ForeignKey("users.user_no"), nullable=False)
    fm_description = Column(
        String(255), comment="해당 파인튜닝 모델에 대한 설명(*디자인에 반영안됨, 우선순위 낮음)"
    )

    training_sessions = relationship(
        "TrainingSession", back_populates="finetuning_model"
    )

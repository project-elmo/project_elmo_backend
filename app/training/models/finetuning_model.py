from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from core.db import Base


class FinetuningModel(Base):
    __tablename__ = "finetuning_model"
    fm_no = Column(
        Integer, primary_key=True, autoincrement=True, comment="AUTO_INCREMENT"
    )
    user_no = Column(Integer, ForeignKey("users.user_no"), nullable=False)
    pm_no = Column(Integer, ForeignKey("pretrained_model.pm_no"), nullable=False)
    fm_name = Column(String(50), nullable=False, comment="파인튜닝 모델의 이름")
    fm_description = Column(
        String(255),
        default="no description",
        comment="해당 파인튜닝 모델에 대한 설명(*디자인에 반영안됨, 우선순위 낮음)",
    )
    task = Column(
        Integer, nullable=False, comment="모델의 목적:: 0 QA 1 Classification 2 Generate"
    )

    training_sessions = relationship(
        "TrainingSession", back_populates="finetuning_model"
    )

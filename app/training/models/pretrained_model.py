from sqlalchemy import (
    Column,
    Integer,
    String,
)
from sqlalchemy.orm import relationship
from typing import TypedDict
from app.training.models.finetuning_model import FinetuningModel
from core.db import Base


class PretrainedModel(Base):
    __tablename__ = "pretrained_model"
    pm_no = Column(
        Integer, primary_key=True, autoincrement=True, comment="AUTO_INCREMENT"
    )
    name = Column(String(50), nullable=False)
    description = Column(String(1000), nullable=False)
    version = Column(String(50), nullable=False)
    base_model = Column(String(255), nullable=False, comment="해당 모델의 베이스모델 명")
    dl_url = Column(String(255), nullable=False)
    dl_mirror = Column(String(255))
    parameters = Column(
        String(1000),
        comment='훈련을 위한 파라미터와 기본값. json 형식의 긴 텍스트로 저장. 예: {"batch_size": 32, "learning_rate": 1e2222, ...}',
    )

    finetuning_models = relationship(
        "FinetuningModel", back_populates="pretrained_model"
    )


class ModelsResponse(TypedDict):
    finetuning_model: FinetuningModel
    pretrained_model: PretrainedModel

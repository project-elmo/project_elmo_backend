from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
)
from core.db import Base


class TrainingParameter(Base):
    __tablename__ = "training_parameter"
    parameter_no = Column(
        Integer, primary_key=True, autoincrement=True, comment="AUTO_INCREMENT"
    )
    session_no = Column(
        Integer, ForeignKey("training_session.session_no"), nullable=False
    )
    fm_no = Column(
        Integer,
        ForeignKey("finetuning_model.fm_no"),
        primary_key=True,
        comment="AUTO_INCREMENT",
    )
    parameters = Column(
        String(1000),
        nullable=False,
        comment='json 형식의 긴 텍스트로 저장. 예: {"batch_size": 32, "learning_rate": 1e2222, ...}',
    )
    dataset = Column(String(255), nullable=False, comment="훈련에 쓰인 dataset file path")

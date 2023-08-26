from sqlalchemy import (
    Boolean,
    Column,
    Float,
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
    model_name = Column(String(255), nullable=False)
    epochs = Column(Integer, nullable=False)
    save_strategy = Column(String(50), nullable=False)
    logging_strategy = Column(String(50), nullable=False)
    evaluation_strategy = Column(String(50), nullable=False)
    learning_rate = Column(Float, nullable=False)
    weight_decay = Column(Float, nullable=False)
    batch_size = Column(Integer, nullable=False)
    eval_steps = Column(Integer, nullable=False)
    save_steps = Column(Integer, nullable=False)
    save_total_limits = Column(
        Integer, nullable=False
    )  # 'unlimited' is represented as -1
    run_on_gpu = Column(Boolean, nullable=False)
    load_best_at_the_end = Column(Boolean, nullable=False)
    dataset = Column(String(255), nullable=False, comment="훈련에 쓰인 dataset file path")

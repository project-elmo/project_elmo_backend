from typing import Union
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    String,
    ForeignKey,
)
from app.training.schemas.training import (
    FinetuningRequestSchema,
    TrainingSessionRequestSchema,
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
    fm_no = Column(Integer, ForeignKey("finetuning_model.fm_no"), nullable=False)
    model_name = Column(String(255), nullable=False)
    task = Column(String(255), nullable=False)
    epochs = Column(Integer, nullable=False)
    save_strategy = Column(String(50), nullable=False)
    logging_strategy = Column(String(50), nullable=False)
    evaluation_strategy = Column(String(50), nullable=False)
    train_loss = Column(Float, nullable=False)
    learning_rate = Column(Float, nullable=False)
    weight_decay = Column(Float, nullable=False)
    batch_size = Column(Integer, nullable=False)
    eval_steps = Column(Integer, nullable=False)
    save_steps = Column(Integer, nullable=False)
    save_total_limits = Column(
        Integer, nullable=False
    )  # 'unlimited' is represented as -1
    max_length = Column(Integer, default=512, nullable=False)
    load_best_at_the_end = Column(Boolean, nullable=False)
    dataset = Column(String(255), nullable=False, comment="훈련에 쓰인 dataset file path")

    @classmethod
    def from_schema(
        cls,
        schema: Union[FinetuningRequestSchema, TrainingSessionRequestSchema],
        session_no: int,
        fm_no: int,
        model_name: str,
        train_loss: float,
    ):
        return cls(
            session_no=session_no,
            fm_no=fm_no,
            epochs=schema.epochs,
            save_strategy=schema.save_strategy,
            task=schema.task,
            train_loss=train_loss,
            logging_strategy=schema.logging_strategy,
            evaluation_strategy=schema.evaluation_strategy,
            learning_rate=schema.learning_rate,
            weight_decay=schema.weight_decay,
            batch_size=schema.batch_size,
            eval_steps=schema.eval_steps,
            save_steps=schema.save_steps,
            save_total_limits=schema.save_total_limits,
            max_length=schema.max_length,
            load_best_at_the_end=schema.load_best_at_the_end,
            dataset=schema.dataset,
            model_name=model_name,
        )

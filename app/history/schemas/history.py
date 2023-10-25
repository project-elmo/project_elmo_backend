from datetime import datetime
from pydantic import BaseModel, ConfigDict, root_validator
from typing import List


class FinetuningModelResponseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    fm_no: int = 5
    fm_name: str
    user_no: int
    pm_no: int
    pm_name: str
    fm_description: str


class TrainingSessionResponseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    session_no: str = "3"
    fm_no: int
    fm_name: str
    pm_no: int
    pm_name: str
    parent_session_no: str
    start_time: datetime
    end_time: datetime
    ts_model_name: str


class TrainingParameterResponseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    parameter_no: int = 1
    session_no: int = 3
    fm_no: int = 5
    model_name: str
    task: int
    train_loss: float
    epochs: int
    save_strategy: str
    logging_strategy: str
    evaluation_strategy: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    eval_steps: int
    save_steps: int
    save_total_limits: int
    load_best_at_the_end: bool
    dataset: str
    dataset_download_link: str = None

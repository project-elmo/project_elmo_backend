from pydantic import BaseModel, ConfigDict
from typing import List


class PretrainedModelResponseSchema(BaseModel):
    pm_no: int
    name: str
    description: str
    version: str
    base_model: str


class FinetuningModelResponseSchema(BaseModel):
    fm_no: int
    user_no: int
    pm_no: int
    ft_model_name: str
    fm_description: str


class TrainingSessionResponseSchema(BaseModel):
    session_no: int
    fm_no: int
    parent_session_no: int
    start_time: str
    end_time: str
    ts_model_name: str


class TrainingParameterResponseSchema(BaseModel):
    parameter_no: int
    session_no: int
    fm_no: int
    model_name: str
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
    run_on_gpu: bool
    load_best_at_the_end: bool
    dataset: str

from pydantic import BaseModel, ConfigDict
from typing import List


class PretrainedModelResponseSchema(BaseModel):
    pm_no: int
    name: str
    description: str
    ft_model_name: str
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


class ProgressResponseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_"))

    task: str  # [downloading, training, None]
    model_name: str
    total: str  # 100 for 'training', file size for 'downloading' eg. 100M or 1GB
    curr_size: str  # always 0 for 'training', current file size for 'downloading'
    curr_percent: int
    start_time: str
    end_time: str
    sec_per_dl: str

    @classmethod
    def no_progress(cls):
        return cls(
            task="None",
            model_name="",
            total="0M",
            curr_size="0M",
            curr_percent=0,
            start_time="00:00",
            end_time="00:00",
            sec_per_dl="0.0MB/s",
        )


class TrainingParameterRequestSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_", "elapsed_", "e"))
    model_name: str
    base_model_name: str
    epochs: int = 3
    save_strategy: str = "steps"
    logging_strategy: str = "steps"
    evaluation_strategy: str = "no"
    learning_rate: float = 5.00e-05
    weight_decay: float = 0.0
    batch_size: int = 8
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limits: int = -1  # "unlimited" is represented as -1
    run_on_gpu: bool = True
    load_best_at_the_end: bool = False
    dataset: str

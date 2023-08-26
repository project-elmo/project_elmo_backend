from pydantic import BaseModel, ConfigDict


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


class FinetuningRequestSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=("model_", "elapsed_", "e"))
    pm_no: int = 1
    fm_name: str = "gpt2_chat"
    base_model_name: str = "gpt2"
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
    dataset: str = ""

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

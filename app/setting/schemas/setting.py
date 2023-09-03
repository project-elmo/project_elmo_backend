from pydantic import BaseModel
from core.config import config


class SettingSchema(BaseModel):
    model_path: str = config.DL_DIR
    result_path: str = config.RESULT_DIR
    is_gpu: bool = False

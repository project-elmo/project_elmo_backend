from pydantic import BaseModel
from core.config import config
import torch

is_gpu = torch.cuda.is_available()


class SettingSchema(BaseModel):
    model_path: str = config.DL_DIR
    result_path: str = config.RESULT_DIR
    is_gpu: bool = is_gpu

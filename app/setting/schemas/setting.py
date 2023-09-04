from pydantic import BaseModel
from core.config import config
import torch

is_gpu_available = torch.cuda.is_available()


class SettingSchema(BaseModel):
    model_path: str = config.DL_DIR
    result_path: str = config.RESULT_DIR
    is_gpu_use: bool = is_gpu_available
    is_gpu_available: bool = is_gpu_available

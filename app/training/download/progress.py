from typing import Union
from app.training.schemas.training import LoggingResponseSchema, ProgressResponseSchema
from core.helpers.cache import *


def reset_progress(total: str, model_name: str):
    Cache.delete_startswith(f"{model_name}_{total}")


def update_progress(progress_data: ProgressResponseSchema):
    Cache.set(
        f"{progress_data.model_name}_{progress_data.task}",
        progress_data.model_dump_json(),
    )

def update_log(logging_data: LoggingResponseSchema):
    Cache.set(
        f"{logging_data.model_name}_{logging_data.task}_log",
        logging_data.model_dump_json(),
    )

def set_result(repo_id: str, result: str):
    Cache.set(TASK_RESULT, repo_id)
    Cache.set(REPO_RESULT.format(repo_id), result)

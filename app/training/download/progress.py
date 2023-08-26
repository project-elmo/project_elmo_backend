from app.training.schemas.training import ProgressResponseSchema
from core.helpers.cache import Cache


def reset_progress(total: str, model_name: str):
    Cache.delete_startswith(f"{model_name}_{total}")


def update_progress(progress_data: ProgressResponseSchema):
    Cache.set(
        f"{progress_data.model_name}_{progress_data.total}",
        progress_data.model_dump_json(),
    )

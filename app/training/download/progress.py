# Global variable to save progress for multiple models
PROGRESS = {}


def reset_progress(model_name: str):
    if model_name in PROGRESS:
        del PROGRESS[model_name]


def initialize_progress(model_name: str, total: int):
    PROGRESS[model_name] = {"current": 0, "total": total}


def update_progress(model_name: str, chunk_length: int):
    if model_name in PROGRESS:
        PROGRESS[model_name]["current"] += chunk_length

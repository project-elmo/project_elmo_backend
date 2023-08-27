import os
import asyncio
from datetime import datetime
from typing import Optional
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    WebSocket,
    Response,
    WebSocketDisconnect,
)
from fastapi.websockets import WebSocketState
from app.training.llm.model_trainer import train_model
from app.training.download import hub_download
from app.training.schemas.training import *
from app.training.services.training import TrainingService
from core.config import config
from core.fastapi.dependencies import PermissionDependency, AllowAll
from core.helpers.cache import Cache
from core.utils.file_util import *

training_router = APIRouter()

TASK_PREFIX = "task_"
TRAINING = "training"
DOWNLOADING = "downloading"
SOCKET_CLOSE = "socket_close"


@training_router.get(
    "/hub_download/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def start_hub_download(background_tasks: BackgroundTasks, model_name: str):
    """Initiates a background download task."""
    task_key = f"{TASK_PREFIX}{DOWNLOADING}"
    background_tasks.add_task(Cache.set, task_key, model_name)
    background_tasks.add_task(hub_download, model_name)
    background_tasks.add_task(Cache.delete_startswith, task_key)

    return Response(status_code=200, content="Download in progress!")


@training_router.post("/training/train_pretrained_model")
async def start_training(training_param: FinetuningRequestSchema):
    """Initiates a background model training task."""
    task_key = f"{TASK_PREFIX}{TRAINING}"

    if is_valid_file_path(training_param.dataset) and copy_file(
        training_param.dataset, config.DATASET_DIR
    ):
        filename = get_filename_from_path(training_param.dataset)
        training_param.dataset = os.path.join(config.DATA_DIR, filename)

        Cache.set(task_key, training_param.pm_name)
        train_model(training_param)
        Cache.delete_startswith, task_key

    return Response(status_code=200, content="Training started in the background")


@training_router.websocket("/ws/progress/")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("websocket_log_cache::", Cache.get_all())

    try:
        tasks = [DOWNLOADING, TRAINING]
        while True:
            for task in tasks:
                task_key = f"{TASK_PREFIX}{task}"

                model_name = Cache.get(task_key)

                if model_name:
                    progress_data: ProgressResponseSchema = Cache.get_startswith(
                        model_name
                    )
                    await ws.send_json(progress_data)

            await asyncio.sleep(1)  # send updates every second

    except WebSocketDisconnect:
        # client diconnected
        pass
    finally:
        if ws.client_state != WebSocketState.DISCONNECTED:
            await ws.close()

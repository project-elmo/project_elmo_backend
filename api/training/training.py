import asyncio
from typing import Optional
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    WebSocket,
    Response,
)
from app.training.llm.model_trainer import train_model
from app.training.download import hub_download
from app.training.schemas.training import *
from core.fastapi.dependencies import PermissionDependency, AllowAll
from core.helpers.cache import Cache

training_router = APIRouter()

TASK_PREFIX = "task_"
TRAINING = "training"
DOWNLOADING = "downloading"


@training_router.get(
    "/hub_download/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def start_hub_download(
    background_tasks: BackgroundTasks, model_name: str, namespace: Optional[str] = None
):
    """Initiates a background download task."""
    task_key = f"{TASK_PREFIX}{DOWNLOADING}"
    background_tasks.add_task(Cache.set, task_key, model_name)
    background_tasks.add_task(hub_download, model_name, namespace)
    background_tasks.add_task(Cache.delete_startswith, task_key)

    return Response(status_code=200, content="Download in progress!")


@training_router.post("/training/pretrained_model")
async def start_training(
    training_param: TrainingParameterRequestSchema, background_tasks: BackgroundTasks
):
    """Initiates a background model training task."""
    task_key = f"{TASK_PREFIX}{TRAINING}"
    background_tasks.add_task(Cache.set, task_key, training_param.base_model_name)
    background_tasks.add_task(train_model, training_param)
    background_tasks.add_task(Cache.delete_startswith, task_key)

    return Response(status_code=200, content="Training started in the background")


@training_router.websocket("/ws/progress/")
async def websocket_endpoint(ws: WebSocket):
    """
    Sends ProgressResponseSchema via socket to provide real-time training and download
    progress to the client.
    """
    await ws.accept()

    try:
        tasks = [DOWNLOADING, TRAINING]
        while True:
            for task in tasks:
                task_key = f"{TASK_PREFIX}{task}"
                model_name = Cache.get(str(task_key))
                print("model_name", model_name)
                print(Cache.get_all())

                if model_name:
                    progress_data: ProgressResponseSchema = Cache.get_startswith(
                        model_name
                    )
                    print("progress_data: ", progress_data)
                    await ws.send_json(progress_data)

            # Close the connection if no tasks are active
            if not any([Cache.get(f"{TASK_PREFIX}{task}") for task in tasks]):
                break

            await asyncio.sleep(1)  # send updates every second

    finally:
        await ws.close()

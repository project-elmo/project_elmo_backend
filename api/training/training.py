import os
import asyncio
from typing import Optional
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    Response,
    WebSocketDisconnect,
    UploadFile,
    File,
)
from fastapi.websockets import WebSocketState
from loguru import logger
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


@training_router.get(
    "/remove_pretrained/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def remove_pretrained(model_name: str):
    transformed_name = transform_model_name(model_name)
    huggingface_dir = get_huggingface_dir()

    file_path = os.path.join(huggingface_dir, transformed_name)

    # Check if the file exists before attempting to delete
    if os.path.exists(file_path):
        os.remove(file_path)
        return {
            "status": "success",
            "message": f"File {transformed_name} deleted successfully.",
        }
    else:
        raise HTTPException(status_code=404, detail="File not found")


@training_router.post("/data_upload/")
async def upload_file(file: UploadFile = File(...)):
    datasets_path = config.DATASET_DIR
    file_location = os.path.join(datasets_path, file.filename)

    with open(file_location, "wb+") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}


@training_router.get("/get_datasets")
async def get_datasets():
    datasets_path = config.DATASET_DIR

    # Check if the datasets directory exists
    if not os.path.isdir(datasets_path):
        raise HTTPException(status_code=404, detail="Datasets directory not found")

    datasets = []
    for filename in os.listdir(datasets_path):
        file_path = os.path.join(datasets_path, filename)
        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(filename)
            data = DatasetResponseSchema(
                file_path=file_path,
                size=os.path.getsize(file_path),
                filename=filename,
                extension=file_extension[1:],
            )
            datasets.append(data)

    return datasets


@training_router.post("/training/train_pretrained_model")
async def start_training(training_param: FinetuningRequestSchema):
    """Initiates a background model training task."""
    task_key = f"{TASK_PREFIX}{TRAINING}"

    file_path = training_param.dataset
    is_file = is_valid_file_path(file_path)
    is_copy_file = copy_file(file_path, config.DATASET_DIR)

    logger.debug(f"{file_path} is_file: {is_file}, is_copy_file: {is_copy_file}")

    if is_file and is_copy_file:
        filename = get_filename_from_path(file_path)
        training_param.dataset = os.path.join(config.DATA_DIR, filename)

        Cache.set(task_key, training_param.pm_name)
        train_model(training_param)
        Cache.delete_startswith, task_key
    else:
        logger.debug("Wrong file path")
        raise HTTPException(status_code=404, detail="File not found")

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

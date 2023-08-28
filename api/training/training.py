import os
import shutil
from typing import List

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
from app.user.schemas import ExceptionResponseSchema
from core.config import config
from core.fastapi.dependencies import PermissionDependency, AllowAll
from core.helpers.cache import Cache
from core.utils.file_util import *

training_router = APIRouter()

TASK_PREFIX = "task_"
TRAINING = "training"
DOWNLOADING = "downloading"
RESULT = "result"
SOCKET_CLOSE = "socket_close"
tasks = [DOWNLOADING, TRAINING, RESULT]


@training_router.get(
    "/hub_download/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def start_hub_download(background_tasks: BackgroundTasks, model_name: str):
    """Initiates a background download task."""
    task_key = f"{TASK_PREFIX}{DOWNLOADING}"
    background_tasks.add_task(Cache.set, task_key, model_name)
    background_tasks.add_task(hub_download, model_name)
    background_tasks.add_task(Cache.delete, task_key)
    background_tasks.add_task(Cache.delete, model_name)

    return Response(status_code=200, content="Download in progress!")


@training_router.get(
    "/remove_pretrained/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def remove_pretrained(model_name: str):
    Cache.delete(f"{model_name}_{DOWNLOADING}")

    transformed_name = transform_model_name(model_name)
    huggingface_dir = get_huggingface_dir()

    dir_path = os.path.join(huggingface_dir, transformed_name)

    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

        return {
            "status": "success",
            "message": f"Directory {transformed_name} deleted successfully.",
        }
    else:
        raise HTTPException(status_code=404, detail="Directory not found")


@training_router.get(
    "/pretrained_models/",
    response_model=List[PretrainedModelResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def list_all_pretrained_models():
    """Retrieve a list of all pre-trained models."""
    models_db = await TrainingService().get_all_pretrained_models()
    huggingface_dir = get_huggingface_dir()

    # Convert ORM objects to Pydantic objects and set is_downloaded
    schema_models = []
    for db_model in models_db:
        transformed_name = transform_model_name(db_model.name)
        model_dir = os.path.join(huggingface_dir, transformed_name)
        logger.debug(f"model_dir={model_dir}")

        is_downloaded = get_is_downloaded(model_dir)
        model_data = PretrainedModelResponseSchema(
            pm_no=db_model.pm_no,
            name=db_model.name,
            description=db_model.description,
            version=db_model.version,
            base_model=db_model.base_model,
            is_downloaded=is_downloaded,
        )
        schema_models.append(model_data)

    return schema_models


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
        if filename == ".gitkeep":
            continue

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


@training_router.post(
    "/training/train_pretrained_model",
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def start_training(training_param: FinetuningRequestSchema):
    """Initiates a background model training task."""
    task_key = f"{TASK_PREFIX}{TRAINING}"

    Cache.set(task_key, training_param.pm_name)
    await train_model(training_param)
    Cache.delete(task_key)
    Cache.delete(f"{training_param.pm_name}_{TRAINING}")

    return Response(status_code=200, content="Training started in the background")


async def send_progress(ws: WebSocket):
    while True:
        for task in tasks:
            task_key = f"{TASK_PREFIX}{task}"

            model_name = Cache.get(task_key)
            key = f"{model_name}_{task}"

            if task == RESULT:
                if model_name:
                    result = Cache.get(key)
                    await ws.send_json(result)
                    Cache.delete(key)
            else:
                if model_name:
                    progress_data: ProgressResponseSchema = Cache.get(key)

                    if progress_data:
                        await ws.send_json(progress_data)
                        Cache.delete(key)

        await asyncio.sleep(0.5)  # send updates every half-second


async def receive_commands(ws: WebSocket):
    while True:
        message = await ws.receive_text()
        if message == "stop_training":
            Cache.set("training_should_continue", "False")


@training_router.websocket("/ws/progress/")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("websocket_log_cache::", Cache.get_all())

    try:
        await asyncio.gather(send_progress(ws), receive_commands(ws))
    except WebSocketDisconnect:
        # client disconnected
        pass
    finally:
        if ws.client_state != WebSocketState.DISCONNECTED:
            await ws.close()

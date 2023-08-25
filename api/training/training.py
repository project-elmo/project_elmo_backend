import asyncio
from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    Query,
    BackgroundTasks,
    Response,
    WebSocket,
)
from app.training.llm.model_trainer import train_model
from app.training.download import download, hub_download
from app.training.models.training_parameter import TrainingParameter
from core.fastapi.dependencies import PermissionDependency, AllowAll
from core.helpers.cache import Cache

training_router = APIRouter()

TASK_PREFIX = "task_"


@training_router.get(
    "/download/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def start_download(model_name: str, namespace: str = None):
    task_key = f"{TASK_PREFIX}downloading"
    await Cache.backend.set(task_key, model_name)
    await download(model_name, namespace)
    await Cache.remove_by_prefix(task_key)
    return Response(status_code=200, content="Download complete!")


@training_router.get(
    "/hub_download/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def start_hub_download(model_name: str, namespace: str = None):
    task_key = f"{TASK_PREFIX}downloading"
    await Cache.backend.set(task_key, model_name)
    await hub_download(model_name, namespace)
    await Cache.remove_by_prefix(task_key)
    return Response(status_code=200, content="Download complete!")


@training_router.get(
    "/download/progress/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def get_download_progress(model_name: str = Query(None)):
    return await Cache.backend.get(model_name) or {"current": 0, "total": 0}


# @training_router.post("/training/pretrained_model")
# async def start_training(
#     training_param: TrainingParameter, background_tasks: BackgroundTasks
# ):
#     task_key = f"{TASK_PREFIX}training"
#     await Cache.backend.set(task_key, training_param.model_name)
#     background_tasks.add_task(train_model, training_param)
#     await Cache.remove_by_prefix(task_key)
#     return {"message": "Training started in the background"}


@training_router.websocket("/ws/progress/")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        tasks = ["downloading", "training"]
        while True:
            for task in tasks:
                task_key = f"{TASK_PREFIX}{task}"
                name = await Cache.backend.get(task_key)

                if name:
                    progress_data = await Cache.backend.get(name) or {
                        "current": 0,
                        "total": 0,
                    }
                    message = {
                        "task": task,
                        "name": name,
                        "current": progress_data["current"],
                        "total": progress_data["total"],
                    }
                    await ws.send_json(message)

            if not any(
                [await Cache.backend.get(f"{TASK_PREFIX}{task}") for task in tasks]
            ):
                # Close the connection if no tasks are active
                break

            await asyncio.sleep(1)  # send updates every second

    except:
        pass
    finally:
        await ws.close()

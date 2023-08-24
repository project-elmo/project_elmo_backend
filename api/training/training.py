from app.training.download import download
from app.training.download.progress import PROGRESS
from core.fastapi.dependencies import PermissionDependency, AllowAll
from fastapi import APIRouter, Response, Depends, HTTPException, status, Query

training_router = APIRouter()


@training_router.get(
    "/download/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def start_download(model_name: str = Query(None), namespace: str | None = None):
    download(model_name, namespace)
    return Response(status_code=200, content="Download complete!")


@training_router.get(
    "/download/progress/", dependencies=[Depends(PermissionDependency([AllowAll]))]
)
async def get_download_progress(model_name: str = Query(None)):
    return PROGRESS.get(model_name, {"current": 0, "total": 0})

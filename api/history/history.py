from typing import List
from fastapi import (
    APIRouter,
    HTTPException,
)
from core.config import config
from core.utils.file_util import get_is_downloaded

from app.history.schemas.history import *
from app.user.schemas import ExceptionResponseSchema
from app.training.services.training import TrainingService

history_router = APIRouter()


@history_router.get(
    "/pretrained_models/",
    response_model=List[PretrainedModelResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def list_all_pretrained_models():
    """Retrieve a list of all pre-trained models."""
    models_db = await TrainingService().get_all_pretrained_models()
    models_dir = config.MODELS_DIR

    # Convert ORM objects to Pydantic objects and set is_downloaded
    schema_models = []
    for db_model in models_db:
        is_downloaded = get_is_downloaded(models_dir, db_model.name)
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


@history_router.get(
    "/finetuned_models/",
    response_model=List[FinetuningModelResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def list_all_finetuned_models():
    """Retrieve a list of all fine-tuned models."""
    models = await TrainingService().get_all_finetuned_models()
    return models


@history_router.get(
    "/training_sessions/{fm_no}/",
    response_model=List[TrainingSessionResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def list_training_sessions_by_fm(fm_no: int):
    """Retrieve a list of training sessions by their fm_no."""
    sessions = await TrainingService().get_training_sessions_by_fm(fm_no=fm_no)

    # Convert the integer fields to strings
    for session in sessions:
        session.session_no = str(session.session_no)
        if session.parent_session_no == 0:
            session.parent_session_no = ""

    return sessions


@history_router.get(
    "/training_parameters/{session_no}/",
    response_model=TrainingParameterResponseSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_training_parameter_by_session_no(session_no: int):
    """Retrieve training parameters by session number."""
    training_param = await TrainingService().get_training_parameter_by_session(
        session_no=session_no
    )

    if not training_param:
        raise HTTPException(status_code=404, detail="Training parameter not found")

    return training_param

from typing import List
from fastapi import (
    APIRouter,
    HTTPException,
)

from app.history.schemas.history import *
from app.user.schemas import ExceptionResponseSchema
from app.training.services.training import TrainingService

history_router = APIRouter()


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
        session.parent_session_no = str(session.parent_session_no)

        if session.parent_session_no == "0":
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

from typing import List
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.training.models.pretrained_model import PretrainedModel
from app.training.schemas.training import (
    FinetuningRequestSchema,
    TrainingSessionRequestSchema,
)
from app.training.models import FinetuningModel, TrainingSession, TrainingParameter

from core.db import session


class InferenceService:
    async def get_session_by_session_no(self, session_no: int) -> TrainingSession:
        query = (
            select(TrainingSession)
            .options(
                joinedload(TrainingSession.finetuning_model).joinedload(
                    FinetuningModel.pretrained_model
                )
            )
            .filter(TrainingSession.session_no == session_no)
        )
        result = await session.execute(query)
        return result.scalar()

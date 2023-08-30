from typing import List
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.training.models.pretrained_model import PretrainedModel
from app.training.schemas.training import FinetuningRequestSchema, TrainingSessionRequestSchema
from app.training.models import FinetuningModel, TrainingSession, TrainingParameter

from core.db import session


class HistoryService:
    async def get_all_pretrained_models(self) -> List[PretrainedModel]:
        query = select(PretrainedModel)
        result = await session.execute(query)
        return result.scalars().all()

    async def get_all_finetuned_models(self) -> List[FinetuningModel]:
        query = select(FinetuningModel).options(joinedload(FinetuningModel.pretrained_model))
        result = await session.execute(query)
        return result.scalars().all()

    async def get_training_sessions_by_fm(self, fm_no: int) -> List[TrainingSession]:
        query = (
            select(TrainingSession)
            .options(
                joinedload(TrainingSession.finetuning_model).joinedload(FinetuningModel.pretrained_model)
            )
            .filter(TrainingSession.fm_no == fm_no)
        )
        result = await session.execute(query)
        return result.scalars().all()

    async def get_training_parameter_by_session(
        self, session_no: int
    ) -> TrainingParameter:
        query = (
            select(TrainingParameter)
            .join(TrainingSession)
            .where(TrainingSession.session_no == session_no)
        )
        result = await session.execute(query)
        return result.scalars().first()
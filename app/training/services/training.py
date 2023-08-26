from typing import List
from sqlalchemy import select
from app.training.models.pretrained_model import PretrainedModel
from core.db import session
from app.training.models import FinetuningModel, TrainingSession, TrainingParameter


class TrainingService:
    async def create_finetuning_model(
        self,
        training_param: TrainingParameter,
        fm_name: str,
        pm_no: int,
        start_time: str,  # YYYY-MM-DD HH:MM:SS
        end_time: str,  # YYYY-MM-DD HH:MM:SS
        ts_model_name: str,
        user_no: int = 1,  # TODO: fix,
        parent_session_no: int = 1,
    ) -> None:
        # Create the fine-tuned model
        ft_model = FinetuningModel(
            user_no=user_no,
            pm_no=pm_no,
            fm_name=fm_name,
        )

        # Create the training session
        training_session = TrainingSession(
            parent_session_no=parent_session_no,
            start_time=start_time,
            end_time=end_time,
            ts_model_name=ts_model_name,
        )

        training_session.finetuning_model = ft_model

        # Create the training parameter
        training_session.training_parameter = training_param

        try:
            session.add(ft_model)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e

    async def get_all_pretrained_models(self) -> List[PretrainedModel]:
        query = select(PretrainedModel)
        result = await session.execute(query)
        return result.scalars().all()

    async def get_all_finetuned_models(self) -> List[FinetuningModel]:
        query = select(FinetuningModel)
        result = await session.execute(query)
        return result.scalars().all()

    async def get_training_sessions_by_fm(self, fm_no: int) -> List[TrainingSession]:
        query = select(TrainingSession).where(TrainingSession.fm_no == fm_no)
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

from typing import List
from loguru import logger
from sqlalchemy import select

from app.training.models.pretrained_model import PretrainedModel
from app.training.schemas.training import FinetuningRequestSchema, TrainingSessionRequestSchema
from app.training.models import FinetuningModel, TrainingSession, TrainingParameter

from core.db import session


class TrainingService:
    async def create_finetuning_model(
        self,
        training_param: FinetuningRequestSchema,
        start_time: str,  # 2023-08-26T21:04:31
        end_time: str,  # 2023-08-26T21:04:31
        uuid: str,
        user_no: int = 1,  # TODO: fix,
        parent_session_no: int = 0,  # This will convert into "". The value of the root node for sessions should be an empty string.
    ) -> None:
        
        try:
            # Create the fine-tuned model
            ft_model = FinetuningModel(
                user_no=user_no,
                pm_no=training_param.pm_no,
                fm_name=training_param.fm_name,
                task=training_param.task,
            )

            # Create the training session
            training_session = TrainingSession(
                parent_session_no=parent_session_no,
                start_time=start_time,
                end_time=end_time,
                ts_model_name=training_param.ts_model_name,
                uuid = uuid,
                finetuning_model = ft_model,
            )
            
            session.add(ft_model)
            await session.flush() 
            
            # Create the training parameter
            training_parameter = TrainingParameter.from_schema(
                training_param, 
                session_no=training_session.session_no, 
                fm_no=ft_model.fm_no
            )
            training_session.training_parameter = training_parameter
            
            await session.commit()

        except Exception as e:
            await session.rollback()
            logger.error(f"Error while inserting: {e}")
            raise e

    async def create_training_session(
        self,
        training_param: TrainingSessionRequestSchema,
        start_time: str,
        end_time: str,
        uuid: str,
        user_no: int = 1,  # TODO: fix,
    ) -> None:
        try:
            # Create the training session
            training_session = TrainingSession(
                fm_no=training_param.fm_no,
                parent_session_no=training_param.parent_session_no,
                start_time=start_time,
                end_time=end_time,
                ts_model_name=training_param.ts_model_name,
                uuid = uuid,
            )
            
            # Create the training parameter
            training_parameter = TrainingParameter.from_schema(
                training_param, 
                session_no=training_session.session_no, 
                fm_no=training_param.fm_no,
            )
            training_session.training_parameter = training_parameter
            
            await session.commit()

        except Exception as e:
            await session.rollback()
            logger.error(f"Error while inserting: {e}")
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

    async def get_uuid_by_session_no(self, session_no: int) -> str:
        query = select(TrainingSession.uuid).where(TrainingSession.session_no == session_no)
        result = await session.execute(query)
        record = result.scalar()
        return record
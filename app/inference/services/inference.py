from typing import List
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from app.inference.models.message import Message
from app.inference.models.test import Test
from app.inference.schemas.inference import MessageRequestSchema, TestResponseSchema

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

    async def get_fm_no_by_session_no(self, session_no: int) -> int:
        query = select(TrainingSession.fm_no).where(
            TrainingSession.session_no == session_no
        )
        result = await session.execute(query)
        record = result.scalar()
        return record

    async def get_test_no_by_session_no(self, session_no: int) -> int:
        query = select(Test.test_no).where(Test.session_no == session_no)
        result = await session.execute(query)
        record = result.scalar()
        return record

    async def create_test(
        self,
        session_no: int,
    ) -> Test:
        try:
            fm_no = await self.get_fm_no_by_session_no(self, session_no)

            # Create the training session
            test = Test(
                session_no=session_no,
                fm_no=fm_no,
            )

            session.add(test)
            await session.commit()
            await session.refresh(test)

            return test

        except Exception as e:
            await session.rollback()
            logger.error(f"Error while inserting: {e}")
            raise e

    async def create_messages(
        self,
        test_request: MessageRequestSchema,
        response: TestResponseSchema,
    ) -> None:
        try:
            test_no = await self.get_test_no_by_session_no(
                self, test_request.session_no
            )

            # Create message for prompt
            prompt = Message(
                msg=test_request.prompt,
                is_user=1,
                test_no=test_no,
            )
            session.add(prompt)

            # Create message for response
            response = Message(
                msg=response.response,
                is_user=0,
                test_no=test_no,
            )
            session.add(response)

            await session.commit()

        except Exception as e:
            await session.rollback()
            logger.error(f"Error while inserting: {e}")
            raise e

    async def get_all_test(self) -> List[Test]:
        query = select(Test).options(joinedload(Test.training_session))
        result = await session.execute(query)
        return result.scalars().all()

    async def get_chat_history(self, test_no) -> List[Message]:
        query = (
            select(Message)
            .options(joinedload(Message.test))
            .filter(Message.test_no == test_no)
        )
        result = await session.execute(query)
        return result.scalars().all()

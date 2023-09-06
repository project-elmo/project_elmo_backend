from typing import List
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from app.inference.models.message import Message
from app.inference.models.test import Test
from app.inference.schemas.inference import (
    MessageRequestSchema,
    MessageResponseSchema,
)

from app.training.models import FinetuningModel, TrainingSession
from core.db import session


class InferenceService:
    async def get_session_by_test_no(self, test_no: int) -> TrainingSession:
        query = (
            select(TrainingSession)
            .join(Test, Test.session_no == TrainingSession.session_no)
            .options(
                joinedload(TrainingSession.finetuning_model).joinedload(
                    FinetuningModel.pretrained_model
                )
            )
            .filter(Test.test_no == test_no)
        )
        result = await session.execute(query)
        return result.scalar()

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

    async def get_fm_by_session_no(self, session_no: int) -> int:
        query = select(TrainingSession).where(TrainingSession.session_no == session_no)
        result = await session.execute(query)
        record = result.scalar()
        return record

    async def get_all_fm_with_test(self) -> List[FinetuningModel]:
        query = select(FinetuningModel).options(
            joinedload(FinetuningModel.pretrained_model),
            joinedload(FinetuningModel.training_sessions).joinedload(
                TrainingSession.tests
            ),
        )
        result = await session.execute(query)
        return result.scalars().unique().all()

    async def get_all_test(self) -> List[Test]:
        query = select(Test).options(
            joinedload(Test.training_session).joinedload(
                TrainingSession.finetuning_model
            )
        )
        result = await session.execute(query)
        return result.scalars().all()

    async def get_test_no_by_session_no(self, session_no: int) -> int:
        query = select(Test.test_no).where(Test.session_no == session_no)
        result = await session.execute(query)
        record = result.scalar()
        return record

    async def create_test(
        self,
        session_no: int,
    ) -> Test:
        """
        Create a test by session number if it does not exist; 
        otherwise, return the existing test
        """
        try:
            query = (
                select(Test)
                .options(
                    joinedload(Test.training_session).joinedload(
                        TrainingSession.finetuning_model
                    )
                )
                .filter(Test.session_no == session_no)
            )            
            result = await session.execute(query)
            existing_test:Test = result.scalar()

            if existing_test:
                return existing_test

            fm: FinetuningModel = await self.get_fm_by_session_no(session_no)
            test = Test(
                session_no=session_no,
                fm_no=fm.fm_no,
            )

            session.add(test)
            await session.commit()
            await session.refresh(test)

            query = (
                select(Test)
                .options(
                    joinedload(Test.training_session).joinedload(
                        TrainingSession.finetuning_model
                    )
                )
                .filter(Test.test_no == test.test_no)
            )
            result = await session.execute(query)
            return result.scalar()

        except Exception as e:
            await session.rollback()
            logger.error(f"Error while inserting: {e}")
            raise e

    async def create_messages(
        self,
        test_request: MessageRequestSchema,
        response: MessageResponseSchema,
    ) -> List[Message]:
        try:
            # Create message for prompt
            prompt = Message(
                msg=test_request.msg,
                is_user=1,
                test_no=test_request.test_no,
            )
            session.add(prompt)

            # Create message for response
            response = Message(
                msg=response,
                is_user=0,
                test_no=test_request.test_no,
            )
            session.add(response)

            await session.commit()
            await session.refresh(prompt)
            await session.refresh(response)

            return [prompt, response]

        except Exception as e:
            await session.rollback()
            logger.error(f"Error while inserting: {e}")
            raise e

    async def get_chat_history(self, test_no) -> List[Message]:
        query = (
            select(Message)
            .options(joinedload(Message.test))
            .filter(Message.test_no == test_no)
        )
        result = await session.execute(query)
        return result.scalars().all()

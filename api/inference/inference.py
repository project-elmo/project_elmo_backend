from typing import List
from fastapi import (
    APIRouter,
)
from loguru import logger
from app.history.schemas.history import (
    TrainingSessionResponseSchema,
)
from app.inference.inference import execute_inference
from app.inference.models.test import Test
from app.inference.schemas.inference import (
    GetTestListResponseSchema,
    MessageResponseSchema,
    MessageRequestSchema,
    TestResponseSchema,
)
from app.inference.services.inference import InferenceService
from app.training.models.finetuning_model import FinetuningModel
from app.training.models.training_session import TrainingSession
from app.user.schemas import ExceptionResponseSchema

test_router = APIRouter()


@test_router.get(
    "/tests",
    response_model=List[GetTestListResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_all_test():
    fm_models: List[FinetuningModel] = await InferenceService().get_all_fm_with_test()

    response_data = []

    for fm in fm_models:
        sessions = []
        tests = []

        training_sessions: List[TrainingSession] = fm.training_sessions

        for ts in training_sessions:
            sessions.append(
                TrainingSessionResponseSchema(
                    session_no=str(ts.session_no),
                    fm_no=fm.fm_no,
                    fm_name=fm.fm_name,
                    pm_no=ts.pm_no,
                    pm_name=ts.pm_name,
                    parent_session_no=str(ts.parent_session_no),
                    start_time=ts.start_time,
                    end_time=ts.end_time,
                    ts_model_name=ts.ts_model_name,
                )
            )

            test: Test = ts.tests
            if test:
                tests.append(
                    TestResponseSchema(
                        test_no=test.test_no,
                        session_no=test.session_no,
                        ts_model_name=ts.ts_model_name,
                        fm_no=test.fm_no,
                        fm_name=test.fm_name,
                    )
                )

        fm_data = GetTestListResponseSchema(
            fm_no=fm.fm_no, fm_name=fm.fm_name, list_sessions=sessions, list_test=tests
        )

        response_data.append(fm_data)

    return response_data


@test_router.post(
    "/create_test",
    response_model=TestResponseSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def create_test(session_no: int):
    """
    Create a test by session number if it does not exist; otherwise, return the existing test.
    """
    test = await InferenceService().create_test(session_no)
    return test


@test_router.post(
    "/create_message",
    response_model=List[MessageResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_msg_resposne(msg_request: MessageRequestSchema):
    """
    Create message by user's input

    Returns:
        MessageResponseSchema: The result of the inference.
    """
    response = await execute_inference(msg_request)
    results = await InferenceService().create_messages(msg_request, response)

    return results


@test_router.get(
    "/chat_history",
    response_model=List[MessageResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_chat_history(test_no: int):
    response = await InferenceService().get_chat_history(test_no)
    return response

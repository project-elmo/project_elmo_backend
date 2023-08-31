from typing import List
from fastapi import (
    APIRouter,
)
from loguru import logger
from app.history.schemas.history import (
    TrainingSessionResponseSchema,
)
from app.inference.inference import execute_inference
from app.inference.schemas.inference import (
    MessageResponseSchema,
    MessageRequestSchema,
    TestResponseSchema,
)
from app.inference.services.inference import InferenceService
from app.user.schemas import ExceptionResponseSchema

test_router = APIRouter()


@test_router.post(
    "/create_test",
    response_model=TestResponseSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def create_test(session_no: int):
    """
    Create a test by session_no
    """
    test = await InferenceService().create_test(session_no)
    return test


@test_router.post(
    "/create_message",
    response_model=MessageResponseSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_test_resposne(test_request: MessageRequestSchema):
    """
    Create message by user's input

    Returns:
        MessageResponseSchema: The result of the infrence.
    """
    response = await execute_inference(test_request)
    await InferenceService().create_messages(test_request, response)

    return MessageResponseSchema(response=response)


@test_router.get(
    "/chat_history",
    response_model=List[MessageResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_chat_history(test_no: int):
    response = await InferenceService().get_chat_history(test_no)
    return response

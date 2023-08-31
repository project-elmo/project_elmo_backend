from fastapi import (
    APIRouter,
)
from loguru import logger
from app.history.schemas.history import (
    TrainingSessionResponseSchema,
)
from app.inference.inference import execute_inference
from app.inference.schemas.inference import TestRequestSchema, TestResponseSchema
from app.user.schemas import ExceptionResponseSchema

test_router = APIRouter()


@test_router.post(
    "/",
    response_model=TestResponseSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_test_resposne(test_request: TestRequestSchema):
    response = await execute_inference(test_request)
    
    return TestResponseSchema(response=response)

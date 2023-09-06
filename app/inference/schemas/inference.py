from typing import List
from app.inference.models.test import Test
from pydantic import BaseModel, ConfigDict
import datetime

from app.history.schemas.history import TrainingSessionResponseSchema


class TestResponseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    test_no: int
    session_no: int
    ts_model_name: str  # 해당 테스트의 부모 세션의 모델 이름
    fm_no: int  # 해당 테스트의 부모 fm_no
    fm_name: str  # 해당 테스트의 부모 fm_name


class GetTestListResponseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fm_no: int
    fm_name: str
    list_sessions: List[TrainingSessionResponseSchema]
    list_test: List[TestResponseSchema]


class MessageRequestSchema(BaseModel):
    test_no: int = 1
    task: int = 0  # 모델의 목적:: 0 QA 1 Classification 2 Generate
    msg: str
    max_length: int = 50  # 최대 길이
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


class MessageResponseSchema(BaseModel):
    msg_no: int
    msg: str
    created_at: datetime.datetime
    is_user: bool
    test_no: int

    class Config:
        from_attributes = True

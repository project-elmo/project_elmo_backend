from pydantic import BaseModel


class TestRequestSchema(BaseModel):
    session_no: int = 1
    task: int = 0  # 모델의 목적:: 0 QA 1 Classification 2 Generate
    prompt: str
    max_length: int = 50  # 최대 길이


class TestResponseSchema(BaseModel):
    response: str

    class Config:
        from_attributes = True

from pydantic import BaseModel


class TestResponseSchema(BaseModel):
    test_no: int
    session_no: int

    class Config:
        from_attributes = True


class MessageRequestSchema(BaseModel):
    session_no: int = 1
    task: int = 0  # 모델의 목적:: 0 QA 1 Classification 2 Generate
    prompt: str
    max_length: int = 50  # 최대 길이


class MessageResponseSchema(BaseModel):
    msg: str

    class Config:
        from_attributes = True

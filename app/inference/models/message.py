import datetime
from sqlalchemy import Column, DateTime, Float, Integer, ForeignKey, String
from sqlalchemy.orm import relationship

from core.db import Base


class Message(Base):
    __tablename__ = "message"

    msg_no = Column(
        Integer, primary_key=True, autoincrement=True, comment="AUTO INCRAMENT"
    )
    msg = Column(String(5000), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    is_user = Column(Integer, comment="1 is user, otherwise model", nullable=False)
    test_no = Column(
        Integer,
        ForeignKey("test.test_no"),
        nullable=False,
        comment="1 for user, otherwise 0",
    )
    max_length = Column(
        Integer,
        default=50,
        nullable=False,
    )
    temperature = Column(
        Float,
        default=1.0,
        nullable=False,
    )
    top_k = Column(
        Integer,
        default=50,
        nullable=False,
    )
    top_p = Column(
        Float,
        default=1.0,
        nullable=False,
    )
    repetition_penalty = Column(
        Float,
        default=1.0,
        nullable=False,
    )
    no_repeat_ngram_size = Column(
        Integer,
        default=0,
        nullable=False,
    )

    test = relationship("Test", back_populates="messages")

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name == "is_user":
            return bool(value)
        return value

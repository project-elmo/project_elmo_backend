import datetime
from sqlalchemy import Column, DateTime, Integer, ForeignKey, String
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
    test_no = Column(Integer, ForeignKey("test.test_no"), nullable=False)

    test = relationship("Test", back_populates="messages")

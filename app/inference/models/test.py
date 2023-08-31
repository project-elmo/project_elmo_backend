from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship

from core.db import Base


class Test(Base):
    __tablename__ = "test"

    test_no = Column(
        Integer, primary_key=True, autoincrement=True, comment="AUTO INCRAMENT"
    )
    session_no = Column(
        Integer, ForeignKey("training_session.session_no"), nullable=False
    )
    fm_no = Column(
        Integer,
        ForeignKey("finetuning_model.fm_no"),
        nullable=False,
        comment="AUTO INCRAMENT",
    )

    training_session = relationship(
        "TrainingSession", uselist=False, back_populates="tests"
    )
    messages = relationship("Message", back_populates="test")

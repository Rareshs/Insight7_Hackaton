from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from datetime import datetime
from .database import Base
from sqlalchemy.orm import relationship

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    messages = Column(JSON, nullable=False)
    ml_score = Column(Float, nullable=True)      
    ml_verdict = Column(String, nullable=True)   
    created_at = Column(DateTime, default=datetime.utcnow)

    ml_steps = relationship("MLStepScore", back_populates="conversation")

class MLStepScore(Base):
    __tablename__ = "ml_step_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    message = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    label = Column(String, nullable=False)

    conversation = relationship("Conversation", back_populates="ml_steps")
    



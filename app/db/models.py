from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    messages = Column(String, nullable=False)
    duration = Column(String, nullable=True) 
    created_at = Column(DateTime, default=datetime.utcnow)

    scores = relationship("Score", back_populates="conversation", cascade="all, delete-orphan")
    flagged_words = relationship("FlaggedWord", back_populates="conversation", cascade="all, delete-orphan")

class Score(Base):
    __tablename__ = "scores"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    risk_score = Column(Float)

    conversation = relationship("Conversation", back_populates="scores")

class FlaggedWord(Base):
    __tablename__ = "flagged_words"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    word = Column(String)

    conversation = relationship("Conversation", back_populates="flagged_words")
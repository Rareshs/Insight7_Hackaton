from typing import List
from pydantic import BaseModel

class Message(BaseModel):
    t: str
    role: str
    text: str

class CallCard(BaseModel):
    conversation_id: str
    risk_score: float
    status: str
    messages: List[Message]

class MLStep(BaseModel):
    message: str
    score: float
    label: str
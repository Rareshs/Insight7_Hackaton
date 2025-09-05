from pydantic import BaseModel
from typing import List

class AnalyzeRequest(BaseModel):
    conversation_id: str
    messages: List[str]

class AnalyzeResponse(BaseModel):
    risk_score: float
    flagged_words: List[str]

class CallCard(BaseModel):
    conversation_id: str
    duration: str
    risk_score: float
    status: str


from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas import AnalyzeRequest, AnalyzeResponse, CallCard
from app.services.analyzer import analyze_messages
from app.db.database import SessionLocal, engine
from app.db import models
from typing import List

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_conversation(req: AnalyzeRequest, db: Session = Depends(get_db)):
    score, flagged = analyze_messages(req.messages)
    full_text = " ".join(req.messages).lower()

    conv = db.get(models.Conversation, req.conversation_id)
    if conv is None:
        conv = models.Conversation(
            id=req.conversation_id,
            messages=full_text,
            duration="02:15" 
        )
        db.add(conv)

    db.add(models.Score(conversation_id=req.conversation_id, risk_score=score))
    for w in flagged:
        db.add(models.FlaggedWord(conversation_id=req.conversation_id, word=w))

    db.commit()
    return AnalyzeResponse(risk_score=score, flagged_words=flagged)


@app.get("/conversations", response_model=List[CallCard])
def list_conversations(db: Session = Depends(get_db)):
    convs = db.query(models.Conversation).all()
    result = []

    for conv in convs:
        score = db.query(models.Score).filter_by(conversation_id=conv.id).first()
        status = (
            "safe" if score.risk_score < 0.4 else
            "suspect" if score.risk_score < 0.7 else
            "ALERT"
        )

        result.append(CallCard(
            conversation_id=conv.id,
            duration=conv.duration or "00:00",
            risk_score=score.risk_score,
            status=status
        ))
    return result

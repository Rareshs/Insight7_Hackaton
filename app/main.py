from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from collections import defaultdict
from datetime import datetime
import os, json
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas import AnalyzeRequest, AnalyzeResponse, CallCard
from app.services.analyzer import analyze_messages
from app.db.database import SessionLocal, engine
from app.db import models
from typing import List

app = FastAPI()

clients: set[WebSocket] = set()
conversations: list[dict] = []           
last_conversation: dict | None = None   

@app.get("/")
def root():
    return RedirectResponse("/chat")

@app.get("/chat")
def chat_page():
    return HTMLResponse(open("app/templates/chat.html", "r", encoding="utf-8").read())

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, role: str = "victim"):
    await ws.accept()
    print(f"[WS CONNECT] role={role}")
    clients.add(ws)
    try:
        while True:
            data = await ws.receive_json()
            payload = {
                "t": datetime.now().strftime("%H:%M"),  
                "role": role,
                "text": data.get("text", "")
            }
            conversations.append(payload)

            dead = []
            for peer in clients:
                try:
                    await peer.send_json(payload)
                except WebSocketDisconnect:
                    dead.append(peer)
                except RuntimeError:
                    dead.append(peer)
            for d in dead:
                clients.discard(d)
    except WebSocketDisconnect:
        print(f"[WS DISCONNECT] role={role}")
        clients.discard(ws)



@app.post("/end")
async def end_conversation(request: Request):
    global last_conversation, conversations

    last_conversation = {
        "ended": True,
        "messages": list(conversations),
    }

    # === Afișare în terminal ===
    print("\n📌 Conversația A FOST ÎNCHISĂ. Variabila last_conversation (dicționar):")
    print(last_conversation)

    # === Salvare în folderul exports/ ===
    os.makedirs("exports", exist_ok=True)
    filename = f"exports/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(last_conversation, f, ensure_ascii=False, indent=2)

    print(f"📂 Conversația a fost salvată în {filename}")

    return JSONResponse({"status": "ended", "messages": len(last_conversation["messages"])})



models.Base.metadata.create_all(bind=engine)

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



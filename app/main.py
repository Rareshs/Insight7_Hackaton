# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# 1) Creează întâi instanța FastAPI
app = FastAPI(title="BankScamGuard")

# 2) Rute HTTP simple (opțional, pentru test rapid)
@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"alive": True}

# 3) Abia ACUM definește WebSocket-ul Twilio
@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()   # Twilio trimite JSON text
            print("Twilio event:", msg[:200])  # vezi în consolă start/media/stop
    except WebSocketDisconnect:
        print("Twilio WS closed")

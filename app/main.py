from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from collections import defaultdict
from datetime import datetime
import os, json

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



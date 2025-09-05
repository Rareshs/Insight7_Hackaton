# app/main.py (adaugă)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from collections import defaultdict

app = FastAPI()

# room -> set of websockets
rooms: dict[str, set[WebSocket]] = defaultdict(set)

@app.get("/")
def root():
    return RedirectResponse("/chat")

@app.get("/chat")
def chat_page():
    # servește o pagină simplă (vezi HTML mai jos)
    return HTMLResponse(open("app/templates/chat.html", "r", encoding="utf-8").read())

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, room: str, role: str):
    await ws.accept()
    rooms[room].add(ws)
    try:
        while True:
            data = await ws.receive_json()  # { "role": "scammer|victim", "text": "..." }
            # atașăm rolul trimis de client și retransmitem în cameră
            payload = {"role": role, "text": data.get("text", "")}
            dead = []
            for peer in rooms[room]:
                try:
                    await peer.send_json(payload)
                except WebSocketDisconnect:
                    dead.append(peer)
            for d in dead:
                rooms[room].discard(d)
    except WebSocketDisconnect:
        rooms[room].discard(ws)
        if not rooms[room]:
            rooms.pop(room, None)

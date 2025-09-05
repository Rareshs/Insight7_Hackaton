# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path
import os, subprocess, traceback, time

# ── env
load_dotenv()
AGORA_APP_ID = os.getenv("AGORA_APP_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── persistent save dir
SAVE_DIR = Path("saved_audio")
SAVE_DIR.mkdir(exist_ok=True)

# ── app
app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "env": {"agora": bool(AGORA_APP_ID), "openai": bool(OPENAI_API_KEY)}}

@app.get("/env")
def get_env():
    return {"AGORA_APP_ID": AGORA_APP_ID}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    channel: Optional[str] = Form("default"),   # sent from frontend
):
    # --- directories & manifest for this channel
    chan_dir = SAVE_DIR / channel
    chan_dir.mkdir(parents=True, exist_ok=True)
    manifest = chan_dir / "chunks.txt"  # ffmpeg concat list

    # 1) Save uploaded chunk to a persistent file
    ts = int(time.time() * 1000)
    webm_path = chan_dir / f"chunk_{ts}.webm"
    data = await file.read()
    with open(webm_path, "wb") as out:
        out.write(data)
    print(f"[UPLOAD] {webm_path.name}  {len(data)} bytes")

    # Keep track of order for merging later
    with open(manifest, "a", encoding="utf-8") as mf:
        mf.write(f"file '{webm_path.name}'\n")

    # 2) Convert this chunk to wav (optional quick check)
    wav_path = webm_path.with_suffix(".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(webm_path), str(wav_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        print(f"[FFMPEG] wrote: {wav_path.name}")
    except subprocess.CalledProcessError as e:
        print("[FFMPEG ERROR]\n", e.stderr.decode(errors="ignore"))
        return JSONResponse({"error": "ffmpeg conversion failed"}, status_code=400)

    # 3) Whisper (optional)
    if not OPENAI_API_KEY:
        return {
            "transcript": f"[saved {wav_path.name}] (OPENAI_API_KEY not set)",
            "saved": str(webm_path),
            "channel": channel,
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(wav_path, "rb") as f:
            tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        return {"transcript": tr.text, "saved": str(webm_path), "channel": channel}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"whisper failed: {e}"}, status_code=500)

@app.post("/merge")
def merge_channel(channel: str):
    """Merge all saved chunks for a channel into one .webm and .wav."""
    chan_dir = SAVE_DIR / channel
    manifest = chan_dir / "chunks.txt"
    if not manifest.exists():
        return JSONResponse({"error": "no chunks manifest for this channel"}, status_code=404)

    merged_webm = chan_dir / f"{channel}_merged.webm"
    merged_wav  = chan_dir / f"{channel}_merged.wav"

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
             "-c", "copy", str(merged_webm)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": "webm concat failed", "stderr": e.stderr.decode(errors='ignore')}, status_code=400)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(merged_webm), str(merged_wav)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": "wav export failed", "stderr": e.stderr.decode(errors='ignore')}, status_code=400)

    return {"merged_webm": str(merged_webm), "merged_wav": str(merged_wav)}

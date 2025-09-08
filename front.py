import time
import random
from datetime import datetime
import requests
import streamlit as st
import random

st.set_page_config(page_title="Anti-Scam Dashboard", layout="wide")

# =============================
# Sidebar (backend + session)
# =============================
st.sidebar.title("Session Setup")
api_base = st.sidebar.text_input("Backend API Base", value="http://127.0.0.1:8000")
session_id = st.sidebar.text_input("Session ID", value="demo-001")
st.sidebar.markdown("---")
auto_wall = st.sidebar.checkbox("Auto-refresh Active Calls (5s)", value=False)
if st.sidebar.button("Clear conversation"):
    st.session_state.clear()

# =============================
# App state
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []  # {speaker, text}
if "risk_score" not in st.session_state:
    st.session_state.risk_score = 0  # 0..100 for UI
if "flagged_words" not in st.session_state:
    st.session_state.flagged_words = []
if "active_calls" not in st.session_state:
    st.session_state.active_calls = []
if "last_wall_fetch" not in st.session_state:
    st.session_state.last_wall_fetch = 0.0
# NEW: detalii selecție (pentru pagina Details)
if "selected_call_id" not in st.session_state:
    st.session_state.selected_call_id = None
if "selected_call_details" not in st.session_state:
    st.session_state.selected_call_details = None

# =============================
# Helpers (API + normalizers)
# =============================

def make_mock_details_from_card(call: dict) -> dict:
    score = int(call.get("score", 0))
    series = [max(1, int(score*s/100)) for s in (10, 25, 40, 60, 80, 100)]
    sample_transcript = [
        {"t": "12:41", "speaker": "agent",  "text": "Hello, I'm from your bank. We detected unusual activity."},
        {"t": "12:42", "speaker": "client", "text": "Oh no, what should I do?"},
        {"t": "12:42", "speaker": "agent",  "text": "Please read the 6-digit code we just sent."},
        {"t": "12:43", "speaker": "client", "text": "432118"},
        {"t": "12:43", "speaker": "agent",  "text": "Now your full card number and CVV to secure the account."},
    ]
    return {
        "id": call.get("id", "----"),
        "duration": call.get("duration", "--:--"),
        "score": score,
        "score_series": series,
        "transcript": sample_transcript,
        "flagged_words": ["otp", "card number", "cvv"],
        "live": bool(call.get("live", True)),
        "last_update": datetime.now().isoformat(timespec="seconds"),
        "raw": {"demo": True},
    }

def _join(base: str, path: str) -> str:
    return base.rstrip("/") + (path if path.startswith("/") else "/" + path)

def _score_to_0_100(score) -> int:
    if isinstance(score, (int, float)):
        return int(round(score * 100)) if score <= 1 else int(round(score))
    return 0

def _dur_to_mmss(dur) -> str:
    if isinstance(dur, (int, float)):
        mm = int(dur // 60); ss = int(dur % 60)
        return f"{mm:02d}:{ss:02d}"
    return dur or "--:--"

def _extract_transcript(payload):
    """
    Caută în payload câmpul corect pentru conversație și-l normalizează la
    [{t, speaker, text}, ...]
    """
    candidates = payload.get("transcript") or payload.get("messages") or payload.get("conversation") or []
    # dacă e listă de stringuri => mapăm
    if candidates and isinstance(candidates[0], str):
        return [{"t": "--:--", "speaker": "agent", "text": x} for x in candidates]
    # dacă are altă schemă, încercăm câteva aliasuri uzuale
    norm = []
    for m in candidates:
        if isinstance(m, dict):
            t = m.get("t") or m.get("time") or m.get("timestamp") or "--:--"
            sp = m.get("speaker") or m.get("role") or m.get("from") or "agent"
            tx = m.get("text") or m.get("message") or m.get("content") or ""
            norm.append({"t": t, "speaker": sp, "text": tx})
    return norm

# =============================
# Backend calls (+ demo fallback)
# =============================
def analyze_messages(base_url: str, messages: list[str]):
    """POST /analyze  →  { risk_score: 0..1 | 0..100, flagged_words: [] }"""
    url = _join(base_url, "/analyze")
    try:
        payload = {"messages": messages}
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        return {
            "ok": True,
            "score": _score_to_0_100(data.get("risk_score", 0)),
            "flagged": data.get("flagged_words", []) or []
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def fetch_active_calls(base_url: str):
    """
    Încearcă:
      1) GET /conversations     (ce aveți acum)
      2) GET /calls/active      (fallback propus)
    Dacă niciuna nu merge, folosește demo-ul local.
    """
    for path in ("/conversations", "/calls/active"):
        try:
            url = _join(base_url, path)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            raw = resp.json() or []

            # acceptă listă sau dict cu chei uzuale
            if isinstance(raw, dict):
                items = (
                    raw.get("conversations")
                    or raw.get("items")
                    or raw.get("data")
                    or raw.get("results")       # <— adăugat
                    or []
                )
            else:
                items = raw

            norm = []
            for c in items:
                if not isinstance(c, dict):
                    continue
                rs = _score_to_0_100(c.get("risk_score", 0))
                dur = _dur_to_mmss(c.get("duration", c.get("duration_seconds", "--:--")))
                norm.append({
                    "id": c.get("conversation_id", c.get("id", "----")),
                    "duration": dur,
                    "score": rs,
                    "live": bool(c.get("is_live", c.get("live", False))),
                    "last_update": c.get("last_update"),
                })
            if norm:
                return {"ok": True, "items": norm}
        except Exception:
            pass

    # fallback demo
    demo = [
        {"id": "A1B2", "duration": "02:15", "score": 12,  "live": False},
        {"id": "B4C5", "duration": "08:42", "score": 65,  "live": False},
        {"id": "C3D4", "duration": "04:20", "score": 30,  "live": False},
        {"id": "D6E7", "duration": "12:58", "score": 91,  "live": True},
    ]
    return {"ok": True, "items": demo, "demo": True}

def fetch_call_details(base_url: str, call_id: str):
    """
    Încearcă:
      1) GET /conversations/{id}
      2) GET /calls/{id}
    Normalizează scorul și transcriptul.
    Dacă pică, întoarce mock detaliat.
    """
    for path in (f"/conversations/{call_id}", f"/calls/{call_id}"):
        try:
            url = _join(base_url, path)
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            p = r.json() or {}

            score_ui = _score_to_0_100(p.get("risk_score", 0))
            series = p.get("score_series", []) or []
            series = [_score_to_0_100(x) for x in series]
            duration = _dur_to_mmss(p.get("duration", p.get("duration_seconds", "--:--")))
            transcript = _extract_transcript(p)

            details = {
                "id": p.get("conversation_id", p.get("id", call_id)),
                "duration": duration,
                "score": score_ui if score_ui else (series[-1] if series else 0),
                "score_series": series or [10, 25, 40, 60, 91],
                "transcript": transcript,
                "flagged_words": p.get("flagged_words", []),
                "raw": p,
                "live": bool(p.get("is_live", p.get("live", False))),
                "last_update": p.get("last_update", datetime.now().isoformat(timespec="seconds")),
            }
            return {"ok": True, "details": details}
        except Exception:
            pass

    # mock detalii dacă API-ul nu e gata
    sample_transcript = [
        {"t": "12:41", "speaker": "agent",  "text": "Hello, I'm from your bank. We detected unusual activity."},
        {"t": "12:42", "speaker": "client", "text": "Oh no, what should I do?"},
        {"t": "12:42", "speaker": "agent",  "text": "Please read the 6-digit code we just sent."},
        {"t": "12:43", "speaker": "client", "text": "432118"},
        {"t": "12:43", "speaker": "agent",  "text": "Now your full card number and CVV, to secure the account."},
    ]
    details = {
        "id": call_id,
        "duration": "12:58",
        "score": random.choice([68, 72, 91]),
        "score_series": [10, 12, 15, 22, 35, 60, 78, 91],
        "transcript": sample_transcript,
        "flagged_words": ["otp", "card number", "cvv"],
        "raw": {"demo": True},
        "live": True,
        "last_update": datetime.now().isoformat(timespec="seconds"),
    }
    return {"ok": True, "details": details, "demo": True}

def status_from_score(score_0_100: int):
    if score_0_100 < 30:
        return ("safe", "#28c76f")
    if score_0_100 < 70:
        return ("suspect", "#ff9f43")
    return ("ALERT", "#ea5455")

# =============================
# Header
# =============================
left, right = st.columns([2,1])
with left:
    st.title("📞 Anti-Scam Dashboard")
    st.caption("Live transcript, risk score, and Active Calls wall")
with right:
    st.metric("Current Session", session_id)

# =============================
# Tabs: Live Monitor / Active Calls
# =============================
tab_live, tab_wall = st.tabs(["🎧 Live Monitor", "📋 Active Calls"])

# -----------------------------
# LIVE MONITOR TAB
# -----------------------------
with tab_live:
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Conversation")
        transcript_box = st.container()
    with col2:
        st.subheader("Risk Analysis")
        score_placeholder = st.empty()
        bar_placeholder = st.empty()
        flagged_box = st.container()

    # Input form
    with st.form("send_message"):
        c1, c2 = st.columns([1,3])
        with c1:
            speaker = st.selectbox("Speaker", ["client", "agent"], index=0)
        with c2:
            text = st.text_input("Message")
        submitted = st.form_submit_button("Send & Analyze")

    if submitted and text:
        st.session_state.messages.append({"speaker": speaker, "text": text})
        res = analyze_messages(api_base, [m["text"] for m in st.session_state.messages])
        if res["ok"]:
            st.session_state.risk_score = res["score"]
            st.session_state.flagged_words = res["flagged"]
        else:
            st.error(f"Analyze failed: {res['error']}")

    # Render transcript
    with transcript_box:
        for m in st.session_state.messages:
            bg = "#0b1020" if m["speaker"] == "agent" else "#0f1a2b"
            st.markdown(
                f"<div style='padding:10px;border-radius:10px;margin-bottom:6px;background:{bg}'>"
                f"<b>{m['speaker'].capitalize()}:</b> {m['text']}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Render risk
    score_placeholder.metric("Risk Score", f"{st.session_state.risk_score}/100")
    bar_placeholder.progress(min(max(st.session_state.risk_score, 0), 100))

    with flagged_box:
        if st.session_state.flagged_words:
            st.caption("Flagged words:")
            for w in st.session_state.flagged_words:
                st.markdown(f"• {w}")
        else:
            st.caption("No suspicious words detected yet.")

# -----------------------------
# ACTIVE CALLS TAB
# -----------------------------
with tab_wall:
    st.subheader("Active Calls (Live Wall)")

    refresh = st.button("Refresh now", key="refresh_wall")

    # Optional auto-refresh every 5 seconds când e bifat
    if auto_wall:
        now = time.time()
        if now - st.session_state.last_wall_fetch > 5:
            refresh = True
            st.session_state.last_wall_fetch = now

    if refresh or not st.session_state.active_calls:
        data = fetch_active_calls(api_base)
        if data["ok"]:
            st.session_state.active_calls = data["items"]
            if data.get("demo"):
                st.caption("Using demo data for Active Calls (backend not available).")
        else:
            st.error(f"Failed to fetch active calls: {data['error']}")

    calls = st.session_state.active_calls or []
    cols = st.columns(2)
    for i, call in enumerate(calls):
        with cols[i % 2]:
            label, color = status_from_score(call.get("score", 0))
            live_dot = "<span style='color:#ff4757'>●</span> Live" if call.get("live") else ""
            score_display = (call.get("score", 0) / 100)
            st.markdown(
                f"""
                <div style='background:#0f1221;border:1px solid #1c2236;border-radius:16px;padding:16px;margin-bottom:16px;'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div style='font-weight:600;font-size:18px'>Call #{call.get('id','----')}</div>
                    <div>{live_dot}</div>
                  </div>
                  <div style='opacity:.85;margin-top:6px'>{call.get('duration','--:--')}</div>
                  <div style='margin-top:8px'>Scam Score: {score_display:.2f}</div>
                  <div style='margin-top:6px;color:{color};font-weight:600'>{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # DETALII: setăm selecția și încercăm să luăm detalii (sau mock)
            if st.button("Details", key=f"details_{call.get('id','----')}"):
                st.session_state.selected_call_id = call.get("id", "----")
                with st.spinner("Loading details..."):
                    det = fetch_call_details(api_base, st.session_state.selected_call_id)
                st.session_state.selected_call_details = det.get("details") if det.get("ok") else None
                st.rerun()

# -----------------------------
# DETAILS VIEW (header, metrici, istoric, conversație + raw JSON)
# -----------------------------
if st.session_state.get("selected_call_id") and st.session_state.get("selected_call_details"):
    d = st.session_state.selected_call_details

    # HEADER
    st.markdown("---")
    st.markdown(f"### 📄 Call Details — **#{d['id']}**")

    # METRICI SUS
    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Risk Score", f"{d['score']}/100")
    with top2:
        st.metric("Duration", d.get("duration", "--:--"))
    with top3:
        label, color = status_from_score(d['score'])
        st.markdown(f"<div style='color:{color};font-weight:700;font-size:18px'>{label}</div>", unsafe_allow_html=True)
    with top4:
        live_badge = "<span style='color:#ff4757'>●</span> Live" if d.get("live") else "Ended"
        st.markdown(live_badge, unsafe_allow_html=True)

    # ISTORIC SCOR
    st.caption("Score history")
    st.line_chart({"risk": d.get("score_series", [])})

    # CONVERSAȚIA
    st.subheader("Conversation")
    for m in d.get("transcript", []):
        bg = "#0b1020" if m.get("speaker") == "agent" else "#0f1a2b"
        st.markdown(
            f"<div style='padding:10px;border-radius:10px;margin-bottom:6px;background:{bg}'>"
            f"<b>{m.get('t','--:--')} — {m.get('speaker','').capitalize()}:</b> {m.get('text','')}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # JSON BRUT (pt. debugging/validare API)
    with st.expander("🔎 Raw JSON (full payload)"):
        st.json(d.get("raw", {}))

    st.markdown("---")
    if st.button("◀ Back to wall"):
        st.session_state.selected_call_id = None
        st.session_state.selected_call_details = None
        st.rerun()
    
    
st.markdown(
    """
    <a href="https://192.168.198.7:8443/chat?room=demo1&role=victim" target="_blank">
        <button style="padding:10px 20px;font-size:16px;font-weight:600;border:none;border-radius:8px;background:#0ea5e9;color:white;cursor:pointer;">
            🕵️ Join Call
        </button>
    </a>
    """,
    unsafe_allow_html=True
)
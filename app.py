import time
from datetime import datetime, timedelta
import requests
import streamlit as st

st.set_page_config(page_title="Anti‑Scam Dashboard", layout="wide")

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

# =============================
# Helpers (API)
# =============================

def _join(base: str, path: str) -> str:
    return base.rstrip("/") + (path if path.startswith("/") else "/" + path)


def analyze_messages(base_url: str, messages: list[str]):
    """POST /analyze  →  { risk_score: 0..1 | 0..100, flagged_words: [] }"""
    url = _join(base_url, "/analyze")
    try:
        payload = {"messages": messages}
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        score = data.get("risk_score", 0)
        # Normalize to 0..100 for UI
        score_ui = int(round(score * 100)) if score <= 1 else int(score)
        words = data.get("flagged_words", []) or []
        return {"ok": True, "score": score_ui, "flagged": words}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_active_calls(base_url: str):
    """GET /calls/active → list of calls or returns demo if missing."""
    url = _join(base_url, "/calls/active")
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        calls = resp.json() or []
        # Normalize to a common structure
        norm = []
        for c in calls:
            rs = c.get("risk_score", 0)
            score_ui = int(round(rs * 100)) if rs <= 1 else int(rs)
            dur = c.get("duration_seconds")
            if isinstance(dur, (int, float)):
                mm = int(dur // 60)
                ss = int(dur % 60)
                duration = f"{mm:02d}:{ss:02d}"
            else:
                duration = c.get("duration", "--:--")
            norm.append({
                "id": c.get("id", "----"),
                "duration": duration,
                "score": score_ui,
                "live": bool(c.get("is_live", False)),
                "last_update": c.get("last_update"),
            })
        return {"ok": True, "items": norm}
    except Exception:
        # Demo data fallback
        demo = [
            {"id": "A1B2", "duration": "02:15", "score": 12,  "live": False},
            {"id": "B4C5", "duration": "08:42", "score": 65,  "live": False},
            {"id": "C3D4", "duration": "04:20", "score": 30,  "live": False},
            {"id": "D6E7", "duration": "12:58", "score": 91,  "live": True},
        ]
        return {"ok": True, "items": demo, "demo": True}


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
    st.title("📞 Anti‑Scam Dashboard")
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

    # Optional auto-refresh every 5 seconds when toggled
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
                st.caption("Using demo data for Active Calls (backend /calls/active not available).")
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
            st.button("Details", key=f"details_{call.get('id','----')}")

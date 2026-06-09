"""
app.py
Production-ready AI chatbot — ChatGPT-style Streamlit UI.
Conversation metadata is persisted to SQLite (chat.db).
LangGraph checkpoints are persisted to SQLite (checkpoints.db).

Run with:
    streamlit run app.py

Environment variables (set in .env):
    OPENAI_API_KEY   or   GOOGLE_API_KEY   or   MISTRAL_API_KEY
"""

import streamlit as st
from dotenv import load_dotenv

from database import DB
from graph import stream_response
from utils import (
    export_chat_txt,
    format_timestamp,
    generate_title,
    new_thread_id,
    now_iso,
)

load_dotenv()

# ============================================================
# Page config — must be the very first Streamlit call
# ============================================================
st.set_page_config(
    page_title="AI Chat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — dark modern UI
# ============================================================
st.markdown(
    """
<style>
/* ── Global reset / base ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d0f !important;
    color: #e8e8ea !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* ── Hide default Streamlit chrome ───────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111114 !important;
    border-right: 1px solid #1e1e24 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

/* ── Sidebar header ──────────────────────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.9rem 1rem 0.6rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
    border-bottom: 1px solid #1e1e24;
    margin-bottom: 0.75rem;
}
.sidebar-logo span.accent { color: #7c6af7; }

/* ── New-chat button ─────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button {
    background: #1e1e28 !important;
    color: #c8c8d4 !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 10px !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 1rem !important;
    width: 100%;
    transition: background 0.15s, border-color 0.15s;
}
[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button:hover {
    background: #28283a !important;
    border-color: #7c6af7 !important;
    color: #ffffff !important;
}

/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.25rem 0 !important;
}

/* ── Chat input ──────────────────────────────────────────── */
[data-testid="stChatInputContainer"] {
    background: #16161c !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 14px !important;
    padding: 0.25rem 0.5rem !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 3px rgba(124,106,247,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e8e8ea !important;
    caret-color: #7c6af7;
}

/* ── Typing indicator ────────────────────────────────────── */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0.6rem 0;
}
.typing-indicator span {
    width: 7px; height: 7px;
    background: #7c6af7;
    border-radius: 50%;
    animation: bounce 1.2s infinite ease-in-out;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%            { transform: translateY(-6px); opacity: 1; }
}

/* ── Welcome screen ──────────────────────────────────────── */
.welcome-wrap {
    text-align: center;
    margin-top: 6rem;
    color: #55556a;
}
.welcome-wrap h2 {
    font-size: 2rem;
    font-weight: 700;
    color: #c8c8d4;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
}
.welcome-wrap p { font-size: 0.92rem; }

/* ── Misc buttons ────────────────────────────────────────── */
[data-testid="stDownloadButton"] button,
.stButton button {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    border: 1px solid #2a2a38 !important;
    background: #1a1a24 !important;
    color: #b0b0c0 !important;
    transition: border-color 0.15s, color 0.15s;
}
[data-testid="stDownloadButton"] button:hover,
.stButton button:hover {
    border-color: #7c6af7 !important;
    color: #ffffff !important;
}

/* ── DB badge ────────────────────────────────────────────── */
.db-badge {
    font-size: 0.65rem;
    color: #3a3a52;
    padding: 0.2rem 0.5rem;
    border: 1px solid #1e1e2c;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* ── Main layout ─────────────────────────────────────────── */
[data-testid="stMain"] { background: #0d0d0f !important; }
.block-container {
    max-width: 800px !important;
    padding: 1.5rem 2rem !important;
    margin: auto;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Session-state bootstrap
# ============================================================

def init_session():
    """Initialise all required session_state keys on first load."""
    if "conversations" not in st.session_state:
        # Load all conversation metadata from SQLite into memory for fast sidebar rendering.
        # We keep this in session_state as a cache; SQLite is always the source of truth.
        st.session_state.conversations = DB.load_all_conversations()

    if "active_thread" not in st.session_state:
        st.session_state.active_thread = None

    if "generating" not in st.session_state:
        st.session_state.generating = False


init_session()


# ============================================================
# Helpers
# ============================================================

def create_new_conversation() -> str:
    """Create a blank conversation in DB + session_state, set active, return thread_id."""
    tid = new_thread_id()
    ts = now_iso()
    meta = {"title": "New Chat", "messages": [], "timestamp": ts}

    # Persist to SQLite
    DB.save_conversation(tid, meta["title"], meta["messages"], ts)

    # Update session cache
    st.session_state.conversations[tid] = meta
    st.session_state.active_thread = tid
    return tid


def delete_conversation(thread_id: str):
    """Remove conversation from SQLite and session cache."""
    DB.delete_conversation(thread_id)
    st.session_state.conversations.pop(thread_id, None)

    if st.session_state.active_thread == thread_id:
        remaining = list(st.session_state.conversations.keys())
        st.session_state.active_thread = remaining[-1] if remaining else None


def get_active_messages() -> list[dict]:
    """Return the message list for the currently active thread (from cache)."""
    tid = st.session_state.active_thread
    if tid and tid in st.session_state.conversations:
        return st.session_state.conversations[tid]["messages"]
    return []


def sorted_conversations() -> list[tuple[str, dict]]:
    """Return conversations sorted by timestamp, newest first."""
    return sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1].get("timestamp", ""),
        reverse=True,
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    # Brand header
    st.markdown(
        '<div class="sidebar-logo">✦ <span class="accent">AI</span> Chat</div>',
        unsafe_allow_html=True,
    )

    # SQLite indicator badge
    st.markdown(
        '<span class="db-badge">💾 SQLite persistent storage</span>',
        unsafe_allow_html=True,
    )

    # New Chat
    if st.button("＋  New Chat", use_container_width=True):
        create_new_conversation()
        st.rerun()

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    # Conversation list
    conversations_sorted = sorted_conversations()

    if conversations_sorted:
        st.markdown(
            "<p style='font-size:0.7rem;color:#444455;padding:0 0.25rem;"
            "margin-bottom:0.35rem;text-transform:uppercase;letter-spacing:0.08em'>"
            "Conversations</p>",
            unsafe_allow_html=True,
        )

    for tid, meta in conversations_sorted:
        col_title, col_del = st.columns([5, 1])

        with col_title:
            label = meta["title"][:38] + ("…" if len(meta["title"]) > 38 else "")
            if st.button(label, key=f"conv_{tid}", use_container_width=True, type="secondary"):
                # Re-load from DB in case another session modified it
                fresh = DB.load_conversation(tid)
                if fresh:
                    st.session_state.conversations[tid] = fresh
                st.session_state.active_thread = tid
                st.rerun()

        with col_del:
            if st.button("✕", key=f"del_{tid}", help="Delete conversation"):
                delete_conversation(tid)
                st.rerun()

    # ── Sidebar footer ──
    if st.session_state.active_thread:
        st.markdown("---")
        tid = st.session_state.active_thread
        msgs = get_active_messages()
        title = st.session_state.conversations.get(tid, {}).get("title", "Chat")

        if msgs:
            txt_data = export_chat_txt(title, msgs)
            st.download_button(
                "⬇  Export as TXT",
                data=txt_data,
                file_name=f"{title[:30].replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if msgs and st.button("🗑  Clear Chat", use_container_width=True):
            ts = now_iso()
            DB.clear_messages(tid, ts)
            st.session_state.conversations[tid]["messages"] = []
            st.session_state.conversations[tid]["timestamp"] = ts
            st.rerun()


# ============================================================
# Main chat area
# ============================================================

active_tid = st.session_state.active_thread

# ── No conversation selected ─────────────────────────────────
if not active_tid:
    st.markdown(
        """
        <div class="welcome-wrap">
            <h2>What can I help with?</h2>
            <p>Start a new conversation from the sidebar, or pick an existing one.</p>
            <p style="font-size:0.78rem;margin-top:1rem;color:#33334a">
                Conversations persist across restarts via SQLite.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Guard: thread was deleted ────────────────────────────────
if active_tid not in st.session_state.conversations:
    st.session_state.active_thread = None
    st.rerun()

conv = st.session_state.conversations[active_tid]
messages: list[dict] = conv["messages"]

# ── Conversation title bar ───────────────────────────────────
st.markdown(
    f"<h3 style='margin:0 0 0.75rem;font-size:1rem;font-weight:600;"
    f"color:#9090a8;letter-spacing:-0.01em'>{conv['title']}</h3>",
    unsafe_allow_html=True,
)

# ── Render existing messages ─────────────────────────────────
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────
user_input = st.chat_input("Message AI Chat…", disabled=st.session_state.generating)

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()

    ts = now_iso()

    # Auto-generate title from the very first message
    if not messages:
        title = generate_title(user_input)
        conv["title"] = title
        DB.update_title(active_tid, title)
        st.session_state.conversations[active_tid]["title"] = title

    # ── Save and display user message ────────────────────────
    DB.append_message(active_tid, "user", user_input, ts)
    messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Stream assistant response ────────────────────────────
    st.session_state.generating = True

    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            '<div class="typing-indicator">'
            "<span></span><span></span><span></span>"
            "</div>",
            unsafe_allow_html=True,
        )

        response_placeholder = st.empty()
        full_response = ""

        try:
            first_chunk = True
            for chunk in stream_response(user_input, active_tid):
                if first_chunk:
                    typing_placeholder.empty()
                    first_chunk = False
                full_response += chunk
                response_placeholder.markdown(full_response + "▍")

            response_placeholder.markdown(full_response)

        except Exception as exc:
            typing_placeholder.empty()
            full_response = f"⚠️ Error: {exc}"
            response_placeholder.markdown(full_response)

    # ── Persist assistant message to SQLite ──────────────────
    ts = now_iso()
    DB.append_message(active_tid, "assistant", full_response, ts)
    messages.append({"role": "assistant", "content": full_response})
    conv["timestamp"] = ts

    st.session_state.generating = False
    st.rerun()
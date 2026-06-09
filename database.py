"""
database.py
SQLite persistence layer for conversation metadata.

This module owns a single SQLite database file (chat.db) with one table:

    conversations
    ─────────────
    thread_id   TEXT  PRIMARY KEY   — UUID identifying the LangGraph thread
    title       TEXT                — Auto-generated from first user message
    timestamp   TEXT                — ISO-8601 UTC, updated on every new message
    messages    TEXT                — JSON-encoded list of {role, content} dicts

LangGraph's own checkpoint data (full message history + state snapshots) is
stored in a SEPARATE file (checkpoints.db) managed by SqliteSaver.  This table
stores only the lightweight metadata the sidebar needs (title, time, messages
for display) so we never have to deserialise LangGraph's internal format.

Usage
-----
    from database import DB
    DB.save_conversation(thread_id, title, messages, timestamp)
    convs = DB.load_all_conversations()   # → {thread_id: {title, messages, timestamp}}
    DB.delete_conversation(thread_id)
"""

import json
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Database file paths
# ---------------------------------------------------------------------------

DB_PATH = Path("chat.db")                  # conversation metadata
CHECKPOINT_DB_PATH = str(Path("checkpoints.db"))  # LangGraph SqliteSaver


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Open (or create) the metadata database and return a connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row       # allow column-name access
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent read-write safety
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the conversations table if it does not exist yet."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id TEXT PRIMARY KEY,
            title     TEXT NOT NULL DEFAULT 'New Chat',
            timestamp TEXT NOT NULL,
            messages  TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Module-level singleton connection
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    """
    Return the module-level SQLite connection, creating it on first call.

    The connection is intentionally kept open for the lifetime of the process
    (safe for Streamlit — one process per session by default).
    """
    global _conn
    if _conn is None:
        _conn = _get_connection()
        _ensure_schema(_conn)
    return _conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DB:
    """Namespace for all database operations."""

    # ── Read ─────────────────────────────────────────────────────────────────

    @staticmethod
    def load_all_conversations() -> dict:
        """
        Load every conversation from the database.

        Returns:
            Dict mapping thread_id → {title, timestamp, messages}
            where `messages` is a Python list of {role, content} dicts.
        """
        conn = get_db()
        rows = conn.execute(
            "SELECT thread_id, title, timestamp, messages FROM conversations"
        ).fetchall()

        result = {}
        for row in rows:
            try:
                msgs = json.loads(row["messages"])
            except json.JSONDecodeError:
                msgs = []
            result[row["thread_id"]] = {
                "title": row["title"],
                "timestamp": row["timestamp"],
                "messages": msgs,
            }
        return result

    @staticmethod
    def load_conversation(thread_id: str) -> dict | None:
        """
        Load a single conversation by thread_id.

        Returns:
            {title, timestamp, messages} dict, or None if not found.
        """
        conn = get_db()
        row = conn.execute(
            "SELECT title, timestamp, messages FROM conversations WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()

        if row is None:
            return None

        try:
            msgs = json.loads(row["messages"])
        except json.JSONDecodeError:
            msgs = []

        return {
            "title": row["title"],
            "timestamp": row["timestamp"],
            "messages": msgs,
        }

    # ── Write ─────────────────────────────────────────────────────────────────

    @staticmethod
    def save_conversation(
        thread_id: str,
        title: str,
        messages: list[dict],
        timestamp: str,
    ) -> None:
        """
        Insert or fully replace a conversation record.

        Args:
            thread_id:  UUID string.
            title:      Human-readable conversation title.
            messages:   List of {role, content} message dicts.
            timestamp:  ISO-8601 UTC string.
        """
        conn = get_db()
        conn.execute(
            """
            INSERT INTO conversations (thread_id, title, timestamp, messages)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title     = excluded.title,
                timestamp = excluded.timestamp,
                messages  = excluded.messages
            """,
            (thread_id, title, timestamp, json.dumps(messages, ensure_ascii=False)),
        )
        conn.commit()

    @staticmethod
    def update_title(thread_id: str, title: str) -> None:
        """Update only the title of an existing conversation."""
        conn = get_db()
        conn.execute(
            "UPDATE conversations SET title = ? WHERE thread_id = ?",
            (title, thread_id),
        )
        conn.commit()

    @staticmethod
    def append_message(thread_id: str, role: str, content: str, timestamp: str) -> None:
        """
        Append a single message to an existing conversation's message list
        and update its timestamp — without loading the whole object first.

        Args:
            thread_id: UUID string.
            role:      'user' or 'assistant'.
            content:   Message text.
            timestamp: ISO-8601 UTC string (updated on the row too).
        """
        conn = get_db()
        # Fetch current messages JSON, append, write back
        row = conn.execute(
            "SELECT messages FROM conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()

        if row is None:
            return  # conversation must exist before appending

        try:
            msgs = json.loads(row["messages"])
        except json.JSONDecodeError:
            msgs = []

        msgs.append({"role": role, "content": content})

        conn.execute(
            "UPDATE conversations SET messages = ?, timestamp = ? WHERE thread_id = ?",
            (json.dumps(msgs, ensure_ascii=False), timestamp, thread_id),
        )
        conn.commit()

    # ── Delete ────────────────────────────────────────────────────────────────

    @staticmethod
    def delete_conversation(thread_id: str) -> None:
        """Permanently remove a conversation and all its messages."""
        conn = get_db()
        conn.execute(
            "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
        )
        conn.commit()

    # ── Clear ─────────────────────────────────────────────────────────────────

    @staticmethod
    def clear_messages(thread_id: str, timestamp: str) -> None:
        """
        Wipe the message list for a conversation (keep the row itself).

        Args:
            thread_id: UUID string.
            timestamp: New timestamp to record.
        """
        conn = get_db()
        conn.execute(
            "UPDATE conversations SET messages = '[]', timestamp = ? WHERE thread_id = ?",
            (timestamp, thread_id),
        )
        conn.commit()
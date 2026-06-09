"""
utils.py
Utility functions for conversation management.

Persistence is now handled entirely by database.py (SQLite).
This module keeps only the stateless helpers:
  - Thread ID generation
  - Title generation
  - Timestamp formatting
  - TXT export
"""

import re
import uuid
from datetime import datetime


# ---------------------------------------------------------------------------
# Conversation lifecycle helpers
# ---------------------------------------------------------------------------

def new_thread_id() -> str:
    """Generate a new unique UUID4 string for a conversation thread."""
    return str(uuid.uuid4())


def generate_title(first_user_message: str, max_length: int = 45) -> str:
    """
    Derive a short, human-readable title from the first user message.

    Args:
        first_user_message: Raw text of the user's opening message.
        max_length:         Maximum character length for the title.

    Returns:
        A clean, truncated string suitable for use as a sidebar label.
    """
    clean = re.sub(r"\s+", " ", first_user_message.strip())
    if len(clean) <= max_length:
        return clean
    return clean[:max_length].rstrip() + "…"


def format_timestamp(iso_string: str) -> str:
    """
    Convert an ISO-8601 timestamp string to a friendly display format.

    Returns a string like "Jun 09 · 14:32", or the original on parse error.
    """
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%b %d · %H:%M")
    except ValueError:
        return iso_string


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def export_chat_txt(title: str, messages: list[dict]) -> str:
    """
    Serialize a conversation to a plain-text string suitable for download.

    Args:
        title:    Conversation title used as the file header.
        messages: List of {"role": str, "content": str} dicts.

    Returns:
        Multi-line string with labelled messages.
    """
    lines = [
        f"# {title}",
        f"Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)
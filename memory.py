"""
memory.py
Provides a persistent SqliteSaver checkpointer backed by checkpoints.db.

FIX: SqliteSaver.from_conn_string() returns a context manager (_GeneratorContextManager),
NOT a BaseCheckpointSaver instance directly — which causes LangGraph to throw:
  "Invalid checkpointer provided. Expected an instance of BaseCheckpointSaver"

Solution: Open a raw sqlite3.Connection and pass it to SqliteSaver(conn) directly.
This gives us a proper BaseCheckpointSaver instance that LangGraph accepts.

The checkpointer + connection are cached in st.session_state so they survive
Streamlit reruns without reopening the file each time.
"""

import sqlite3

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from database import CHECKPOINT_DB_PATH


def get_checkpointer() -> SqliteSaver:
    """
    Return the single SqliteSaver instance for this Streamlit session.

    Opens a persistent SQLite connection to checkpoints.db and wraps it
    in SqliteSaver directly (bypassing the context-manager API which
    returns a _GeneratorContextManager, not a BaseCheckpointSaver).
    """
    if "checkpointer" not in st.session_state:
        # Open the SQLite connection manually
        conn = sqlite3.connect(
            CHECKPOINT_DB_PATH,
            check_same_thread=False,   # required for Streamlit's threading model
        )
        conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent read-write

        # Instantiate SqliteSaver directly — this IS a BaseCheckpointSaver
        checkpointer = SqliteSaver(conn)

        # Cache both so the connection stays open across reruns
        st.session_state["checkpointer"] = checkpointer
        st.session_state["checkpoint_conn"] = conn

    return st.session_state["checkpointer"]
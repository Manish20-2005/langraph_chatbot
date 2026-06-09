"""
graph.py
Builds and compiles the LangGraph StateGraph that powers the chatbot.

Architecture
------------
START → chat_node → END

The single `chat_node` calls the configured LLM with the full message history
stored in ChatState.  The compiled graph is cached in st.session_state to
avoid rebuilding it on every Streamlit rerun.
"""

import os
import time
import random
from typing import Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from memory import get_checkpointer
from state import ChatState

load_dotenv()

# ---------------------------------------------------------------------------
# Rate-limit retry config
# ---------------------------------------------------------------------------

MAX_RETRIES = 4          # maximum number of retry attempts
BASE_DELAY  = 5.0        # initial wait in seconds before first retry
MAX_DELAY   = 60.0       # cap on wait time between retries


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a 429 / rate-limit error."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "rate_limited" in msg
        or "too many requests" in msg
    )


def _with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) and retry on rate-limit errors using
    exponential backoff with jitter.

    Raises the original exception if all retries are exhausted.
    """
    delay = BASE_DELAY
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise
            # Add ±20 % jitter so parallel sessions don't all retry at once
            jitter = delay * 0.2 * random.random()
            wait = min(delay + jitter, MAX_DELAY)
            print(f"[graph] Rate limited — waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY)   # exponential backoff


# ---------------------------------------------------------------------------
# LLM factory – supports OpenAI, Google Gemini, and Mistral
# ---------------------------------------------------------------------------

def _build_llm():
    """
    Instantiate and return the chat model.

    Priority:
        1. OPENAI_API_KEY   → ChatOpenAI        (default: gpt-4o-mini)
        2. GOOGLE_API_KEY   → ChatGoogleGenerativeAI (default: gemini-1.5-flash)
        3. MISTRAL_API_KEY  → ChatMistralAI     (default: mistral-large-latest)

    Raises:
        EnvironmentError if no key is set.
    """
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI  # type: ignore
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, streaming=True, temperature=0.7)

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        model_name = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=model_name, streaming=True, temperature=0.7)

    if os.getenv("MISTRAL_API_KEY"):
        from langchain_mistralai import ChatMistralAI  # type: ignore
        model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        return ChatMistralAI(model=model_name, streaming=True, temperature=0.7)

    raise EnvironmentError(
        "No LLM API key found. Set OPENAI_API_KEY, GOOGLE_API_KEY, or MISTRAL_API_KEY in .env"
    )


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------

def chat_node(state: ChatState) -> dict:
    """
    Core LangGraph node.

    Invokes the LLM with retry logic to handle 429 rate-limit responses
    from the API gracefully.
    """
    llm = _build_llm()
    response: BaseMessage = _with_retry(llm.invoke, state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph builder / cache
# ---------------------------------------------------------------------------

def get_graph():
    """
    Return the compiled LangGraph application.

    The graph is built once and stored in st.session_state["graph"] so it
    is not re-compiled on every Streamlit rerun.
    """
    if "graph" not in st.session_state:
        checkpointer = get_checkpointer()

        builder = StateGraph(ChatState)
        builder.add_node("chat_node", chat_node)
        builder.add_edge(START, "chat_node")
        builder.add_edge("chat_node", END)

        st.session_state["graph"] = builder.compile(checkpointer=checkpointer)

    return st.session_state["graph"]


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def stream_response(user_message: str, thread_id: str) -> Iterator[str]:
    """
    Stream tokens from the LLM for a given user message and thread.

    Handles 429 rate-limit errors by:
      1. Yielding a user-visible "waiting…" notice immediately.
      2. Sleeping with exponential backoff.
      3. Re-trying the full stream from the beginning.

    Args:
        user_message: The raw text typed by the user.
        thread_id:    UUID string identifying the current conversation thread.

    Yields:
        Individual text chunks (tokens) as they arrive from the LLM.
    """
    from langchain_core.messages import HumanMessage

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": [HumanMessage(content=user_message)]}

    delay = BASE_DELAY

    for attempt in range(MAX_RETRIES + 1):
        try:
            for chunk in graph.stream(input_state, config=config, stream_mode="messages"):
                message_chunk, _ = chunk
                if hasattr(message_chunk, "content") and message_chunk.content:
                    yield message_chunk.content
            return  # stream completed successfully

        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise

            jitter = delay * 0.2 * random.random()
            wait = min(delay + jitter, MAX_DELAY)
            wait_int = int(wait)

            # Surface the wait to the user in the chat bubble
            yield (
                f"\n\n⏳ *Mistral rate limit hit — retrying in {wait_int}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})…*\n\n"
            )
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY)
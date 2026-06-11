"""
graph.py
Builds and compiles the LangGraph StateGraph that powers the chatbot.

Architecture (ReAct agent with tools)
--------------------------------------
START → agent_node → [tool_node (if tool calls)] → agent_node → … → END

The agent decides when to call tools.  ToolNode executes the chosen tool
and returns the result back to agent_node, which then generates a final reply.

Tools available:
  - web_search         : DuckDuckGo web search
  - get_stock_price    : Real-time stock data (yfinance)
  - get_company_info   : Company fundamentals (yfinance)
  - calculator         : Safe math expression evaluator
  - get_weather        : Current weather (Open-Meteo, free)
  - convert_currency   : Live currency conversion
  - get_news           : Latest news headlines (DuckDuckGo)
  - get_datetime       : Current date/time in any timezone

Observability
-------------
LangSmith tracing is enabled automatically when these env vars are set in .env:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langsmith_api_key
    LANGCHAIN_PROJECT=your_project_name      (optional, default: "ai-chatbot")
"""

import os
import time
import random
from typing import Iterator, Literal

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from memory import get_checkpointer
from state import ChatState
from tools import ALL_TOOLS

load_dotenv()


# ---------------------------------------------------------------------------
# LangSmith Observability
# ---------------------------------------------------------------------------

def _log_langsmith_status() -> None:
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    project = os.getenv("LANGCHAIN_PROJECT", "ai-chatbot")
    api_key = os.getenv("LANGCHAIN_API_KEY", "")

    if tracing == "true" and api_key:
        print("━" * 55)
        print(f"[LangSmith] ✅  Tracing ENABLED")
        print(f"[LangSmith] 📁  Project : {project}")
        print(f"[LangSmith] 🔗  Dashboard: https://smith.langchain.com")
        print("━" * 55)
    elif tracing == "true" and not api_key:
        print("[LangSmith] ⚠️  LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is missing!")
    else:
        print("[LangSmith] ℹ️  Tracing DISABLED — set LANGCHAIN_TRACING_V2=true in .env to enable")


_log_langsmith_status()


# ---------------------------------------------------------------------------
# Rate-limit retry config
# ---------------------------------------------------------------------------

MAX_RETRIES = 4
BASE_DELAY  = 5.0
MAX_DELAY   = 60.0


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "rate_limited" in msg
        or "too many requests" in msg
    )


def _with_retry(fn, *args, **kwargs):
    delay = BASE_DELAY
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise
            jitter = delay * 0.2 * random.random()
            wait = min(delay + jitter, MAX_DELAY)
            print(f"[graph] Rate limited — waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a powerful AI assistant with access to real-time tools.

## Your tools:
- **web_search** — Search the web for any information, current events, or facts
- **get_stock_price** — Get real-time stock prices and market data
- **get_company_info** — Get company fundamentals and business summaries
- **calculator** — Evaluate any mathematical expression safely
- **get_weather** — Get current weather for any city worldwide
- **convert_currency** — Convert between any currencies with live rates
- **get_news** — Get the latest news headlines on any topic
- **get_datetime** — Get current date and time in any timezone

## Guidelines:
- Use tools proactively whenever a question needs current data, calculations, or facts.
- For stock questions, ALWAYS use get_stock_price or get_company_info — do not guess.
- For math, ALWAYS use the calculator tool — do not compute mentally.
- For weather, news, or current events, ALWAYS search rather than estimating.
- After using a tool, summarise the results clearly and helpfully.
- If a tool fails, explain what happened and offer an alternative approach.
- Be concise but thorough. Format responses with markdown for readability.
"""


# ---------------------------------------------------------------------------
# LLM factory – supports OpenAI, Google Gemini, and Mistral
# ---------------------------------------------------------------------------

def _build_llm():
    """
    Instantiate the chat model and bind ALL_TOOLS to it.

    Priority:
        1. OPENAI_API_KEY   → ChatOpenAI        (default: gpt-4o-mini)
        2. GOOGLE_API_KEY   → ChatGoogleGenerativeAI (default: gemini-1.5-flash)
        3. MISTRAL_API_KEY  → ChatMistralAI     (default: mistral-large-latest)
    """
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI  # type: ignore
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, streaming=True, temperature=0.7)

    elif os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        model_name = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
        llm = ChatGoogleGenerativeAI(model=model_name, streaming=True, temperature=0.7)

    elif os.getenv("MISTRAL_API_KEY"):
        from langchain_mistralai import ChatMistralAI  # type: ignore
        model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        llm = ChatMistralAI(model=model_name, streaming=True, temperature=0.7)

    else:
        raise EnvironmentError(
            "No LLM API key found. Set OPENAI_API_KEY, GOOGLE_API_KEY, or MISTRAL_API_KEY in .env"
        )

    # Bind tools so the LLM knows what it can call
    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def agent_node(state: ChatState) -> dict:
    """
    Main agent node — calls the LLM (with tools bound).

    The LLM decides whether to:
      (a) Call a tool  → returns an AIMessage with tool_calls populated
      (b) Answer directly → returns an AIMessage with plain content
    """
    from langchain_core.messages import SystemMessage

    llm = _build_llm()

    # Prepend system message with instructions + tool guidance
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

    response: BaseMessage = _with_retry(llm.invoke, messages_with_system)
    return {"messages": [response]}


def should_continue(state: ChatState) -> Literal["tools", "end"]:
    """
    Conditional edge: check if the last message contains tool calls.
    If yes → run tool_node; if no → end the graph.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder / cache
# ---------------------------------------------------------------------------

def get_graph():
    """
    Return the compiled LangGraph application.

    Graph topology:
        START
          │
        agent_node ──(has tool calls?)──► tool_node
          ▲                                   │
          └───────────────────────────────────┘
          │
        END (when no tool calls)

    Cached in st.session_state so it survives Streamlit reruns.
    """
    if "graph" not in st.session_state:
        checkpointer = get_checkpointer()

        # ToolNode automatically routes to the correct tool by name
        tool_node = ToolNode(ALL_TOOLS)

        builder = StateGraph(ChatState)

        # Nodes
        builder.add_node("agent_node", agent_node)
        builder.add_node("tool_node", tool_node)

        # Edges
        builder.add_edge(START, "agent_node")
        builder.add_conditional_edges(
            "agent_node",
            should_continue,
            {"tools": "tool_node", "end": END},
        )
        builder.add_edge("tool_node", "agent_node")  # loop back after tool execution

        st.session_state["graph"] = builder.compile(checkpointer=checkpointer)

    return st.session_state["graph"]


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def stream_response(user_message: str, thread_id: str) -> Iterator[str]:
    """
    Stream tokens from the LLM for a given user message and thread.

    Handles the ReAct loop internally — the caller just receives the final
    text tokens of the assistant's reply.

    Handles 429 rate-limit errors with exponential backoff.
    """
    from langchain_core.messages import HumanMessage, AIMessageChunk

    graph = get_graph()

    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"chat-{thread_id[:8]}",
        "tags": ["chatbot", "streamlit"],
        "metadata": {"thread_id": thread_id},
    }

    input_state = {"messages": [HumanMessage(content=user_message)]}
    delay = BASE_DELAY

    for attempt in range(MAX_RETRIES + 1):
        try:
            for chunk in graph.stream(input_state, config=config, stream_mode="messages"):
                message_chunk, metadata = chunk

                # Only yield tokens from agent_node (not tool results)
                if (
                    metadata.get("langgraph_node") == "agent_node"
                    and isinstance(message_chunk, AIMessageChunk)
                    and message_chunk.content
                    # Skip chunks that only carry tool_call metadata
                    and not getattr(message_chunk, "tool_call_chunks", None)
                ):
                    yield message_chunk.content
            return

        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise

            jitter = delay * 0.2 * random.random()
            wait = min(delay + jitter, MAX_DELAY)
            wait_int = int(wait)

            yield (
                f"\n\n⏳ *Rate limit hit — retrying in {wait_int}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})…*\n\n"
            )
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY)
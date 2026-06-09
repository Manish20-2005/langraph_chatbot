"""
state.py
Defines the TypedDict state schema used by the LangGraph workflow.
Each conversation node reads from and writes to this shared state object.
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """
    State object for the conversation graph.

    Fields:
        messages: Accumulated list of HumanMessage / AIMessage objects.
                  The add_messages reducer appends new messages on each node call
                  instead of overwriting the list.
    """
    messages: Annotated[list, add_messages]
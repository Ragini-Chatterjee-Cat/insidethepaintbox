"""AgentState schema for the Paintbox LangGraph agent."""
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str           # "artwork" | "general" | "commission"
    user_prefs: dict      # liked series, artworks, tone
    thread_id: str

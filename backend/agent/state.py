"""
AgentState — the shared state object that flows through every node of the
Paintbox LangGraph.

Purpose: defines the single piece of data every node in the graph reads
from and writes back into (LangGraph merges each node's return dict into
this state before calling the next node).

Imported by: agent/graph.py (as the schema passed to StateGraph(AgentState))
and agent/nodes.py (as the type hint on every node method's `state` param).
Not called directly — it's a data shape, not a function.
"""
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """One conversation turn's working state as it moves through the graph."""
    messages: Annotated[list, add_messages]  # full message history, LangGraph appends automatically
    intent: str           # "artwork" | "general" | "commission" — set by the classify node
    user_prefs: dict      # liked series, artworks, tone — loaded/saved by db.py
    thread_id: str        # the conversation's persistent id, used for memory + prefs lookup

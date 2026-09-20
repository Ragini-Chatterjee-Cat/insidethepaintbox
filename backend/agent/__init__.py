"""
Paintbox LangGraph agent package — the public interface to the agent.

Purpose: everything outside agent/ (in practice, just app.py) talks to
the agent only through the three functions in this file. graph.py and
nodes.py are implementation details this module hides.

Imported by: app.py only —
  - chat_with_memory() -> POST /chat/v2
  - get_conversation_history() -> GET /chat/history/{thread_id}
  - clear_conversation() -> DELETE /chat/history/{thread_id}
"""
import logging
from typing import List

from langchain_core.messages import AIMessage, HumanMessage

from .graph import compiled_graph
from .nodes import PREFS_DIR


def chat_with_memory(message: str, thread_id: str) -> str:
    """Run one full turn of the graph for `thread_id` and return the
    assistant's reply text (the last AIMessage the graph produced)."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = compiled_graph.invoke(
            {"messages": [HumanMessage(content=message)], "thread_id": thread_id},
            config=config,
        )
    except Exception as e:
        logging.error(f"Graph invoke failed for thread {thread_id}: {type(e).__name__}: {e}")
        raise
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "I'm sorry, I couldn't generate a response. Please try again."

def get_conversation_history(thread_id: str) -> List[dict]:
    """Read `thread_id`'s message history back out of the graph's
    checkpointer (Postgres or in-memory) as a plain role/content list,
    without running the graph."""
    config = {"configurable": {"thread_id": thread_id}}
    state = compiled_graph.get_state(config)
    if not state or not state.values or "messages" not in state.values:
        return []
    history = []
    for msg in state.values["messages"]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": msg.content})
    return history


def clear_conversation(thread_id: str) -> bool:
    """Delete `thread_id`'s saved preferences file. Note: only clears the
    JSON-fallback prefs file — if POSTGRES_URI is set, prefs live in
    Postgres instead and this is currently a no-op for them, and either
    way the graph's own message history (in the checkpointer) is
    untouched by this call."""
    prefs_file = PREFS_DIR / f"{thread_id}.json"
    if prefs_file.exists():
        prefs_file.unlink()
    return True

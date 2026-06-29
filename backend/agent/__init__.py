"""Paintbox LangGraph agent package."""
# C'est quoi 
from pathlib import Path
# Porquoi pas le list par python
from typing import List

from langchain_core.messages import AIMessage, HumanMessage

from .graph import compiled_graph
from .nodes import PREFS_DIR


def chat_with_memory(message: str, thread_id: str) -> str:
    import logging
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
    prefs_file = PREFS_DIR / f"{thread_id}.json"
    if prefs_file.exists():
        prefs_file.unlink()
    return True

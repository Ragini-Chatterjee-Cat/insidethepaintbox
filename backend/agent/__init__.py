"""Paintbox LangGraph agent package."""
from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, HumanMessage

from .graph import compiled_graph
from .nodes import PREFS_DIR


def chat_with_memory(message: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke(
        {"messages": [HumanMessage(content=message)], "thread_id": thread_id},
        config=config,
    )
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

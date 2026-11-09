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

from langchain_core.messages import AIMessage, HumanMessage

from .graph import compiled_graph


def _text_of(content) -> str:
    """A message's text. LangChain stores it as a plain string, or - when
    Claude wrote text AND called a tool in one reply - as a list of content
    blocks, which used to leak through as a list and crash the history
    endpoint's string-only response model."""
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or block.get("type") == "text"
    )


def chat_with_memory(message: str, thread_id: str, visitor_id: str) -> str:
    """Run one full turn of the graph for `thread_id` and return the
    assistant's reply text (the last AIMessage the graph produced).
    Message history is per thread; preferences are per `visitor_id`, so
    they carry over into the visitor's later conversations."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = compiled_graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "thread_id": thread_id,
                "visitor_id": visitor_id,
            },
            config=config,
        )
    except Exception as e:
        logging.error(f"Graph invoke failed for thread {thread_id}: {type(e).__name__}: {e}")
        raise
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and _text_of(msg.content):
            return _text_of(msg.content)
    return "I'm sorry, I couldn't generate a response. Please try again."

def get_conversation_history(thread_id: str) -> list[dict]:
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
            history.append({"role": "user", "content": _text_of(msg.content)})
        # Skip replies that called a tool: their text was a preamble
        # ("let me look...") the visitor never saw, not an answer.
        elif isinstance(msg, AIMessage) and not msg.tool_calls and _text_of(msg.content):
            history.append({"role": "assistant", "content": _text_of(msg.content)})
    return history


def clear_conversation(thread_id: str) -> bool:
    """Delete `thread_id`'s message history in the checkpointer. The
    visitor's preferences are deliberately kept: they belong to the
    visitor, not to this one conversation."""
    compiled_graph.checkpointer.delete_thread(thread_id)
    return True

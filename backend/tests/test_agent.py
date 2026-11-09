"""Tests for agent/__init__.py - the agent's public interface."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent


def _thread_with_a_tool_turn():
    """What a real checkpoint holds after one searched turn: when Claude
    writes text AND calls a tool in the same reply, LangChain stores that
    message's content as a list of blocks, not a string."""
    return [
        HumanMessage(content="show me something purple"),
        AIMessage(
            content=[
                {"type": "text", "text": "Let me look for purple pieces."},
                {"type": "tool_use", "id": "t1", "name": "search_artworks", "input": {"query": "purple"}},
            ],
            tool_calls=[{"name": "search_artworks", "args": {"query": "purple"}, "id": "t1"}],
        ),
        ToolMessage(content="- HURT (Series: Thoughts)", tool_call_id="t1"),
        AIMessage(content=[{"type": "text", "text": "Hurt is a magenta and purple piece."}]),
    ]


def test_history_handles_messages_stored_as_content_blocks(monkeypatch):
    """Regression test (seen live): GET /chat/history returned a 500 for any
    thread where a reply was stored as a list of content blocks."""
    monkeypatch.setattr(
        agent.compiled_graph, "get_state",
        lambda _config: SimpleNamespace(values={"messages": _thread_with_a_tool_turn()}),
    )

    history = agent.get_conversation_history("t")

    assert history == [
        {"role": "user", "content": "show me something purple"},
        # the "let me look" preamble belongs to a tool call - visitors never saw it
        {"role": "assistant", "content": "Hurt is a magenta and purple piece."},
    ]


def test_chat_with_memory_returns_plain_text_for_block_content(monkeypatch):
    monkeypatch.setattr(
        agent.compiled_graph, "invoke",
        lambda _input, config: {"messages": _thread_with_a_tool_turn()},
    )

    assert agent.chat_with_memory("hi", "t", "v") == "Hurt is a magenta and purple piece."

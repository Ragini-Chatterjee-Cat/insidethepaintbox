"""Tests for agent/nodes.py - PaintboxAgent's node logic.

ChatAnthropic/ToolNode instances are real (construction alone makes no
network call), but every .invoke() call used in these tests is replaced
with a controlled fake, so no test here ever calls a real API.
"""
import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import PaintboxAgent, _trim


def _ai_with_tool_call():
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "search_artworks", "args": {"query": "x"}, "id": "1"}]
    return msg


def test_trim_keeps_only_last_n_messages():
    messages = [HumanMessage(content=str(i)) for i in range(20)]
    trimmed = _trim(messages, n=5)
    assert len(trimmed) == 5
    assert trimmed[0].content == "15"


def test_trim_skips_forward_to_first_human_message():
    """Never start a trimmed window on an orphaned ToolMessage."""
    messages = [
        HumanMessage(content="earlier, cut off"),
        AIMessage(content="tool call response"),
        HumanMessage(content="this should be the start"),
        AIMessage(content="final"),
    ]
    trimmed = _trim(messages, n=3)
    assert trimmed[0].content == "this should be the start"


def test_trim_returns_unchanged_if_no_human_message_found():
    messages = [AIMessage(content="a"), AIMessage(content="b")]
    assert _trim(messages, n=5) == messages


def test_react_node_forces_final_answer_when_loop_never_breaks(monkeypatch):
    """Regression test: if the model calls a tool on every single
    iteration, the loop used to exhaust all 5 rounds ending on tool
    results with no final answer - chat_with_memory() would then search
    backward through the WHOLE conversation and could return a stale
    reply from an earlier turn. The fix forces one more untooled call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = PaintboxAgent()

    # Every call to the tool-bound LLM requests a tool call - never stops
    # on its own. ChatAnthropic/RunnableBinding are pydantic models that
    # reject setting arbitrary attributes like `.invoke` in place, so
    # replace the whole attribute with a mock instead of patching a method
    # on the real object.
    agent._llm_with_tools = MagicMock()
    agent._llm_with_tools.invoke.return_value = _ai_with_tool_call()
    agent._tool_node = MagicMock()
    agent._tool_node.invoke.return_value = {"messages": [AIMessage(content="[tool result]")]}
    final_answer = AIMessage(content="Here's what I found after all that searching.")
    agent._llm_main = MagicMock()
    agent._llm_main.invoke.return_value = final_answer

    state = {"messages": [HumanMessage(content="tell me everything")], "user_prefs": {}}
    result = agent.react_node(state)

    # The tool-bound LLM was called exactly 5 times (the loop cap).
    assert agent._llm_with_tools.invoke.call_count == 5
    # The fallback untooled call fired exactly once.
    agent._llm_main.invoke.assert_called_once()
    # The very last message is the forced final answer, not a tool result.
    assert result["messages"][-1] is final_answer
    assert result["messages"][-1].content


def test_react_node_does_not_force_extra_call_when_model_answers_normally(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = PaintboxAgent()

    normal_answer = AIMessage(content="Icarus is a mythical-series piece about ambition.")
    agent._llm_with_tools = MagicMock()
    agent._llm_with_tools.invoke.return_value = normal_answer
    agent._tool_node = MagicMock()
    agent._llm_main = MagicMock()

    state = {"messages": [HumanMessage(content="tell me about Icarus")], "user_prefs": {}}
    result = agent.react_node(state)

    assert agent._llm_with_tools.invoke.call_count == 1
    agent._llm_main.invoke.assert_not_called()  # the fix's fallback never fires
    assert result["messages"] == [normal_answer]


def test_extract_preferences_keeps_newest_items_when_over_cap(monkeypatch):
    """Regression test: set() has no ordering, so which items survived a
    truncation cap used to be effectively random. dict.fromkeys with the
    new extraction first means this turn's info is what's kept."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = PaintboxAgent()

    existing_series = [f"Series {i}" for i in range(10)]
    extracted_payload = json.dumps({
        "liked_series": ["Brand New Series"],
        "mentioned_artworks": [],
        "tone": "curious",
    })
    agent._llm_extractor = MagicMock()
    agent._llm_extractor.invoke.return_value = AIMessage(content=extracted_payload)

    state = {
        "messages": [HumanMessage(content="I love this new series")],
        "user_prefs": {"liked_series": existing_series, "mentioned_artworks": [], "tone": ""},
    }
    result = agent.extract_preferences(state)

    liked = result["user_prefs"]["liked_series"]
    assert "Brand New Series" in liked  # the just-mentioned item survived the cap
    assert liked[0] == "Brand New Series"  # newest first
    assert len(liked) == 10  # still capped


def test_extract_preferences_returns_empty_dict_on_invalid_json(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = PaintboxAgent()
    agent._llm_extractor = MagicMock()
    agent._llm_extractor.invoke.return_value = AIMessage(content="not valid json")

    state = {"messages": [HumanMessage(content="hi")], "user_prefs": {}}
    assert agent.extract_preferences(state) == {}


def test_extract_preferences_returns_empty_dict_on_valid_but_non_dict_json(monkeypatch):
    """Regression test: `null` or `[]` parse successfully as JSON, so the
    old except (JSONDecodeError, ValueError) block never caught them - the
    next line's extracted.get(...) would then raise AttributeError and
    500 the whole turn, even though a perfectly good reply already exists."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = PaintboxAgent()
    agent._llm_extractor = MagicMock()

    for bad_payload in ("null", "[]", '"just a string"'):
        agent._llm_extractor.invoke.return_value = AIMessage(content=bad_payload)
        state = {"messages": [HumanMessage(content="hi")], "user_prefs": {}}
        assert agent.extract_preferences(state) == {}


def test_extract_preferences_preserves_tone_when_none_detected(monkeypatch):
    """Regression test: extracted.get("tone", existing_tone) never fell
    back to existing_tone in practice, because EXTRACT_SYSTEM always
    includes a "tone" key (as "" when none is found) - .get()'s default
    only applies when the key is missing, not when it's present-but-falsy.
    A turn with no detected tone was silently wiping the saved one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = PaintboxAgent()
    agent._llm_extractor = MagicMock()
    agent._llm_extractor.invoke.return_value = AIMessage(content=json.dumps({
        "liked_series": [], "mentioned_artworks": [], "tone": "",
    }))

    state = {
        "messages": [HumanMessage(content="ok show me more")],
        "user_prefs": {"liked_series": [], "mentioned_artworks": [], "tone": "enthusiastic"},
    }
    result = agent.extract_preferences(state)

    assert result["user_prefs"]["tone"] == "enthusiastic"


def test_route_intent_maps_each_intent_to_its_node():
    agent = PaintboxAgent.__new__(PaintboxAgent)  # no __init__, route_intent needs no LLMs
    assert agent.route_intent({"intent": "commission"}) == "commission_intake"
    assert agent.route_intent({"intent": "artwork"}) == "react_node"
    assert agent.route_intent({"intent": "general"}) == "general_chat"
    assert agent.route_intent({}) == "general_chat"  # missing intent defaults safely

"""
PaintboxAgent — all LangGraph node implementations for the Paintbox agent.

Purpose: every step the agent can take (load prefs, classify intent, run
the tool-calling gallery guide, chat plainly, hand off to commissions,
extract preferences, save them) is a method on PaintboxAgent here. This
is where the actual LLM calls happen — agent/graph.py only wires these
methods into a state machine, it contains no logic of its own.

Imported by: agent/graph.py only. build_graph() instantiates
PaintboxAgent() once and registers each of its methods below as a node
in the compiled graph; LangGraph calls them itself as the graph runs,
nothing here is called directly by app.py or any other module.
"""
import json
import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

import db
from tools import ARTWORK_TOOLS

from .prompts import (
    CLASSIFY_SYSTEM,
    EXTRACT_SYSTEM,
    GALLERY_SYSTEM,
    GENERAL_SYSTEM,
)
from .state import AgentState

PREFS_DIR = Path(os.environ.get("PREFS_DIR", "./user_prefs"))
PREFS_DIR.mkdir(parents=True, exist_ok=True)

MAX_HISTORY = 10


def _trim(messages: list, n: int = MAX_HISTORY) -> list:
    """Keep only the last `n` messages, then skip forward to the first
    HumanMessage in that slice — never let a trimmed list start on an
    orphaned ToolMessage with no preceding tool call for context."""
    trimmed = messages[-n:] if len(messages) > n else messages
    for i, msg in enumerate(trimmed):
        if isinstance(msg, HumanMessage):
            return trimmed[i:]
    return trimmed


class PaintboxAgent:
    """Encapsulates all graph nodes and their shared LLM instances. One
    instance is created per process (in build_graph()) and reused for
    every conversation turn that process handles."""

    def __init__(self):
        """Build the four Haiku clients this agent uses (one per node
        that calls the model, each with its own max_tokens ceiling) plus
        the tool-calling variant and its LangGraph ToolNode wrapper."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set!")

        def _make_llm(max_tokens: int) -> ChatAnthropic:
            """Build one Claude Haiku client pinned to `max_tokens` output."""
            return ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=api_key,
                temperature=0,
                max_tokens=max_tokens,
                max_retries=3,
            )

        self._llm_classifier = _make_llm(20)
        self._llm_main       = _make_llm(1024)
        self._llm_chat       = _make_llm(512)
        self._llm_extractor  = _make_llm(512)
        self._llm_with_tools = self._llm_main.bind_tools(ARTWORK_TOOLS)
        self._tool_node      = ToolNode(ARTWORK_TOOLS, handle_tool_errors=True)

    # -----------------------------------------------------------------------
    # Node: load_preferences
    # -----------------------------------------------------------------------

    def load_preferences(self, state: AgentState) -> dict:
        """First node in the graph. Loads this thread's saved prefs (via
        db.py) into state so later nodes can personalize the reply."""
        thread_id = state.get("thread_id", "")
        prefs = db.load_user_prefs(thread_id, PREFS_DIR) if thread_id else {}
        return {"user_prefs": prefs}

    # -----------------------------------------------------------------------
    # Node: classify
    # -----------------------------------------------------------------------

    def classify(self, state: AgentState) -> dict:
        """Ask the cheap classifier LLM to label the latest visitor message
        as "artwork", "commission", or "general" — route_intent() below
        uses this to pick the next node."""
        messages = _trim(state["messages"])
        last_user = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        result = self._llm_classifier.invoke([
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=last_user),
        ])
        intent = result.content.strip().lower()
        if "commission" in intent:
            return {"intent": "commission"}
        if "artwork" in intent:
            return {"intent": "artwork"}
        return {"intent": "general"}

    # -----------------------------------------------------------------------
    # Node: react_node
    # -----------------------------------------------------------------------

    def react_node(self, state: AgentState) -> dict:
        """Entered when classify() said "artwork". Runs a ReAct-style
        tool-calling loop (up to 5 iterations) against ARTWORK_TOOLS,
        injecting known visitor preferences as extra context, until the
        model answers without requesting another tool call."""
        messages = [GALLERY_SYSTEM] + _trim(list(state["messages"]))

        prefs = state.get("user_prefs", {})
        if prefs:
            pref_lines = []
            if prefs.get("liked_series"):
                pref_lines.append(f"Visitor has shown interest in: {', '.join(prefs['liked_series'])}")
            if prefs.get("mentioned_artworks"):
                pref_lines.append(f"Previously discussed artworks: {', '.join(prefs['mentioned_artworks'])}")
            if pref_lines:
                messages.insert(1, SystemMessage(content="User context: " + ". ".join(pref_lines)))

        new_messages = []
        for _ in range(5):
            response = self._llm_with_tools.invoke(messages + new_messages)
            new_messages.append(response)
            if not getattr(response, "tool_calls", None):
                break
            tool_results = self._tool_node.invoke({"messages": messages + new_messages})
            new_messages.extend(tool_results["messages"])
        else:
            # Loop exhausted without ever getting a tool-call-free response —
            # every iteration called a tool, so new_messages ends on tool
            # results, not an answer. Force one more call, without tools,
            # so the visitor gets a real reply instead of chat_with_memory()
            # falling back to a stale AIMessage from earlier in the thread.
            final = self._llm_main.invoke(messages + new_messages)
            new_messages.append(final)

        return {"messages": new_messages}

    # -----------------------------------------------------------------------
    # Node: general_chat
    # -----------------------------------------------------------------------

    def general_chat(self, state: AgentState) -> dict:
        """Entered when classify() said "general". A single plain Haiku
        reply for small talk or off-topic messages — no tools involved."""
        messages = [GENERAL_SYSTEM] + _trim(list(state["messages"]))
        response = self._llm_chat.invoke(messages)
        return {"messages": [response]}

    # -----------------------------------------------------------------------
    # Node: commission_intake
    # -----------------------------------------------------------------------

    def commission_intake(self, state: AgentState) -> dict:
        """Entered when classify() said "commission". No LLM call — just
        a fixed reply pointing the visitor at the commissions page."""
        reply = AIMessage(content=(
            "I'd love to help you get a commission started! You can fill in all the details "
            "on Ragini's commissions page: "
            "https://insidethepaintbox.netlify.app/pages/commissions.html"
        ))
        return {"messages": [reply]}

    # -----------------------------------------------------------------------
    # Node: extract_preferences
    # -----------------------------------------------------------------------

    def extract_preferences(self, state: AgentState) -> dict:
        """Runs after every branch (react_node/general_chat/
        commission_intake) converges. Asks the extractor LLM to pull any
        liked series, mentioned artworks, or tone out of the last few
        turns, and merges them into the existing prefs dict."""
        messages = state["messages"]
        recent = messages[-4:]
        conversation_text = "\n".join(
            f"{type(m).__name__}: {m.content}"
            for m in recent
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        )

        if not conversation_text.strip():
            return {}

        result = self._llm_extractor.invoke([
            SystemMessage(content=EXTRACT_SYSTEM),
            HumanMessage(content=conversation_text),
        ])

        try:
            extracted = json.loads(result.content.strip())
        except (json.JSONDecodeError, ValueError):
            return {}

        if not isinstance(extracted, dict):
            # The extractor is asked for a JSON object but nothing enforces
            # it structurally - valid JSON like `null` or `[]` would parse
            # fine and then crash the next line's .get() calls, which
            # would 500 the whole turn even though react_node/general_chat
            # already produced a perfectly good reply for the visitor.
            return {}

        # dict.fromkeys dedupes while preserving order; putting this turn's
        # extraction first means the newest items are the ones kept once
        # the [:N] cap below is applied, rather than an arbitrary subset
        # (set() has no ordering, so which items survived a cap was random).
        existing = state.get("user_prefs", {})
        liked_series = list(dict.fromkeys(
            extracted.get("liked_series", []) + existing.get("liked_series", [])
        ))
        mentioned = list(dict.fromkeys(
            extracted.get("mentioned_artworks", []) + existing.get("mentioned_artworks", [])
        ))
        # `or`, not .get(..., default): EXTRACT_SYSTEM always includes a
        # "tone" key (empty string when none is found), so .get()'s default
        # never actually applied - every turn where the extractor didn't
        # detect a new tone was silently overwriting a previously saved one
        # with "".
        tone = extracted.get("tone") or existing.get("tone", "")

        return {"user_prefs": {
            "liked_series": liked_series[:10],
            "mentioned_artworks": mentioned[:20],
            "tone": tone,
        }}

    # -----------------------------------------------------------------------
    # Node: save_preferences
    # -----------------------------------------------------------------------

    def save_preferences(self, state: AgentState) -> dict:
        """Last node before END. Persists the (possibly updated) prefs
        dict via db.py so the next turn's load_preferences() sees it."""
        thread_id = state.get("thread_id", "")
        prefs = state.get("user_prefs", {})
        if thread_id and prefs:
            db.save_user_prefs(thread_id, prefs, PREFS_DIR)
        return {}

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------

    def route_intent(self, state: AgentState) -> str:
        """Conditional-edge function registered on the "classify" node in
        agent/graph.py — maps state["intent"] to the name of the next
        node to run."""
        intent = state.get("intent", "general")
        if intent == "commission":
            return "commission_intake"
        if intent == "artwork":
            return "react_node"
        return "general_chat"

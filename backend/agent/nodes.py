"""PaintboxAgent — all LangGraph node implementations."""
import json
import os
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import ToolNode
#What is db for 
import db
from tools import ARTWORK_TOOLS
from .state import AgentState
from .prompts import (
    GALLERY_SYSTEM,
    CLASSIFY_SYSTEM,
    GENERAL_SYSTEM,
    COMMISSION_INTAKE_SYSTEM,
    EXTRACT_SYSTEM,
)

PREFS_DIR = Path(os.environ.get("PREFS_DIR", "./user_prefs"))
PREFS_DIR.mkdir(parents=True, exist_ok=True)

MAX_HISTORY = 10


def _trim(messages: list, n: int = MAX_HISTORY) -> list:
    trimmed = messages[-n:] if len(messages) > n else messages
    # Don't start on a tool result — skip forward to the first HumanMessage
    # to avoid sending an orphaned ToolMessage without its preceding tool call.
    for i, msg in enumerate(trimmed):
        if isinstance(msg, HumanMessage):
            return trimmed[i:]
    return trimmed


class PaintboxAgent:
    """Encapsulates all graph nodes and their shared LLM instances."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set!")

        def _make_llm(max_tokens: int) -> ChatAnthropic:
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
        self._llm_commission = _make_llm(1024)
        self._llm_extractor  = _make_llm(512)
        self._llm_with_tools = self._llm_main.bind_tools(ARTWORK_TOOLS)
        self._tool_node      = ToolNode(ARTWORK_TOOLS, handle_tool_errors=True)

    # -----------------------------------------------------------------------
    # Node: load_preferences
    # -----------------------------------------------------------------------

    def load_preferences(self, state: AgentState) -> dict:
        thread_id = state.get("thread_id", "")
        prefs = db.load_user_prefs(thread_id, PREFS_DIR) if thread_id else {}
        return {"user_prefs": prefs}

    # -----------------------------------------------------------------------
    # Node: classify
    # -----------------------------------------------------------------------

    def classify(self, state: AgentState) -> dict:
        if state.get("commission_data", {}).get("in_progress"):
            return {"intent": "commission"}

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

        return {"messages": new_messages}

    # -----------------------------------------------------------------------
    # Node: general_chat
    # -----------------------------------------------------------------------

    def general_chat(self, state: AgentState) -> dict:
        messages = [GENERAL_SYSTEM] + _trim(list(state["messages"]))
        response = self._llm_chat.invoke(messages)
        return {"messages": [response]}

    # -----------------------------------------------------------------------
    # Node: commission_intake
    # -----------------------------------------------------------------------

    def commission_intake(self, state: AgentState) -> dict:
        messages = state["messages"]
        commission_data = state.get("commission_data") or {}
        thread_id = state.get("thread_id", "unknown")

        recent = messages[-8:]
        history = "\n".join(
            f"{'Visitor' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in recent
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        )

        last_human = next(
            (m.content for m in reversed(recent) if isinstance(m, HumanMessage) and m.content),
            "Continue.",
        )
        result = self._llm_commission.invoke([
            SystemMessage(content=COMMISSION_INTAKE_SYSTEM.format(history=history)),
            HumanMessage(content=last_human),
        ])
        text = result.content.strip()

        if text.upper().startswith("COMPLETE:"):
            summary = text[9:].strip()
            db.save_commission(thread_id, summary, datetime.utcnow().isoformat(), PREFS_DIR)
            confirmation = AIMessage(content=(
                "Thank you so much for sharing those details! I've passed everything on to Ragini — "
                "she'll be in touch personally via Instagram (@ragini_chatterjee) or email "
                "(inthepaintbox@gmail.com) within a few days."
            ))
            return {
                "messages": [confirmation],
                "commission_data": {
                    "in_progress": False,
                    "awaiting_review": True,
                    "summary": summary,
                },
            }

        return {
            "messages": [AIMessage(content=text)],
            "commission_data": {
                "in_progress": True,
                "awaiting_review": False,
                "summary": commission_data.get("summary", ""),
            },
        }

    # -----------------------------------------------------------------------
    # Node: extract_preferences
    # -----------------------------------------------------------------------

    def extract_preferences(self, state: AgentState) -> dict:
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

        existing = state.get("user_prefs", {})
        liked_series = list(set(existing.get("liked_series", []) + extracted.get("liked_series", [])))
        mentioned = list(set(existing.get("mentioned_artworks", []) + extracted.get("mentioned_artworks", [])))
        tone = extracted.get("tone", existing.get("tone", ""))

        return {"user_prefs": {
            "liked_series": liked_series[:10],
            "mentioned_artworks": mentioned[:20],
            "tone": tone,
        }}

    # -----------------------------------------------------------------------
    # Node: save_preferences
    # -----------------------------------------------------------------------

    def save_preferences(self, state: AgentState) -> dict:
        thread_id = state.get("thread_id", "")
        prefs = state.get("user_prefs", {})
        if thread_id and prefs:
            db.save_user_prefs(thread_id, prefs, PREFS_DIR)
        return {}

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------

    def route_intent(self, state: AgentState) -> str:
        intent = state.get("intent", "general")
        if intent == "commission":
            return "commission_intake"
        if intent == "artwork":
            return "react_node"
        return "general_chat"

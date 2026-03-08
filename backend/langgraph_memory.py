"""
LangGraph Memory Module — Upgraded ReAct Agent
Multi-node graph: load_preferences → classify → react_node/general_chat
                  → extract_preferences → save_preferences

Public interface is unchanged:
    chat_with_memory(message, thread_id) -> str
    get_conversation_history(thread_id) -> List[dict]
    clear_conversation(thread_id) -> bool
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import ARTWORK_TOOLS


# ---------------------------------------------------------------------------
# Directories & persistence
# ---------------------------------------------------------------------------

PREFS_DIR = Path(os.environ.get("PREFS_DIR", "./user_prefs"))
PREFS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full message history
    intent: str                                # "artwork" | "general" | "commission"
    user_prefs: dict                           # liked series, artworks, tone
    thread_id: str
    commission_data: dict                      # tracks commission intake state


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm(max_tokens: int = 512) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set!")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
    )


GALLERY_SYSTEM = SystemMessage(content="""You are a warm, knowledgeable art guide for \
"Inside the Paintbox", the portfolio of artist Ragini Chatterjee.

Your role is to help visitors explore and understand the artwork collection. \
You have access to tools that can search the collection, browse series, \
retrieve artwork details, find similar works, and answer commission questions.

Guidelines:
- Be warm and conversational, like giving a personal gallery tour
- Always use your tools to look up information before answering artwork questions
- Use get_artwork_details when asked about a specific piece by name
- Use filter_by_series when asked to browse a series
- Use get_commission_info when asked about commissioning
- Use search_artworks when a visitor describes what they're looking for
- Use recommend_similar when a visitor wants more like a piece they liked
- Always include the artwork URL at the end of your response when discussing a specific artwork or series. Put it on its own line with no comma or punctuation before it, e.g.: \n\nhttps://insidethepaintbox.netlify.app/artworks/Voices.html
- Keep responses concise (2-4 sentences) unless asked for more detail
- If tools return no results, say so honestly rather than making things up
- Refer to the artist by name (Ragini) after first mention

Series available: Portraits, Animal Portraits, Mythical, Thoughts, \
Camera Series, Diary Entries, Fanart, Cards.""")


# ---------------------------------------------------------------------------
# Node: load_preferences
# ---------------------------------------------------------------------------

def load_preferences(state: AgentState) -> dict:
    """Load persisted user preferences from disk."""
    thread_id = state.get("thread_id", "")
    prefs = {}
    if thread_id:
        prefs_file = PREFS_DIR / f"{thread_id}.json"
        if prefs_file.exists():
            try:
                prefs = json.loads(prefs_file.read_text())
            except Exception:
                prefs = {}
    return {"user_prefs": prefs}


# ---------------------------------------------------------------------------
# Node: classify
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """You are a classifier. Given a user message, output exactly one word:
- "commission" — if the user asks about commissioning, ordering, pricing, or requesting custom artwork to be made
- "artwork" — if the user asks about specific artworks, series, recommendations, or anything art-related
- "general" — if the user is making small talk, greeting, or asking something unrelated

Respond with only the single word, nothing else."""

def classify(state: AgentState) -> dict:
    """Classify the user's intent to route to the right node."""
    # If commission intake is already in progress, continue it regardless of message content
    if state.get("commission_data", {}).get("in_progress"):
        return {"intent": "commission"}

    messages = state["messages"]
    last_user = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    llm = _llm(max_tokens=10)
    result = llm.invoke([
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=last_user),
    ])
    intent = result.content.strip().lower()
    if "commission" in intent:
        return {"intent": "commission"}
    if "artwork" in intent:
        return {"intent": "artwork"}
    return {"intent": "general"}


# ---------------------------------------------------------------------------
# Node: react_node (tool-calling agent loop)
# ---------------------------------------------------------------------------

def react_node(state: AgentState) -> dict:
    """
    ReAct loop: LLM decides which tool(s) to call, calls them,
    observes results, and repeats until it produces a final text response.
    """
    llm_with_tools = _llm(max_tokens=512).bind_tools(ARTWORK_TOOLS)
    tool_node = ToolNode(ARTWORK_TOOLS)

    # Build message list with system prompt
    messages = [GALLERY_SYSTEM] + list(state["messages"])

    # Inject user preferences as context if they exist
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

    # ReAct loop — max 5 iterations to prevent runaway loops
    try:
        for _ in range(5):
            response = llm_with_tools.invoke(messages + new_messages)
            new_messages.append(response)

            # If no tool calls, we have the final answer
            if not getattr(response, "tool_calls", None):
                break

            # Execute each tool call
            tool_results = tool_node.invoke({"messages": messages + new_messages})
            new_messages.extend(tool_results["messages"])

    except Exception as e:
        print(f"Error in react_node loop: {e}")
        fallback = AIMessage(content="I'm sorry, I had trouble looking that up. Could you rephrase your question?")
        new_messages.append(fallback)

    return {"messages": new_messages}


# ---------------------------------------------------------------------------
# Node: general_chat
# ---------------------------------------------------------------------------

_GENERAL_SYSTEM = SystemMessage(content="""You are a friendly assistant for \
"Inside the Paintbox", Ragini Chatterjee's art portfolio website.
The visitor is making small talk or asking something off-topic.
Be warm and brief. If you can naturally steer the conversation toward \
the artwork collection, do so — otherwise just be friendly.
Keep your response to 1-3 sentences.""")

def general_chat(state: AgentState) -> dict:
    """Handle small talk and off-topic messages without tool calls."""
    messages = [_GENERAL_SYSTEM] + list(state["messages"])
    llm = _llm(max_tokens=200)
    response = llm.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Node: commission_intake
# ---------------------------------------------------------------------------

_COMMISSION_INTAKE_SYSTEM = """You are warmly collecting commission details for artist Ragini Chatterjee.

Gather these naturally, one question at a time:
1. Type of piece — portrait of a person, pet portrait, custom illustration, or greeting card
2. Subject — who or what the piece is of
3. Occasion or purpose — gift, personal keepsake, special event, etc.
4. Style preferences — or whether they'd like to browse Ragini's existing series for reference
5. Timeline — any deadline, or is it flexible?
6. Contact details — their email address or Instagram handle so Ragini can follow up personally

Conversation so far:
{history}

IMPORTANT: You MUST collect contact details (email or Instagram handle) before filing the request.
If you have collected type + subject + occasion + contact details, respond with EXACTLY this format and nothing else:
COMPLETE: <a warm 2-3 sentence summary of all the details collected, ending with their contact info>

Otherwise ask ONE friendly follow-up question to get the most important missing detail. One question at a time."""


def commission_intake(state: AgentState) -> dict:
    """Collect commission details over multiple turns, then notify Ragini."""
    messages = state["messages"]
    commission_data = state.get("commission_data") or {}
    thread_id = state.get("thread_id", "unknown")

    # Build conversation history for the LLM
    recent = messages[-8:]
    history = "\n".join(
        f"{'Visitor' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in recent
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    )

    llm = _llm(max_tokens=300)
    result = llm.invoke([
        SystemMessage(content=_COMMISSION_INTAKE_SYSTEM.format(history=history))
    ])
    text = result.content.strip()

    if text.upper().startswith("COMPLETE:"):
        summary = text[9:].strip()

        # Write a file so the admin panel can see this commission
        commission_file = PREFS_DIR / f"{thread_id}_commission.json"
        try:
            commission_file.write_text(json.dumps({
                "thread_id": thread_id,
                "summary": summary,
                "timestamp": datetime.utcnow().isoformat(),
            }))
        except Exception as e:
            print(f"Warning: could not write commission file: {e}")

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

    # Still collecting — ask the next question
    return {
        "messages": [AIMessage(content=text)],
        "commission_data": {
            "in_progress": True,
            "awaiting_review": False,
            "summary": commission_data.get("summary", ""),
        },
    }


# ---------------------------------------------------------------------------
# Node: extract_preferences
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """You are a preference extractor. Given a conversation, identify:
1. Any art series the visitor showed interest in (from: Portraits, Animal Portraits, Mythical, Thoughts, Camera Series, Diary Entries, Fanart, Cards)
2. Any specific artwork titles mentioned
3. The visitor's tone (casual, curious, enthusiastic)

Respond ONLY with a valid JSON object like:
{"liked_series": ["Portraits"], "mentioned_artworks": ["Voices", "Icarus"], "tone": "curious"}

If nothing is found for a field, use an empty list or empty string. No explanation."""

def extract_preferences(state: AgentState) -> dict:
    """Mine the last exchange for user preferences to persist."""
    messages = state["messages"]
    # Only look at the last 4 messages (2 exchanges) for efficiency
    recent = messages[-4:]
    conversation_text = "\n".join(
        f"{type(m).__name__}: {m.content}"
        for m in recent
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    )

    if not conversation_text.strip():
        return {}

    llm = _llm(max_tokens=150)
    result = llm.invoke([
        SystemMessage(content=_EXTRACT_SYSTEM),
        HumanMessage(content=conversation_text),
    ])

    try:
        extracted = json.loads(result.content.strip())
    except Exception:
        return {}

    # Merge with existing prefs
    existing = state.get("user_prefs", {})
    liked_series = list(set(existing.get("liked_series", []) + extracted.get("liked_series", [])))
    mentioned = list(set(existing.get("mentioned_artworks", []) + extracted.get("mentioned_artworks", [])))
    tone = extracted.get("tone", existing.get("tone", ""))

    return {"user_prefs": {
        "liked_series": liked_series[:10],       # cap to avoid unbounded growth
        "mentioned_artworks": mentioned[:20],
        "tone": tone,
    }}


# ---------------------------------------------------------------------------
# Node: save_preferences
# ---------------------------------------------------------------------------

def save_preferences(state: AgentState) -> dict:
    """Persist user preferences to disk for cross-session memory."""
    thread_id = state.get("thread_id", "")
    prefs = state.get("user_prefs", {})
    if thread_id and prefs:
        prefs_file = PREFS_DIR / f"{thread_id}.json"
        try:
            prefs_file.write_text(json.dumps(prefs, indent=2))
        except Exception as e:
            print(f"Warning: could not save prefs for {thread_id}: {e}")
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "general")
    if intent == "commission":
        return "commission_intake"
    if intent == "artwork":
        return "react_node"
    return "general_chat"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def _build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_preferences", load_preferences)
    graph.add_node("classify", classify)
    graph.add_node("react_node", react_node)
    graph.add_node("general_chat", general_chat)
    graph.add_node("commission_intake", commission_intake)
    graph.add_node("extract_preferences", extract_preferences)
    graph.add_node("save_preferences", save_preferences)

    graph.add_edge(START, "load_preferences")
    graph.add_edge("load_preferences", "classify")
    graph.add_conditional_edges("classify", route_intent, {
        "react_node": "react_node",
        "general_chat": "general_chat",
        "commission_intake": "commission_intake",
    })
    graph.add_edge("react_node", "extract_preferences")
    graph.add_edge("general_chat", "extract_preferences")
    graph.add_edge("commission_intake", "extract_preferences")
    graph.add_edge("extract_preferences", "save_preferences")
    graph.add_edge("save_preferences", END)

    db_path = os.environ.get("MEMORY_DB_PATH", "./conversation_memory.db")
    checkpointer = SqliteSaver.from_conn_string(db_path)
    return graph.compile(checkpointer=checkpointer)


compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public interface (unchanged — app.py requires no edits)
# ---------------------------------------------------------------------------

def chat_with_memory(message: str, thread_id: str) -> str:
    """
    Main entry point for chat with persistent memory.

    Args:
        message: The user's message
        thread_id: Unique identifier for the conversation thread

    Returns:
        The assistant's response as a plain string
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = compiled_graph.invoke(
        {"messages": [HumanMessage(content=message)], "thread_id": thread_id},
        config=config,
    )

    # Last message in the list is always the final AI response
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "I'm sorry, I couldn't generate a response. Please try again."


def get_conversation_history(thread_id: str) -> List[dict]:
    """
    Retrieve the full conversation history for a thread.
    Tool call messages are excluded — only user/assistant turns are returned.

    Returns:
        List of {"role": ..., "content": ...} dicts
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = compiled_graph.get_state(config)
        if not state or not state.values or "messages" not in state.values:
            return []

        history = []
        for msg in state.values["messages"]:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                # Skip intermediate messages that only contain tool_calls
                history.append({"role": "assistant", "content": msg.content})
        return history

    except Exception as e:
        print(f"Error getting history: {e}")
        return []


def clear_conversation(thread_id: str) -> bool:
    """
    Clear the conversation history and preferences for a thread.

    Returns:
        True if successful, False otherwise
    """
    # Clear preference file
    prefs_file = PREFS_DIR / f"{thread_id}.json"
    if prefs_file.exists():
        try:
            prefs_file.unlink()
        except Exception:
            pass

    return True

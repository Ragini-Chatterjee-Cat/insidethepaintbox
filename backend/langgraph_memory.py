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
import sqlite3
from pathlib import Path
from typing import Annotated, List, TypedDict

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

DB_PATH = os.environ.get("MEMORY_DB_PATH", "./conversation_memory.db")
PREFS_DIR = Path(os.environ.get("PREFS_DIR", "./user_prefs"))
PREFS_DIR.mkdir(parents=True, exist_ok=True)

sqlite_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory_saver = SqliteSaver(sqlite_conn)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full message history
    intent: str                                # "artwork" | "general"
    user_prefs: dict                           # liked series, artworks, tone
    thread_id: str


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
- "artwork" — if the user asks about specific artworks, series, commissions, recommendations, or anything art-related
- "general" — if the user is making small talk, greeting, or asking something unrelated to the art collection

Respond with only the single word, nothing else."""

def classify(state: AgentState) -> dict:
    """Classify the user's intent to route to the right node."""
    messages = state["messages"]
    last_user = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    llm = _llm(max_tokens=5)
    result = llm.invoke([
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=last_user),
    ])
    intent = result.content.strip().lower()
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
    return "react_node" if state.get("intent") == "artwork" else "general_chat"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def _build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_preferences", load_preferences)
    graph.add_node("classify", classify)
    graph.add_node("react_node", react_node)
    graph.add_node("general_chat", general_chat)
    graph.add_node("extract_preferences", extract_preferences)
    graph.add_node("save_preferences", save_preferences)

    graph.add_edge(START, "load_preferences")
    graph.add_edge("load_preferences", "classify")
    graph.add_conditional_edges("classify", route_intent, {
        "react_node": "react_node",
        "general_chat": "general_chat",
    })
    graph.add_edge("react_node", "extract_preferences")
    graph.add_edge("general_chat", "extract_preferences")
    graph.add_edge("extract_preferences", "save_preferences")
    graph.add_edge("save_preferences", END)

    return graph.compile(checkpointer=memory_saver)


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
    try:
        # Try the modern API first
        memory_saver.delete_thread(thread_id)
    except AttributeError:
        # Fallback: manual SQL delete
        try:
            cursor = sqlite_conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            sqlite_conn.commit()
        except Exception as e:
            print(f"Error clearing conversation: {e}")
            return False
    except Exception as e:
        print(f"Error clearing conversation: {e}")
        return False

    # Also clear preference file
    prefs_file = PREFS_DIR / f"{thread_id}.json"
    if prefs_file.exists():
        try:
            prefs_file.unlink()
        except Exception:
            pass

    return True

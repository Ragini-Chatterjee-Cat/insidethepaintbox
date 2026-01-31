"""
LangGraph Memory Module
Handles persistent conversation state using LangGraph with SQLite checkpointing
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

from rag import query_rag


class ConversationState(TypedDict):
    """State schema for the conversation graph"""
    messages: Annotated[list, add_messages]  # Auto-accumulates messages
    current_response: str


# Initialize SQLite checkpointer for persistence
DB_PATH = os.environ.get("MEMORY_DB_PATH", "./conversation_memory.db")
sqlite_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory_saver = SqliteSaver(sqlite_conn)


def process_message(state: ConversationState) -> ConversationState:
    """
    Process the user's message and generate a response.
    Uses the RAG pipeline with conversation history from state.
    """
    messages = state["messages"]

    # Get the latest user message
    user_message = messages[-1].content if messages else ""

    # Convert LangGraph messages to history format for RAG
    history = []
    for msg in messages[:-1]:  # Exclude current message
        role = "user" if msg.type == "human" else "assistant"
        history.append({"role": role, "content": msg.content})

    # Call existing RAG pipeline
    response = query_rag(user_message, history)

    return {
        "messages": [{"role": "assistant", "content": response}],
        "current_response": response
    }


def build_conversation_graph() -> StateGraph:
    """Build the LangGraph conversation graph"""
    builder = StateGraph(ConversationState)

    # Add the processing node
    builder.add_node("process", process_message)

    # Define the flow
    builder.add_edge(START, "process")
    builder.add_edge("process", END)

    return builder.compile(checkpointer=memory_saver)


# Create the compiled graph
conversation_graph = build_conversation_graph()


def chat_with_memory(message: str, thread_id: str) -> str:
    """
    Main entry point for chat with persistent memory.

    Args:
        message: The user's message
        thread_id: Unique identifier for the conversation thread
                   (e.g., user session ID, stored in localStorage)

    Returns:
        The assistant's response
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the graph with the user's message
    result = conversation_graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )

    return result["current_response"]


def get_conversation_history(thread_id: str) -> List[dict]:
    """
    Retrieve the full conversation history for a thread.

    Args:
        thread_id: The conversation thread ID

    Returns:
        List of message dictionaries with role and content
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = conversation_graph.get_state(config)
        if state and state.values and "messages" in state.values:
            history = []
            for msg in state.values["messages"]:
                role = "user" if msg.type == "human" else "assistant"
                history.append({"role": role, "content": msg.content})
            return history
    except Exception as e:
        print(f"Error getting history: {e}")

    return []


def clear_conversation(thread_id: str) -> bool:
    """
    Clear the conversation history for a thread.

    Args:
        thread_id: The conversation thread ID

    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete checkpoints for this thread
        cursor = sqlite_conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        sqlite_conn.commit()
        return True
    except Exception as e:
        print(f"Error clearing conversation: {e}")
        return False

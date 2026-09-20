"""
Builds and compiles the Paintbox LangGraph — the agent's node graph and
persistence layer.

Purpose: wires PaintboxAgent's methods (agent/nodes.py) into a state
machine (load_preferences -> classify -> [react_node | general_chat |
commission_intake] -> extract_preferences -> save_preferences -> END)
and attaches a checkpointer so conversation state survives between
requests.

Imported by: agent/__init__.py only, which imports `compiled_graph` and
calls `.invoke()`/`.get_state()` on it per chat request. build_graph()
itself runs exactly once, at import time (see the module-level call at
the bottom of this file) — nothing calls it again after that.
"""
import os

from langgraph.graph import END, START, StateGraph

from .nodes import PaintboxAgent
from .state import AgentState


def build_graph():
    """Construct and compile the graph: register every PaintboxAgent
    node/edge, attach a Postgres checkpointer if POSTGRES_URI is set
    (falling back to in-memory otherwise), and return the compiled graph."""
    agent = PaintboxAgent()
    graph = StateGraph(AgentState)

    # --- nodes ---------------------------------------------------------
    graph.add_node("load_preferences",    agent.load_preferences)
    graph.add_node("classify",            agent.classify)
    graph.add_node("react_node",          agent.react_node)
    graph.add_node("general_chat",        agent.general_chat)
    graph.add_node("commission_intake",   agent.commission_intake)
    graph.add_node("extract_preferences", agent.extract_preferences)
    graph.add_node("save_preferences",    agent.save_preferences)

    # --- edges -----------------------------------------------------------
    graph.add_edge(START, "load_preferences")
    graph.add_edge("load_preferences", "classify")
    graph.add_conditional_edges("classify", agent.route_intent, {
        "react_node":        "react_node",
        "general_chat":      "general_chat",
        "commission_intake": "commission_intake",
    })
    graph.add_edge("react_node",          "extract_preferences")
    graph.add_edge("general_chat",        "extract_preferences")
    graph.add_edge("commission_intake",   "extract_preferences")
    graph.add_edge("extract_preferences", "save_preferences")
    graph.add_edge("save_preferences",    END)

    # --- checkpointer: Postgres if configured, else in-memory ------------
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    postgres_uri = os.environ.get("POSTGRES_URI")
    if postgres_uri:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool
            # A pool of up to 5 connections. `check` validates a connection before
            # handing it out (reconnecting if the DB/network silently dropped it while
            # idle) instead of only discovering it's dead when a real query fails.
            # `max_idle` proactively recycles idle connections before that can happen.
            # Keepalives detect a dead peer at the TCP level faster than the default.
            pool = ConnectionPool(
                postgres_uri,
                max_size=5,
                kwargs={
                    "autocommit": True,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 3,
                },
                check=ConnectionPool.check_connection,
                max_idle=300,
            )
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
        except Exception as e:
            import logging
            logging.warning(f"Postgres connection failed, falling back to MemorySaver: {e}")
            checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# Built once, at import time — every chat request reuses this same
# compiled graph (see agent/__init__.py's chat_with_memory()).
compiled_graph = build_graph()

"""Builds and compiles the Paintbox LangGraph."""
import os
from langgraph.graph import END, START, StateGraph
from .state import AgentState
from .nodes import PaintboxAgent


def build_graph():
    agent = PaintboxAgent()
    graph = StateGraph(AgentState)

    graph.add_node("load_preferences",    agent.load_preferences)
    graph.add_node("classify",            agent.classify)
    graph.add_node("react_node",          agent.react_node)
    graph.add_node("general_chat",        agent.general_chat)
    graph.add_node("commission_intake",   agent.commission_intake)
    graph.add_node("guardrail",           agent.guardrail_node)
    graph.add_node("extract_preferences", agent.extract_preferences)
    graph.add_node("save_preferences",    agent.save_preferences)

    graph.add_edge(START, "load_preferences")
    graph.add_edge("load_preferences", "classify")
    graph.add_conditional_edges("classify", agent.route_intent, {
        "react_node":        "react_node",
        "general_chat":      "general_chat",
        "commission_intake": "commission_intake",
    })
    graph.add_edge("react_node",          "guardrail")
    graph.add_edge("general_chat",        "guardrail")
    graph.add_edge("commission_intake",   "guardrail")
    graph.add_edge("guardrail",           "extract_preferences")
    graph.add_edge("extract_preferences", "save_preferences")
    graph.add_edge("save_preferences",    END)

    postgres_uri = os.environ.get("POSTGRES_URI")
    if postgres_uri:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        conn = psycopg.connect(postgres_uri, autocommit=True)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
    else:
        checkpointer = None

    return graph.compile(checkpointer=checkpointer)


compiled_graph = build_graph()

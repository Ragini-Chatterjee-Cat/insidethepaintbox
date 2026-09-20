"""
CommissionInfoTool — returns commission information.

Purpose: hands back the static commission-request text from tools/base.py
verbatim — no search, no lookup, the simplest tool in the set.

Called by: nothing in this codebase calls it directly. It's registered in
tools/__init__.py's ARTWORK_TOOLS list, bound to the LLM in
agent/nodes.py's PaintboxAgent.__init__(), and invoked by LangGraph's
ToolNode inside PaintboxAgent.react_node() whenever the model decides a
visitor is asking about commissioning custom work.
"""
from langchain_core.tools import BaseTool

from .base import COMMISSION_INFO


class CommissionInfoTool(BaseTool):
    name: str = "get_commission_info"
    description: str = (
        "Return information about how to commission custom artwork from the artist. "
        "Use this when a visitor asks about commissioning, ordering, pricing, or requesting custom artwork."
    )
    handle_tool_error: bool = True

    def _run(self) -> str:
        """Return the commission-request text as-is; no arguments to process."""
        return COMMISSION_INFO

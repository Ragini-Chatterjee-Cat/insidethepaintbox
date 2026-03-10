"""CommissionInfoTool — returns commission information."""
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
        return COMMISSION_INFO

    def _arun(self):
        raise NotImplementedError("Use _run instead.")

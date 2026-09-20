"""
Artwork tools package for Inside the Paintbox.

Purpose: instantiates every LangChain tool class in this package once
into a single list, ARTWORK_TOOLS, so the rest of the app never has to
know the individual tool classes exist.

Imported by: agent/nodes.py only — PaintboxAgent.__init__() binds
ARTWORK_TOOLS to the tool-calling LLM (`self._llm_main.bind_tools(...)`)
and wraps it in a LangGraph ToolNode (`self._tool_node`). No functions
here to call; this module's only job is building that one list at
import time.
"""
from .commission_info import CommissionInfoTool
from .filter_by_series import FilterBySeriesTool
from .get_artwork_details import GetArtworkDetailsTool
from .recommend_similar import RecommendSimilarTool
from .reveal_secret import RevealSecretTool
from .search_artworks import SearchArtworksTool

ARTWORK_TOOLS = [
    SearchArtworksTool(),
    FilterBySeriesTool(),
    GetArtworkDetailsTool(),
    CommissionInfoTool(),
    RecommendSimilarTool(),
    RevealSecretTool(),
]

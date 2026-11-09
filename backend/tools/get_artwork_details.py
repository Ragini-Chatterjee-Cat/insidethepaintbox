"""
GetArtworkDetailsTool — full details about a specific artwork by title.

Purpose: answers "tell me about <artwork name>" with the full indexed
content for that one piece - resolved by exact title, then a unique
partial title, then the reranker's best guess (tools/base.resolve_artwork).

Called by: nothing in this codebase calls it directly. It's registered in
tools/__init__.py's ARTWORK_TOOLS list, bound to the LLM in
agent/nodes.py's PaintboxAgent.__init__(), and invoked by LangGraph's
ToolNode inside PaintboxAgent.react_node() whenever the model decides a
visitor named one specific artwork by title.
"""
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from .base import resolve_artwork


class GetArtworkDetailsInput(BaseModel):
    artwork_name: str = Field(description="The title of a specific artwork, e.g. 'Voices' or 'Icarus'")


class GetArtworkDetailsTool(BaseTool):
    name: str = "get_artwork_details"
    description: str = (
        "Get full details about a specific individual artwork by its title. "
        "Use this ONLY when a visitor asks about a particular piece by name. "
        "Do NOT use this for series names — use filter_by_series for those instead."
    )
    args_schema: type[BaseModel] = GetArtworkDetailsInput
    handle_tool_error: bool = True

    def _run(self, artwork_name: str) -> str:
        """Resolve `artwork_name` to one artwork (exact title, partial
        title, then reranked best guess) and return its full details."""
        match = resolve_artwork(artwork_name)
        if match is None:
            raise ToolException(f"Could not find any artwork named '{artwork_name}'.")

        meta = match["meta"]
        output = f"Artwork: {meta.get('title', 'Unknown')}\n"
        if meta.get("series"):
            output += f"Series: {meta['series']}\n"
        if meta.get("url"):
            output += f"URL: {meta['url']}\n"
        output += f"\n{match['content']}"
        return output

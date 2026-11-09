"""
SearchArtworksTool — semantic search over the artwork collection.

Purpose: lets the agent browse by mood/theme/style ("something dark and
emotional") rather than by exact title: vector search over the same
multimodal embedding the collection was indexed with, then a reranker
(tools/base.find_relevant) to order and filter the candidates.

Called by: nothing in this codebase calls it directly. It's registered in
tools/__init__.py's ARTWORK_TOOLS list, bound to the LLM in
agent/nodes.py's PaintboxAgent.__init__(), and invoked by LangGraph's
ToolNode inside PaintboxAgent.react_node() whenever the model decides
(from this class's `description` below) that a visitor's message calls
for a general/thematic search rather than a named lookup.
"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import RERANK_STRONG_SCORE, description_of, find_relevant


class SearchArtworksInput(BaseModel):
    query: str = Field(description="A descriptive search query, e.g. 'dark emotional artwork'")


class SearchArtworksTool(BaseTool):
    name: str = "search_artworks"
    description: str = (
        "Search the artwork collection using semantic similarity. "
        "Use this when a visitor asks for recommendations, best works, popular pieces, or to browse generally. "
        "Also use it for themes, moods, styles, or subjects ('something emotional', 'animals', 'colourful'). "
        "Do NOT use this when the visitor names a specific artwork title (use get_artwork_details) "
        "or a series name (use filter_by_series)."
    )
    args_schema: type[BaseModel] = SearchArtworksInput
    handle_tool_error: bool = True

    def _run(self, query: str) -> str:
        """Return up to 5 artworks relevant to `query` (vector search, then
        reranked), each labelled as a strong or weak match, or a fallback
        message if nothing is relevant."""
        matches = find_relevant(query)[:5]
        if not matches:
            return (
                "No artworks matched that description. "
                "The collection includes: Portraits, Animal Portraits, Mythical, "
                "Thoughts, Camera Series, Diary Entries, Fanart, and Cards."
            )

        output_lines = []
        for match in matches:
            meta = match["meta"]
            entry = f"- {meta.get('title', 'Unknown')}"
            if meta.get("series"):
                entry += f" (Series: {meta['series']})"
            # Weak = the closest thing in the collection, not a real match
            # (e.g. other celebrity portraits for "Taylor Swift"), so the
            # model can say so instead of presenting it as what was asked.
            if match["score"] is not None and match["score"] < RERANK_STRONG_SCORE:
                entry += " [weak match]"
            if meta.get("url"):
                entry += f"\n  URL: {meta['url']}"
            description = description_of(match["content"])
            if description:
                entry += f"\n  About: {description[:300]}" + ("..." if len(description) > 300 else "")
            output_lines.append(entry)

        return "Found these artworks:\n\n" + "\n\n".join(output_lines)

"""
SearchArtworksTool — semantic search over the artwork collection.

Purpose: lets the agent browse by mood/theme/style ("something dark and
emotional") rather than by exact title, using the same multimodal
embedding the collection was indexed with.

Called by: nothing in this codebase calls it directly. It's registered in
tools/__init__.py's ARTWORK_TOOLS list, bound to the LLM in
agent/nodes.py's PaintboxAgent.__init__(), and invoked by LangGraph's
ToolNode inside PaintboxAgent.react_node() whenever the model decides
(from this class's `description` below) that a visitor's message calls
for a general/thematic search rather than a named lookup.
"""
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import CONFIDENCE_THRESHOLD, collection, embed_query


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
        """Embed `query`, return up to 5 confidently-matching (non-secret)
        artworks as a formatted text block, or a fallback message if
        nothing clears CONFIDENCE_THRESHOLD."""
        query_embedding = embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"secret": False},
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return "No artworks found matching that description."

        output_lines = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            if distance > CONFIDENCE_THRESHOLD:
                continue
            meta = results["metadatas"][0][i]
            content = results["documents"][0][i]
            title = meta.get("title", "Unknown")
            series = meta.get("series", "")
            url = meta.get("url", "")
            entry = f"- {title}"
            if series:
                entry += f" (Series: {series})"
            if url:
                entry += f"\n  URL: {url}"
            snippet = content[:200].replace("\n", " ").strip()
            if snippet:
                entry += f"\n  About: {snippet}" + ("..." if len(content) > 200 else "")
            output_lines.append(entry)

        if not output_lines:
            return (
                "No artworks closely matched that description. "
                "The collection includes: Portraits, Animal Portraits, Mythical, "
                "Thoughts, Camera Series, Diary Entries, Fanart, and Cards."
            )

        return "Found these artworks:\n\n" + "\n\n".join(output_lines)

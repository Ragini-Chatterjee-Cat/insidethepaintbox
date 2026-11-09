"""
RecommendSimilarTool — find artworks similar to a given one.

Purpose: answers "I liked X, show me more like it" by looking up X, then
finding the (non-secret) artworks whose stored text+image embeddings are
nearest to X's own - so "similar" includes how the pieces look.

Called by: nothing in this codebase calls it directly. It's registered in
tools/__init__.py's ARTWORK_TOOLS list, bound to the LLM in
agent/nodes.py's PaintboxAgent.__init__(), and invoked by LangGraph's
ToolNode inside PaintboxAgent.react_node() whenever the model decides a
visitor wants more of something they already liked.
"""
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from .base import collection, is_overview_page, resolve_artwork


class RecommendSimilarInput(BaseModel):
    artwork_name: str = Field(description="Name of the artwork to base recommendations on")


class RecommendSimilarTool(BaseTool):
    name: str = "recommend_similar"
    description: str = (
        "Find artworks that are visually or thematically similar to a given artwork. "
        "Use this when a visitor liked a piece and wants to see more like it."
    )
    args_schema: type[BaseModel] = RecommendSimilarInput
    handle_tool_error: bool = True

    def _run(self, artwork_name: str) -> str:
        """Resolve `artwork_name` to an artwork, then find up to 4 others
        whose stored embeddings are nearest to its own. Those embeddings
        were made from each piece's text AND images, so "similar" covers
        how the pieces look, not just how they're described."""
        source = resolve_artwork(artwork_name)
        if source is None:
            raise ToolException(f"Could not find artwork named '{artwork_name}' to base recommendations on.")
        source_title = source["meta"].get("title", artwork_name)

        source_embedding = collection.get(ids=[source["id"]], include=["embeddings"])["embeddings"][0]
        similar = collection.query(
            query_embeddings=[source_embedding],
            n_results=10,
            where={"secret": False},
            include=["metadatas"],
        )

        lines = [f"Artworks similar to '{source_title}':\n"]
        count = 0
        for doc_id, meta in zip(similar["ids"][0], similar["metadatas"][0]):
            if doc_id == source["id"] or is_overview_page(meta):
                continue
            line = f"- {meta.get('title', 'Unknown')}"
            if meta.get("series"):
                line += f" (Series: {meta['series']})"
            if meta.get("url"):
                line += f"\n  {meta['url']}"
            lines.append(line)
            count += 1
            if count >= 4:
                break

        if count == 0:
            raise ToolException(f"No similar artworks found for '{artwork_name}'.")
        return "\n".join(lines)

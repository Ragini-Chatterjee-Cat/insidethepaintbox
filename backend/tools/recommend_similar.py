"""RecommendSimilarTool — find artworks similar to a given one."""
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from .base import collection, embed_query


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
        source_embedding = embed_query(artwork_name)
        source_results = collection.query(
            query_embeddings=[source_embedding],
            n_results=1,
            include=["documents", "metadatas"],
        )

        if not source_results["documents"] or not source_results["documents"][0]:
            raise ToolException(f"Could not find artwork named '{artwork_name}' to base recommendations on.")

        source_content = source_results["documents"][0][0]
        source_meta = source_results["metadatas"][0][0]
        source_title = source_meta.get("title", artwork_name)

        content_embedding = embed_query(source_content)
        similar = collection.query(
            query_embeddings=[content_embedding],
            n_results=6,
            where={"secret": False},
            include=["metadatas", "distances"],
        )

        lines = [f"Artworks similar to '{source_title}':\n"]
        count = 0
        for meta in similar["metadatas"][0]:
            title = meta.get("title", "Unknown")
            if title.lower() == source_title.lower():
                continue
            series = meta.get("series", "")
            url = meta.get("url", "")
            line = f"- {title}"
            if series:
                line += f" (Series: {series})"
            if url:
                line += f"\n  {url}"
            lines.append(line)
            count += 1
            if count >= 4:
                break

        if count == 0:
            raise ToolException(f"No similar artworks found for '{artwork_name}'.")

        return "\n".join(lines)

    def _arun(self, artwork_name: str):
        raise NotImplementedError("Use _run instead.")

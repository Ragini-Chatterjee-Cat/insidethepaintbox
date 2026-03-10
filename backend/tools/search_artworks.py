"""SearchArtworksTool — semantic search over the artwork collection."""
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from .base import collection, embedding_model, CONFIDENCE_THRESHOLD


class SearchArtworksInput(BaseModel):
    query: str = Field(description="A descriptive search query, e.g. 'dark emotional artwork'")


class SearchArtworksTool(BaseTool):
    name: str = "search_artworks"
    description: str = (
        "Search the artwork collection using semantic similarity. "
        "Use this when a visitor asks about specific themes, styles, subjects, or artwork descriptions "
        "— NOT when they name a specific artwork title or a series name."
    )
    args_schema: type[BaseModel] = SearchArtworksInput
    handle_tool_error: bool = True

    def _run(self, query: str) -> str:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
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
                entry += f"\n  About: {snippet}..."
            output_lines.append(entry)

        if not output_lines:
            return (
                "No artworks closely matched that description. "
                "The collection includes: Portraits, Animal Portraits, Mythical, "
                "Thoughts, Camera Series, Diary Entries, Fanart, and Cards."
            )

        return "Found these artworks:\n\n" + "\n\n".join(output_lines)

    def _arun(self, query: str):
        raise NotImplementedError("Use _run instead.")

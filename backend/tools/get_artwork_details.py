"""GetArtworkDetailsTool — full details about a specific artwork by title."""
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from .base import collection, embedding_model


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
        query_embedding = embedding_model.encode(artwork_name).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            raise ToolException(f"Could not find any artwork named '{artwork_name}'.")

        best_idx = 0
        name_lower = artwork_name.lower()
        for i, meta in enumerate(results["metadatas"][0]):
            title = meta.get("title", "").lower()
            if name_lower in title or title in name_lower:
                best_idx = i
                break

        meta = results["metadatas"][0][best_idx]
        content = results["documents"][0][best_idx]
        title = meta.get("title", "Unknown")
        series = meta.get("series", "")
        url = meta.get("url", "")

        output = f"Artwork: {title}\n"
        if series:
            output += f"Series: {series}\n"
        if url:
            output += f"URL: {url}\n"
        output += f"\n{content}"
        return output

    def _arun(self, artwork_name: str):
        raise NotImplementedError("Use _run instead.")

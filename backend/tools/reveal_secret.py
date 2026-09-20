"""RevealSecretTool — surfaces hidden artworks, only when a visitor explicitly asks."""
from langchain_core.tools import BaseTool, ToolException
from .base import collection


class RevealSecretTool(BaseTool):
    name: str = "reveal_secret_artwork"
    description: str = (
        "Reveal hidden, unlisted artworks that aren't part of normal browsing or search. "
        "Use this ONLY when a visitor explicitly asks about something secret, hidden, unlisted, "
        "or an easter egg — e.g. 'do you have any secrets', 'is there a hidden piece', "
        "'show me something no one else knows about'. Never call this for ordinary browsing, "
        "recommendation, or search requests, and never mention these pieces unprompted."
    )
    handle_tool_error: bool = True

    def _run(self) -> str:
        results = collection.get(
            where={"secret": True},
            include=["documents", "metadatas"],
        )

        if not results["metadatas"]:
            raise ToolException("No secret artworks are hidden away right now.")

        lines = ["You found a secret! Hidden pieces:\n"]
        for meta, content in zip(results["metadatas"], results["documents"]):
            title = meta.get("title", "Unknown")
            url = meta.get("url", "")
            snippet = content[:200].replace("\n", " ").strip()
            entry = f"- {title}"
            if url:
                entry += f"\n  URL: {url}"
            if snippet:
                entry += f"\n  About: {snippet}..."
            lines.append(entry)

        return "\n\n".join(lines)

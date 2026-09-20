"""
RevealSecretTool — surfaces hidden artworks, only when a visitor explicitly asks.

Purpose: three artwork pages (Bibbity, Bare, Saree) are flagged
"secret": true in Chroma metadata and excluded from search_artworks and
recommend_similar's results. This tool is the only path back to them —
it exists so they can be found on purpose, not by accident.

Called by: nothing in this codebase calls it directly. It's registered in
tools/__init__.py's ARTWORK_TOOLS list, bound to the LLM in
agent/nodes.py's PaintboxAgent.__init__(), and invoked by LangGraph's
ToolNode inside PaintboxAgent.react_node() — but only when the model
judges (per this class's `description` and GALLERY_SYSTEM's rule in
agent/prompts.py) that a visitor explicitly asked about something
hidden/secret. It is never called for ordinary browsing.
"""
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
        """Fetch every document flagged secret=True and list them; no
        arguments, since this always reveals the whole hidden set."""
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

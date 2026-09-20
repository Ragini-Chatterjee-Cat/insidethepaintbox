"""
FilterBySeriesTool — list all artworks in a named series.

Purpose: answers "show me the Mythical series" with a metadata filter,
not a similarity search — series membership is exact, so there's no
reason to embed anything here.

Called by: nothing in this codebase calls it directly. It's registered in
tools/__init__.py's ARTWORK_TOOLS list, bound to the LLM in
agent/nodes.py's PaintboxAgent.__init__(), and invoked by LangGraph's
ToolNode inside PaintboxAgent.react_node() whenever the model decides a
visitor named one of the eight series directly.
"""
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from document_loader import SERIES_ARTWORK_MAP

from .base import SERIES_URLS, collection


class FilterBySeriesInput(BaseModel):
    series: str = Field(description="The series name (case-insensitive), e.g. 'Mythical'")


class FilterBySeriesTool(BaseTool):
    name: str = "filter_by_series"
    description: str = (
        "List all artworks that belong to a specific series. "
        "Use this when a visitor names a series directly — e.g. 'Diary Entries', 'Mythical', 'Portraits'. "
        "Valid series: Portraits, Animal Portraits, Mythical, Thoughts, Camera Series, Diary Entries, Fanart, Cards."
    )
    args_schema: type[BaseModel] = FilterBySeriesInput
    handle_tool_error: bool = True

    def _run(self, series: str) -> str:
        """Fuzzy-match `series` against the known series names, then list
        every artwork Chroma has tagged with that series in its metadata."""
        series_lower = series.lower().strip()
        matched_series = None
        for known in SERIES_ARTWORK_MAP.keys():
            if series_lower == known.lower() or series_lower in known.lower():
                matched_series = known
                break

        if not matched_series:
            available = ", ".join(SERIES_ARTWORK_MAP.keys())
            raise ToolException(f"Series '{series}' not found. Available: {available}")

        results = collection.get(
            where={"$and": [{"series": matched_series}, {"secret": False}]},
            include=["metadatas"],
        )

        series_url = SERIES_URLS.get(matched_series, "")

        if not results["metadatas"]:
            filenames = SERIES_ARTWORK_MAP[matched_series]
            titles = [f.replace(".html", "").replace("-", " ").title() for f in filenames]
            header = f"Artworks in '{matched_series}':"
            if series_url:
                header += f"\nSeries page: {series_url}"
            return header + "\n" + "\n".join(f"- {t}" for t in titles)

        series_header = f"Artworks in the '{matched_series}' series:"
        if series_url:
            series_header += f"\nSeries page: {series_url}"
        lines = [series_header + "\n"]
        for meta in results["metadatas"]:
            title = meta.get("title", "Unknown")
            url = meta.get("url", "")
            line = f"- {title}"
            if url:
                line += f"  ({url})"
            lines.append(line)

        return "\n".join(lines)

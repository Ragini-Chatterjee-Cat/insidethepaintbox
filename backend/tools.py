"""
ReAct Agent Tools for Inside the Paintbox
All tools operate against the ChromaDB collection directly,
reusing the shared collection and embedding_model from rag.py.
"""

from langchain_core.tools import tool
from document_loader import SERIES_ARTWORK_MAP

# Import shared singletons from rag.py — avoids double-initialising
# the embedding model and ChromaDB client.
from rag import collection, embedding_model

COMMISSION_INFO = """
Ragini Chatterjee accepts commissions for custom artwork.

To request a commission:
- Contact via the website contact form: https://insidethepaintbox.netlify.app/pages/contact.html
- Or reach out on Instagram: @ragini_chatterjee

Commission types available:
- Portraits (people, pets, characters)
- Custom illustrations in any of the existing series styles
- Cards for special occasions

Please include reference photos and details about what you'd like when reaching out.
Response time is typically within a few days.
"""

CONFIDENCE_THRESHOLD = 1.5


@tool
def search_artworks(query: str) -> str:
    """
    Search the artwork collection using semantic similarity.
    Use this when a visitor asks about specific themes, styles,
    subjects, or artwork descriptions.
    Returns matching artworks with titles, series, descriptions, and URLs.

    Args:
        query: A descriptive search query, e.g. 'dark emotional artwork'
               or 'portrait of a woman' or 'mythical creatures'
    """
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
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

    except Exception as e:
        return f"Search encountered an error: {str(e)}"


@tool
def filter_by_series(series: str) -> str:
    """
    List all artworks that belong to a specific series.
    Use this when a visitor asks to browse a particular series,
    or asks 'what series do you have?' or 'show me all [series name]'.

    Valid series: Portraits, Animal Portraits, Mythical, Thoughts,
    Camera Series, Diary Entries, Fanart, Cards.

    Args:
        series: The series name (case-insensitive)
    """
    try:
        series_lower = series.lower().strip()
        matched_series = None
        for known in SERIES_ARTWORK_MAP.keys():
            if series_lower == known.lower() or series_lower in known.lower():
                matched_series = known
                break

        if not matched_series:
            available = ", ".join(SERIES_ARTWORK_MAP.keys())
            return f"Series '{series}' not found. Available: {available}"

        results = collection.get(
            where={"series": matched_series},
            include=["metadatas"]
        )

        if not results["metadatas"]:
            # Fall back to static map
            filenames = SERIES_ARTWORK_MAP[matched_series]
            titles = [f.replace(".html", "").replace("-", " ").title() for f in filenames]
            return f"Artworks in '{matched_series}':\n" + "\n".join(f"- {t}" for t in titles)

        lines = [f"Artworks in the '{matched_series}' series:\n"]
        for meta in results["metadatas"]:
            title = meta.get("title", "Unknown")
            url = meta.get("url", "")
            line = f"- {title}"
            if url:
                line += f"  ({url})"
            lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving series: {str(e)}"


@tool
def get_artwork_details(artwork_name: str) -> str:
    """
    Get full details about a specific named artwork.
    Use this when a visitor asks about a particular piece by name,
    e.g. 'tell me about Voices' or 'what is Icarus about?'

    Args:
        artwork_name: The name of the artwork, e.g. 'Voices' or 'Icarus'
    """
    try:
        query_embedding = embedding_model.encode(artwork_name).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"] or not results["documents"][0]:
            return f"Could not find any artwork named '{artwork_name}'."

        # Prefer an exact title match over pure semantic distance
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

    except Exception as e:
        return f"Error looking up artwork: {str(e)}"


@tool
def get_commission_info() -> str:
    """
    Return information about how to commission custom artwork from the artist.
    Use this when a visitor asks about commissioning, ordering, pricing,
    or requesting custom artwork from Ragini Chatterjee.
    """
    return COMMISSION_INFO


@tool
def recommend_similar(artwork_name: str) -> str:
    """
    Find artworks that are visually or thematically similar to a given artwork.
    Use this when a visitor liked a piece and wants to see more like it,
    or asks 'what else is similar to X?' or 'recommend something like Y'.

    Args:
        artwork_name: Name of the artwork to base recommendations on
    """
    try:
        # Look up the source artwork's full document
        source_embedding = embedding_model.encode(artwork_name).tolist()
        source_results = collection.query(
            query_embeddings=[source_embedding],
            n_results=1,
            include=["documents", "metadatas"]
        )

        if not source_results["documents"] or not source_results["documents"][0]:
            return f"Could not find artwork named '{artwork_name}' to base recommendations on."

        source_content = source_results["documents"][0][0]
        source_meta = source_results["metadatas"][0][0]
        source_title = source_meta.get("title", artwork_name)

        # Use full content for richer semantic similarity
        content_embedding = embedding_model.encode(source_content).tolist()
        similar = collection.query(
            query_embeddings=[content_embedding],
            n_results=6,
            include=["metadatas", "distances"]
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
            return f"No similar artworks found for '{artwork_name}'."

        return "\n".join(lines)

    except Exception as e:
        return f"Error finding similar artworks: {str(e)}"


# Single export for langgraph_memory.py to import
ARTWORK_TOOLS = [
    search_artworks,
    filter_by_series,
    get_artwork_details,
    get_commission_info,
    recommend_similar,
]

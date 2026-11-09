"""
Shared singletons and constants for the artwork tool classes.

Purpose: every tool in this package needs the same Chroma collection,
the same embedding function, and the same series-URL/commission text —
this file is where those live once instead of being redefined per tool.

Imported by: every file in tools/ (search_artworks.py, get_artwork_details.py,
filter_by_series.py, recommend_similar.py, commission_info.py,
reveal_secret.py). The module-level values are resolved once on first
import; find_relevant()/resolve_artwork() are the shared retrieval steps.
"""
import logging

from rag import collection, embed_query, rerank

logger = logging.getLogger(__name__)

# --- shared values ------------------------------------------------------

BASE_URL = "https://insidethepaintbox.netlify.app"
# Only used when the reranker is unavailable: Chroma distance above this =
# "not a real match". Distance alone is poorly calibrated (it rejected the
# phoenix painting for "a phoenix" and accepted unrelated portraits for
# "Taylor Swift"), so normally the rerank scores below decide instead.
CONFIDENCE_THRESHOLD = 1.5
# Rerank scores (0-1), calibrated on a ~40-question retrieval eval plus 12
# held-out questions: clear matches score >= 0.55, unrelated queries
# (spaceships, the ocean) < 0.3, and 0.3-0.55 mixes real answers the page
# text barely supports ("a Korean singer" -> Byun, 0.39) with near-misses
# ("Taylor Swift" -> other celebrity portraits, 0.37) - so that band is
# kept but flagged weak, and the model is told to present it as such.
RERANK_MIN_SCORE = 0.3
RERANK_STRONG_SCORE = 0.55
# Name lookups ("tell me about the phoenix one") accept a reranked guess that
# is strong, or decent and clearly ahead of the runner-up. Filler words drag
# real references down ("the phoenix one" -> From the Ashes 0.54 vs 0.47
# next), while names that don't exist produce near-ties between unrelated
# pieces ("Beyonce" 0.37 vs 0.35, "the self portrait" 0.50 vs 0.49).
RERANK_LOOKUP_MIN_SCORE = 0.45
RERANK_CLEAR_LEAD = 0.05
N_CANDIDATES = 20  # vector-search candidates handed to the reranker

# Filenames here must match frontend/pages/series/ exactly - Netlify's
# static hosting is case-sensitive
SERIES_URLS = {
    "Portraits":        f"{BASE_URL}/pages/series/Portraits.html",
    "Animal Portraits": f"{BASE_URL}/pages/series/portraits-animals.html",
    "Mythical":         f"{BASE_URL}/pages/series/Mythical.html",
    "Thoughts":         f"{BASE_URL}/pages/series/Thoughts.html",
    "Camera Series":    f"{BASE_URL}/pages/series/Cam.html",
    "Diary Entries":    f"{BASE_URL}/pages/series/diary-entries.html",
    "Fanart":           f"{BASE_URL}/pages/series/Fanart.html",
    "Cards":            f"{BASE_URL}/pages/series/Cards.html",
}

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


# --- shared retrieval -----------------------------------------------------

def is_overview_page(meta: dict) -> bool:
    """A series' landing page (it shows every piece in the series) rather
    than an artwork. Useful for "what is the Mythical series about", but in
    search results it just crowds out the actual artworks."""
    return not meta.get("series") and "/pages/series/" in meta.get("url", "")


def description_of(content: str) -> str:
    """The artist's own description, without the title/series/URL header
    that document_loader.py puts in front of it."""
    return content.split("\n\n", 1)[1] if "\n\n" in content else ""


def find_relevant(query: str, n_candidates: int = N_CANDIDATES, include_overviews: bool = False) -> list[dict]:
    """Vector-search for candidates, then rerank them against `query`.
    Returns matches scoring at least RERANK_MIN_SCORE, best first, as dicts
    with id/meta/content/score - score is None if the reranker was
    unavailable and the vector-distance cutoff was used instead."""
    results = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=n_candidates,
        where={"secret": False},
        include=["documents", "metadatas", "distances"],
    )
    candidates = [
        {"id": i, "meta": m, "content": c, "distance": d}
        for i, m, c, d in zip(
            results["ids"][0], results["metadatas"][0], results["documents"][0], results["distances"][0]
        )
        if include_overviews or not is_overview_page(m)
    ]
    if not candidates:
        return []
    try:
        scores = rerank(query, [c["content"] for c in candidates])
    except Exception:
        # Search keeps working (less accurately) if the reranker is down.
        logger.exception("Rerank failed; falling back to vector distance")
        return [dict(c, score=None) for c in candidates if c["distance"] <= CONFIDENCE_THRESHOLD]
    ranked = sorted(
        (dict(c, score=s) for c, s in zip(candidates, scores)), key=lambda c: c["score"], reverse=True
    )
    return [c for c in ranked if c["score"] >= RERANK_MIN_SCORE]


def resolve_artwork(name: str) -> dict | None:
    """Find the one artwork a visitor means by `name`: an exact title match,
    then a unique partial one ("Athena" -> ATHENIAN), else the reranker's
    best guess if it's convincing ("the pandora one", "the phoenix one").
    Returns None rather than guess - a weak best guess for a piece that
    doesn't exist ("Beyonce") would hand back an unrelated artwork."""
    name_lower = name.lower().strip()
    found = collection.get(where={"secret": False}, include=["documents", "metadatas"])
    docs = [
        {"id": i, "meta": m, "content": c, "score": None}
        for i, m, c in zip(found["ids"], found["metadatas"], found["documents"])
    ]
    exact = [d for d in docs if d["meta"].get("title", "").lower() == name_lower]
    if exact:
        return exact[0]
    partial = [
        d for d in docs
        if not is_overview_page(d["meta"])
        and (title := d["meta"].get("title", "").lower())
        and (name_lower in title or title in name_lower)
    ]
    if len(partial) == 1:
        return partial[0]
    # Series overview pages are never "the artwork" a visitor means - and
    # names that don't exist tend to land on them ("the Mona Lisa" -> PORTRAITS).
    matches = find_relevant(name, n_candidates=10)
    if not matches:
        return None
    best = matches[0]
    if best["score"] is None:  # reranker unavailable: distance cutoff already applied
        return best
    runner_up = matches[1]["score"] if len(matches) > 1 else 0.0
    if best["score"] >= RERANK_STRONG_SCORE or (
        best["score"] >= RERANK_LOOKUP_MIN_SCORE and best["score"] - runner_up >= RERANK_CLEAR_LEAD
    ):
        return best
    return None

"""Tests for the artwork tool classes - patches embed_query/rerank in
tools/base.py, where the shared retrieval helpers use them (not rag.py,
where they're defined), since `from rag import X` binds base its own name."""
from langchain_core.tools import ToolException

import tools.base as base_mod
import tools.filter_by_series as filter_by_series_mod
from tools.filter_by_series import FilterBySeriesTool
from tools.get_artwork_details import GetArtworkDetailsTool
from tools.recommend_similar import RecommendSimilarTool
from tools.search_artworks import SearchArtworksTool

ARTWORK_URL = "https://insidethepaintbox.netlify.app/artworks/x.html"
OVERVIEW_URL = "https://insidethepaintbox.netlify.app/pages/series/Mythical.html"


def _meta(title, series="Mythical", url=ARTWORK_URL):
    return {"title": title, "series": series, "url": url}


def _fake_query_result(items):
    """items: (title, distance) or (title, distance, meta) tuples."""
    metas = [it[2] if len(it) > 2 else _meta(it[0]) for it in items]
    return {
        "ids": [[f"doc_{i}" for i in range(len(items))]],
        "documents": [[f"Artwork: {it[0]}\nURL: x\n\nabout {it[0]}" for it in items]],
        "metadatas": [metas],
        "distances": [[it[1] for it in items]],
    }


def _fake_get_result(titles):
    return {
        "ids": [f"doc_{i}" for i in range(len(titles))],
        "documents": [f"Artwork: {t}\nURL: x\n\nabout {t}" for t in titles],
        "metadatas": [_meta(t) for t in titles],
    }


def _patch_retrieval(monkeypatch, query_items, scores, all_titles=()):
    """Fake the vector search, the reranker (scores line up with
    query_items) and the full-collection listing used for title lookups."""
    seen = {"query": [], "get": []}

    def fake_query(**kwargs):
        seen["query"].append(kwargs)
        return _fake_query_result(query_items)

    def fake_get(**kwargs):
        seen["get"].append(kwargs)
        return _fake_get_result(list(all_titles))

    def fake_rerank(_query, documents):
        assert len(documents) == len(scores), "reranker should see exactly the candidates"
        return list(scores)

    monkeypatch.setattr(base_mod, "embed_query", lambda _q: [0.0])
    monkeypatch.setattr(base_mod, "rerank", fake_rerank)
    monkeypatch.setattr(base_mod.collection, "query", fake_query)
    monkeypatch.setattr(base_mod.collection, "get", fake_get)
    return seen


def test_get_artwork_details_prefers_an_exact_title_over_search(monkeypatch):
    _patch_retrieval(monkeypatch, [("Voices", 0.1)], [0.9], all_titles=["Voices", "Icarus"])

    result = GetArtworkDetailsTool()._run("icarus")

    assert result.startswith("Artwork: Icarus")


def test_get_artwork_details_accepts_a_strong_reranked_guess(monkeypatch):
    """No title contains "Vices", but the reranker is confident it's Voices."""
    _patch_retrieval(monkeypatch, [("Voices", 1.6), ("Icarus", 1.7)], [0.8, 0.2], all_titles=["Voices", "Icarus"])

    result = GetArtworkDetailsTool()._run("Vices")

    assert result.startswith("Artwork: Voices")


def test_get_artwork_details_refuses_a_weak_guess(monkeypatch):
    """Regression test: asking for a piece that doesn't exist ("Beyonce")
    must not hand back the nearest unrelated portrait as if it were it."""
    _patch_retrieval(monkeypatch, [("In Memory", 1.2)], [0.46], all_titles=["In Memory"])

    try:
        GetArtworkDetailsTool()._run("Beyonce")
        assert False, "expected a ToolException"
    except ToolException:
        pass


def test_get_artwork_details_excludes_secret_pieces(monkeypatch):
    """Regression test: a visitor who knew a secret artwork's exact name
    could get its full details directly - every lookup must filter them."""
    seen = _patch_retrieval(monkeypatch, [("Voices", 0.2)], [0.9], all_titles=["Voices"])

    GetArtworkDetailsTool()._run("Something else")

    assert all(kw["where"] == {"secret": False} for kw in seen["get"] + seen["query"])


def test_recommend_similar_excludes_secret_pieces_and_itself(monkeypatch):
    """Regression test: the source lookup didn't filter secrets, so a secret
    piece's real title could leak into the response header."""
    seen = _patch_retrieval(
        monkeypatch, [("Icarus", 0.0), ("Draconic", 0.3)], [], all_titles=["Icarus", "Draconic"]
    )
    monkeypatch.setattr(
        base_mod.collection, "get",
        lambda **kw: seen["get"].append(kw) or (
            {"embeddings": [[0.1]]} if "ids" in kw else _fake_get_result(["Icarus", "Draconic"])
        ),
    )

    result = RecommendSimilarTool()._run("Icarus")

    assert "Draconic" in result
    assert "- Icarus" not in result  # never recommends the piece itself
    assert all(kw["where"] == {"secret": False} for kw in seen["query"])
    assert any(kw.get("where") == {"secret": False} for kw in seen["get"])


def test_filter_by_series_excludes_secret_pieces(monkeypatch):
    seen_kwargs = {}

    def fake_get(**kwargs):
        seen_kwargs.update(kwargs)
        return {"metadatas": [{"title": "Icarus", "url": ""}]}

    monkeypatch.setattr(filter_by_series_mod.collection, "get", fake_get)

    FilterBySeriesTool()._run("Mythical")

    assert seen_kwargs["where"] == {"$and": [{"series": "Mythical"}, {"secret": False}]}


def test_search_artworks_excludes_secret_pieces(monkeypatch):
    seen = _patch_retrieval(monkeypatch, [("Icarus", 0.2)], [0.9])

    SearchArtworksTool()._run("something mythical")

    assert seen["query"][0]["where"] == {"secret": False}


def test_search_artworks_drops_irrelevant_results(monkeypatch):
    _patch_retrieval(monkeypatch, [("Barely related", 1.0)], [0.1])

    result = SearchArtworksTool()._run("something mythical")

    assert "No artworks matched" in result


def test_search_artworks_ranks_by_rerank_score_not_distance(monkeypatch):
    """Regression test: "bubble tea" - the vector search put Lucky Panda
    6th; the reranker, which reads its description, puts it first."""
    _patch_retrieval(monkeypatch, [("Whine", 1.4), ("Lucky Panda", 1.55)], [0.45, 0.59])

    result = SearchArtworksTool()._run("bubble tea")

    assert result.index("Lucky Panda") < result.index("Whine")


def test_search_artworks_labels_weak_matches(monkeypatch):
    _patch_retrieval(monkeypatch, [("Harry Styles", 1.2), ("Neymar", 1.3)], [0.47, 0.6])

    result = SearchArtworksTool()._run("a portrait of Taylor Swift")

    assert "Harry Styles (Series: Mythical) [weak match]" in result
    assert "Neymar (Series: Mythical)\n" in result  # strong: unlabelled


def test_search_artworks_leaves_out_series_overview_pages(monkeypatch):
    overview = _meta("MYTHICAL.", series="", url=OVERVIEW_URL)
    _patch_retrieval(monkeypatch, [("MYTHICAL.", 0.1, overview), ("Draconic", 0.2)], [0.8])

    result = SearchArtworksTool()._run("dragons")

    assert "Draconic" in result and "MYTHICAL." not in result


def test_search_artworks_falls_back_to_distance_if_reranker_is_down(monkeypatch):
    _patch_retrieval(monkeypatch, [("Draconic", 0.5), ("Far away", 1.9)], [])

    def broken_rerank(*_a):
        raise RuntimeError("Voyage is down")

    monkeypatch.setattr(base_mod, "rerank", broken_rerank)

    result = SearchArtworksTool()._run("dragons")

    assert "Draconic" in result and "Far away" not in result


def test_filter_by_series_matches_case_insensitively(monkeypatch):
    monkeypatch.setattr(
        filter_by_series_mod.collection, "get",
        lambda **_k: {"metadatas": [{"title": "Icarus", "url": "https://example.com/icarus"}]},
    )

    result = FilterBySeriesTool()._run("mythical")

    assert "Icarus" in result


def test_filter_by_series_raises_for_unknown_series():
    try:
        FilterBySeriesTool()._run("Not A Real Series")
        assert False, "expected a ToolException"
    except ToolException as e:
        assert "not found" in str(e)

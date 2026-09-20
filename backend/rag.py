"""
RAG (Retrieval-Augmented Generation) Module — document indexing and embedding.

Purpose: the only place that talks to Voyage AI (embeddings) and ChromaDB
(vector storage). Retrieval and generation for chat happen elsewhere, in
the LangGraph agent (agent/nodes.py) and the tool classes (tools/) —
this module just builds and maintains the searchable index they query.

Embeddings are multimodal (text + image) via Voyage AI's voyage-multimodal-3,
so a search like "something dark and emotional" can match an artwork's actual
image, not just whatever its written description happens to say.

Imported by:
  - app.py: index_documents()/get_collection_stats(), called at startup
    (in the background) and by the /reindex and /stats endpoints.
  - tools/base.py: collection + embed_query, re-exported to every tool
    class for their own similarity searches at chat time.
"""

import hashlib
import logging
import os
import random
import threading
import time
from pathlib import Path

import chromadb
import voyageai
from PIL import Image
from voyageai.error import (
    APIConnectionError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    Timeout,
    TryAgain,
)

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "voyage-multimodal-3")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")

_voyage_api_key = os.environ.get("VOYAGE_API_KEY")
if not _voyage_api_key:
    raise ValueError("VOYAGE_API_KEY environment variable not set!")
_voyage_client = voyageai.Client(api_key=_voyage_api_key)

# Errors worth retrying: rate limits, timeouts, connection drops, and 5xx
# server errors are all transient. AuthenticationError/InvalidRequestError/
# MalformedRequestError are deliberately excluded — retrying a bad API key
# or a malformed request just delays the same failure, and for indexing
# (44+ documents) would multiply one permanent error into dozens of
# wasted retry cycles.
_RETRYABLE_VOYAGE_ERRORS = (
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    Timeout,
    APIConnectionError,
    TryAgain,
)


# --- embedding -----------------------------------------------------------

def _voyage_embed(inputs: list, input_type: str, max_attempts: int = 4) -> list:
    """Call Voyage's multimodal_embed with exponential backoff + jitter on
    transient errors (max_attempts total tries: 1s, 2s, 4s between them).
    Non-retryable errors (auth, bad request) raise immediately."""
    for attempt in range(1, max_attempts + 1):
        try:
            return _voyage_client.multimodal_embed(
                inputs, model=EMBEDDING_MODEL_NAME, input_type=input_type
            )
        except _RETRYABLE_VOYAGE_ERRORS:
            if attempt == max_attempts:
                raise
            delay = 2 ** (attempt - 1) + random.uniform(0, 0.5)
            logger.warning(
                "Voyage API call failed (attempt %d/%d), retrying in %.1fs",
                attempt, max_attempts, delay,
            )
            time.sleep(delay)


def embed_document(text: str, image_paths: list[str] | None = None) -> list[float]:
    """Embed a document for indexing, folding in all of its images when
    available (a piece like Head in the Clouds has more than one)."""
    content = [text]
    for image_path in image_paths or []:
        try:
            img = Image.open(image_path)
            img.load()  # Image.open() only reads the header; .load() forces
            # a full decode now, inside this try/except, so a file that's
            # corrupted past its header is caught and skipped here rather
            # than failing later inside the Voyage SDK call below - which
            # would raise outside this per-image guard and cause
            # index_documents' outer except to drop the ENTIRE document
            # (title, description, URL, everything), not just the image.
            content.append(img)
        except Exception:
            logger.exception("Could not open image %s; skipping it", image_path)
    result = _voyage_embed([content], input_type="document")
    return result.embeddings[0]


def embed_query(text: str) -> list[float]:
    """Embed a search query (text only)."""
    result = _voyage_embed([[text]], input_type="query")
    return result.embeddings[0]


# --- Chroma collection + content-change detection -------------------------

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="artworks",
    metadata={"description": "Inside the Paintbox artwork collection"},
)

# Sits next to the Chroma data (same persistent volume) so a reboot can tell
# whether the artwork content actually changed since it last indexed.
_SIGNATURE_PATH = Path(CHROMA_DB_PATH) / ".content_signature"

# index_documents() runs in a worker thread (via asyncio.to_thread, from
# both startup indexing and /reindex) - this keeps two overlapping runs
# from concurrently wiping and rebuilding the same collection.
_index_lock = threading.Lock()


def _content_signature(documents: list[dict]) -> str:
    """Hash of every document's text + image bytes + secret flag, plus the
    embedding model name so a model change also forces a reindex
    (mismatched vector sizes otherwise fail silently at query time)."""
    hasher = hashlib.sha256()
    hasher.update(EMBEDDING_MODEL_NAME.encode("utf-8"))
    for doc in sorted(documents, key=lambda d: d.get("source", "")):
        hasher.update(doc.get("content", "").encode("utf-8"))
        # Toggling an artwork's secret status alone (no other content
        # change) must still force a reindex, or the stale metadata value
        # would keep it wrongly visible/hidden until an unrelated edit.
        hasher.update(str(bool(doc.get("secret", False))).encode("utf-8"))
        for image_path in doc.get("image_paths") or []:
            try:
                hasher.update(Path(image_path).read_bytes())
            except OSError:
                logger.warning("Could not read %s while hashing content", image_path)
    return hasher.hexdigest()


def index_documents(documents: list[dict], force: bool = False) -> int:
    """Replace the collection's contents with the given documents, unless
    the content is unchanged since the last index (skip check bypassed by
    force=True, used by the manual /reindex endpoint).

    Each document must have a "content" key and may have an "image_paths"
    list; "title", "source", "subtitle", "url", and "series" are stored as
    metadata if present. Returns the number of documents indexed (or the
    existing count, if skipped).
    """
    if not documents:
        print("No documents to index!")
        return 0

    with _index_lock:
        signature = _content_signature(documents)
        if not force and collection.count() > 0 and _SIGNATURE_PATH.exists():
            if _SIGNATURE_PATH.read_text().strip() == signature:
                print("Artwork content unchanged since last index — skipping reindex")
                return collection.count()

        # Embed everything into a staging list FIRST, before touching the
        # existing collection. index_documents used to delete the whole
        # collection up front and rebuild it document-by-document - if
        # Voyage was down for most of the run, the old (complete) index
        # was already gone before the replacement failed to fill back in.
        # This way, a bad run leaves the existing collection untouched.
        prepared = []
        for i, doc in enumerate(documents):
            try:
                embedding = embed_document(doc["content"], doc.get("image_paths"))
                prepared.append((i, doc, embedding))
            except Exception:
                logger.exception("Failed to embed document %d (%s)", i, doc.get("title", "untitled"))

        if not prepared:
            logger.error("Every document failed to embed - keeping the existing collection")
            return collection.count()

        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        indexed = 0
        for i, doc, embedding in prepared:
            try:
                collection.add(
                    documents=[doc["content"]],
                    embeddings=[embedding],
                    metadatas=[{
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                        "subtitle": doc.get("subtitle", ""),
                        "url": doc.get("url", ""),
                        "series": doc.get("series", ""),
                        "secret": bool(doc.get("secret", False)),
                    }],
                    ids=[f"doc_{i}"],
                )
                indexed += 1
            except Exception:
                logger.exception("Failed to add document %d (%s)", i, doc.get("title", "untitled"))

        if indexed == len(documents):
            _SIGNATURE_PATH.write_text(signature)
        else:
            # Partial failure: don't record a signature, so a retry on the
            # next boot re-embeds everything instead of trusting an
            # incomplete index.
            _SIGNATURE_PATH.unlink(missing_ok=True)
        print(f"Indexed {indexed}/{len(documents)} documents")
        return indexed


# --- stats -----------------------------------------------------------------

def get_collection_stats() -> dict:
    """Return the current document count and collection name."""
    return {
        "document_count": collection.count(),
        "collection_name": collection.name,
    }

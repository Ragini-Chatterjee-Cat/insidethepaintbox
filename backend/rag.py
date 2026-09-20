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
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import voyageai
from PIL import Image

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "voyage-multimodal-3")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")

_voyage_api_key = os.environ.get("VOYAGE_API_KEY")
if not _voyage_api_key:
    raise ValueError("VOYAGE_API_KEY environment variable not set!")
_voyage_client = voyageai.Client(api_key=_voyage_api_key)


# --- embedding -----------------------------------------------------------

def embed_document(text: str, image_paths: Optional[List[str]] = None) -> List[float]:
    """Embed a document for indexing, folding in all of its images when
    available (a piece like Head in the Clouds has more than one)."""
    content = [text]
    for image_path in image_paths or []:
        try:
            content.append(Image.open(image_path))
        except Exception:
            logger.exception("Could not open image %s; skipping it", image_path)
    result = _voyage_client.multimodal_embed(
        [content], model=EMBEDDING_MODEL_NAME, input_type="document"
    )
    return result.embeddings[0]


def embed_query(text: str) -> List[float]:
    """Embed a search query (text only)."""
    result = _voyage_client.multimodal_embed(
        [[text]], model=EMBEDDING_MODEL_NAME, input_type="query"
    )
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


def _content_signature(documents: List[Dict]) -> str:
    """Hash of every document's text + image bytes, plus the embedding model
    name so a model change also forces a reindex (mismatched vector sizes
    otherwise fail silently at query time)."""
    hasher = hashlib.sha256()
    hasher.update(EMBEDDING_MODEL_NAME.encode("utf-8"))
    for doc in sorted(documents, key=lambda d: d.get("source", "")):
        hasher.update(doc.get("content", "").encode("utf-8"))
        for image_path in doc.get("image_paths") or []:
            try:
                hasher.update(Path(image_path).read_bytes())
            except OSError:
                logger.warning("Could not read %s while hashing content", image_path)
    return hasher.hexdigest()


def index_documents(documents: List[Dict], force: bool = False) -> int:
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

    signature = _content_signature(documents)
    if not force and collection.count() > 0 and _SIGNATURE_PATH.exists():
        if _SIGNATURE_PATH.read_text().strip() == signature:
            print("Artwork content unchanged since last index — skipping reindex")
            return collection.count()

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    indexed = 0
    for i, doc in enumerate(documents):
        try:
            embedding = embed_document(doc["content"], doc.get("image_paths"))
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
            logger.exception("Failed to index document %d (%s)", i, doc.get("title", "untitled"))

    if indexed == len(documents):
        _SIGNATURE_PATH.write_text(signature)
    else:
        # Partial failure: don't record a signature, so a retry on the next
        # boot re-embeds everything instead of trusting an incomplete index.
        _SIGNATURE_PATH.unlink(missing_ok=True)
    print(f"Indexed {indexed}/{len(documents)} documents")
    return indexed


# --- stats -----------------------------------------------------------------

def get_collection_stats() -> Dict:
    """Return the current document count and collection name."""
    return {
        "document_count": collection.count(),
        "collection_name": collection.name,
    }

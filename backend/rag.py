"""
RAG (Retrieval-Augmented Generation) Module
Handles document indexing and embedding. Retrieval and generation for
chat happen in the LangGraph agent (see agent/nodes.py, tools/).

Embeddings are multimodal (text + image) via Voyage AI's voyage-multimodal-3,
so a search like "something dark and emotional" can match an artwork's actual
image, not just whatever its written description happens to say.
"""

import logging
import os
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


def embed_document(text: str, image_path: Optional[str] = None) -> List[float]:
    """Embed a document for indexing, folding in its image when available."""
    content = [text]
    if image_path:
        try:
            content.append(Image.open(image_path))
        except Exception:
            logger.exception("Could not open image %s; embedding text only", image_path)
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


chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# Drop and recreate on boot: documents are always fully reindexed at startup
# anyway (see app.py's lifespan), and this guarantees a clean collection
# rather than one left over from a previous embedding model with a different
# vector size (Chroma locks a collection's dimensionality on first write).
try:
    chroma_client.delete_collection("artworks")
except Exception:
    pass
collection = chroma_client.get_or_create_collection(
    name="artworks",
    metadata={"description": "Inside the Paintbox artwork collection"},
)


def index_documents(documents: List[Dict]) -> int:
    """Replace the collection's contents with the given documents.

    Each document must have a "content" key and may have an "image_path"
    key; "title", "source", "subtitle", "url", and "series" are stored as
    metadata if present. Returns the number of documents indexed.
    """
    if not documents:
        print("No documents to index!")
        return 0

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    indexed = 0
    for i, doc in enumerate(documents):
        try:
            embedding = embed_document(doc["content"], doc.get("image_path"))
            collection.add(
                documents=[doc["content"]],
                embeddings=[embedding],
                metadatas=[{
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "subtitle": doc.get("subtitle", ""),
                    "url": doc.get("url", ""),
                    "series": doc.get("series", ""),
                }],
                ids=[f"doc_{i}"],
            )
            indexed += 1
        except Exception:
            logger.exception("Failed to index document %d (%s)", i, doc.get("title", "untitled"))

    print(f"Indexed {indexed}/{len(documents)} documents")
    return indexed


def get_collection_stats() -> Dict:
    """Return the current document count and collection name."""
    return {
        "document_count": collection.count(),
        "collection_name": collection.name,
    }

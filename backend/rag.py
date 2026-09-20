"""
RAG (Retrieval-Augmented Generation) Module
Handles document indexing and embedding. Retrieval and generation for
chat happen in the LangGraph agent (see agent/nodes.py, tools/).
"""

import logging
import os
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="artworks",
    metadata={"description": "Inside the Paintbox artwork collection"},
)


def index_documents(documents: List[Dict]) -> int:
    """Replace the collection's contents with the given documents.

    Each document must have a "content" key; "title", "source", "subtitle",
    "url", and "series" are stored as metadata if present. Returns the
    number of documents indexed.
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
            embedding = embedding_model.encode(doc["content"]).tolist()
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

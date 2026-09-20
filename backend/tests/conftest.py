"""
Shared pytest fixtures and test-time stubs.

Purpose: rag.py and agent/nodes.py both construct real external clients
(ChromaDB, Voyage AI, Anthropic) as a side effect of being imported, and
raise at import time if their API keys are unset. This file stubs
ChromaDB (so tests never touch disk or need the real package installed)
and supplies fake-but-present API keys, before any test module imports
anything from the backend - so the whole suite runs fast, deterministically,
and without network access or real credentials.
"""
import os
import sys
from unittest.mock import MagicMock

# Fake API keys before anything under test gets imported - rag.py and
# agent/nodes.py both raise ValueError at import time otherwise.
os.environ.setdefault("VOYAGE_API_KEY", "test-voyage-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

# Stub chromadb: rag.py constructs a real PersistentClient() at import
# time. Replace it with a fake so tests never touch disk or need the
# real (heavy) chromadb package installed - individual tests that care
# about collection behavior configure this mock's return values themselves.
if "chromadb" not in sys.modules:
    _fake_collection = MagicMock(name="chroma_collection")
    _fake_collection.count.return_value = 0
    _fake_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}

    _fake_chromadb = MagicMock(name="chromadb")
    _fake_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = (
        _fake_collection
    )
    sys.modules["chromadb"] = _fake_chromadb

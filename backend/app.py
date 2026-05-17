"""
Inside the Paintbox - RAG Chatbot API
FastAPI backend for the art website chatbot
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import json
import secrets
import traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from rag import query_rag, index_documents, get_collection_stats
from document_loader import load_all_artworks, load_about_page
from agent import chat_with_memory, get_conversation_history, clear_conversation
from agent.nodes import PREFS_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Index documents automatically
    print("Starting up... Indexing artwork documents...")
    try:
        # Load documents from the website
        docs = load_all_artworks("../frontend/")

        # Also load about page
        about = load_about_page("../frontend/")
        if about:
            docs.append(about)

        # Index them
        if docs:
            index_documents(docs)
            print(f"Successfully indexed {len(docs)} documents!")
        else:
            print("Warning: No documents found to index!")
    except Exception as e:
        print(f"Error during startup indexing: {e}")

    yield

    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Inside the Paintbox API",
    description="RAG Chatbot API for Ragini Chatterjee's Art Portfolio",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS (allow your Netlify site to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",      # Local development
        "http://127.0.0.1:5500",      # Local development
        "http://localhost:3000",       # Local development
        "https://*.netlify.app",       # Netlify preview URLs
        "*"                            # Allow all (update in production!)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequestV2(BaseModel):
    message: str
    thread_id: str  # Persistent thread ID for memory

class ChatResponseV2(BaseModel):
    response: str
    thread_id: str

class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[ChatMessage]

class IndexResponse(BaseModel):
    status: str
    documents_indexed: int

class HealthResponse(BaseModel):
    status: str
    documents_count: int


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    stats = get_collection_stats()
    return HealthResponse(
        status="ok",
        documents_count=stats["document_count"]
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    stats = get_collection_stats()
    return HealthResponse(
        status="healthy",
        documents_count=stats["document_count"]
    )



@app.post("/chat/v2", response_model=ChatResponseV2)
async def chat_v2(request: ChatRequestV2):
    """
    Chat endpoint with persistent memory (LangGraph)
    Uses thread_id for conversation persistence across sessions
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not request.thread_id or not request.thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required")

    try:
        response = chat_with_memory(request.message, request.thread_id)
        return ChatResponseV2(response=response, thread_id=request.thread_id)
    except Exception as e:
        print(f"Error in chat v2 endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Sorry, I encountered an error. Please try again."
        )


@app.get("/chat/history/{thread_id}", response_model=HistoryResponse)
async def get_history(thread_id: str):
    """
    Get conversation history for a thread
    """
    try:
        history = get_conversation_history(thread_id)
        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in history]
        return HistoryResponse(thread_id=thread_id, messages=messages)
    except Exception as e:
        print(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving history")


@app.delete("/chat/history/{thread_id}")
async def delete_history(thread_id: str):
    """
    Clear conversation history for a thread
    """
    try:
        success = clear_conversation(thread_id)
        if success:
            return {"status": "cleared", "thread_id": thread_id}
        raise HTTPException(status_code=500, detail="Failed to clear history")
    except Exception as e:
        print(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail="Error clearing history")


@app.post("/reindex", response_model=IndexResponse)
async def reindex_documents():
    """
    Manually reindex all documents
    Call this if you've updated your artwork pages
    """
    try:
        docs = load_all_artworks("../frontend/")
        about = load_about_page("../frontend/")
        if about:
            docs.append(about)

        count = index_documents(docs)
        return IndexResponse(status="success", documents_indexed=count)
    except Exception as e:
        print(f"Error reindexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed documents"""
    return get_collection_stats()


# ---------------------------------------------------------------------------
# Admin helpers
# ---------------------------------------------------------------------------

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

def _check_admin(request: Request):
    auth = request.headers.get("Authorization", "")
    if not ADMIN_TOKEN or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth[7:]
    if not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/admin/pending")
async def get_pending_commissions(request: Request):
    """Return all unreviewed commission requests (admin only)."""
    _check_admin(request)
    commissions = []
    for f in sorted(PREFS_DIR.glob("*_commission.json")):
        try:
            data = json.loads(f.read_text())
            # Only show ones that haven't been replied to yet
            reply_file = PREFS_DIR / f"{data['thread_id']}_reply.json"
            if not reply_file.exists():
                commissions.append(data)
        except Exception:
            continue
    return {"commissions": commissions}


@app.post("/admin/respond/{thread_id}")
async def respond_to_commission(thread_id: str, request: Request):
    """Send Ragini's personal reply back into the visitor's chat thread (admin only)."""
    _check_admin(request)
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    reply_file = PREFS_DIR / f"{thread_id}_reply.json"
    reply_file.write_text(json.dumps({
        "thread_id": thread_id,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }))
    return {"status": "sent", "thread_id": thread_id}


@app.get("/chat/updates/{thread_id}")
async def poll_for_reply(thread_id: str):
    """Visitor polls this to check whether Ragini has replied to their commission."""
    reply_file = PREFS_DIR / f"{thread_id}_reply.json"
    if reply_file.exists():
        try:
            data = json.loads(reply_file.read_text())
            return {"has_reply": True, "message": data["message"]}
        except Exception:
            pass
    return {"has_reply": False}


# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

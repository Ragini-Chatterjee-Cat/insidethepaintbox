"""
Inside the Paintbox - RAG Chatbot API
FastAPI backend for the art website chatbot
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
import db
from rag import index_documents, get_collection_stats
from document_loader import load_all_artworks, load_about_page
from agent import chat_with_memory, get_conversation_history, clear_conversation


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    db.setup_tables()
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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "https://insidethepaintbox.netlify.app",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
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



MAX_MESSAGE_LENGTH = 500   # chars — long prompts signal off-topic abuse
MAX_USER_TURNS = 20  # max user messages per session


@app.post("/chat/v2", response_model=ChatResponseV2)
@limiter.limit("15/minute")
async def chat_v2(request: Request, body: ChatRequestV2):
    """Chat endpoint with persistent memory (LangGraph)."""
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not body.thread_id or not body.thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required")

    if len(body.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long (max {MAX_MESSAGE_LENGTH} characters)."
        )

    try:
        history = get_conversation_history(body.thread_id)
        user_turns = sum(1 for m in history if m["role"] == "user")
        if user_turns >= MAX_USER_TURNS:
            return ChatResponseV2(
                response=(
                    "We've had quite the gallery tour! This session has reached its limit. "
                    "Feel free to refresh the page to start a fresh conversation."
                ),
                thread_id=body.thread_id,
            )

        response = chat_with_memory(body.message, body.thread_id)
        return ChatResponseV2(response=response, thread_id=body.thread_id)
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


# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

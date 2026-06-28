"""
RAG (Retrieval-Augmented Generation) Module
Handles document indexing, embedding, and retrieval
"""

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
from document_loader import SERIES_ARTWORK_MAP

# Initialize the embedding model (runs locally, free)
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# Initialize ChromaDB (local vector database)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="artworks",
    metadata={"description": "Inside the Paintbox artwork collection"}
)


def get_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set!")
    return anthropic.Anthropic(api_key=api_key)


def index_documents(documents: List[Dict]):
    """Index documents into the vector database"""
    if not documents:
        print("No documents to index!")
        return 0

    # Clear existing documents
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Add new documents
    for i, doc in enumerate(documents):
        try:
            # Create embedding
            embedding = embedding_model.encode(doc["content"]).tolist()

            # Add to collection
            collection.add(
                documents=[doc["content"]],
                embeddings=[embedding],
                metadatas=[{
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "subtitle": doc.get("subtitle", ""),
                    "url": doc.get("url", ""),
                    "series": doc.get("series", "")
                }],
                ids=[f"doc_{i}"]
            )
        except Exception as e:
            print(f"Error indexing document {i}: {e}")

    print(f"Indexed {len(documents)} documents")
    return len(documents)


def search_similar(query: str, n_results: int = 3) -> List[Dict]:
    """Search for similar documents"""
    # Embed the query
    query_embedding = embedding_model.encode(query).tolist()

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    similar_docs = []
    for i in range(len(results["documents"][0])):
        similar_docs.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return similar_docs


def generate_response(question: str, context_docs: List[Dict], conversation_history: List[Dict] = None) -> str:
    """
    Generate a response using Two-Step Chain-of-Thought Reasoning.
    Step 1: Analyze the question and reason about what the user wants
    Step 2: Generate the final response based on that reasoning
    """
    # Filter out low-confidence documents (distance > 1.5 means low relevance)
    CONFIDENCE_THRESHOLD = 1.5
    confident_docs = [doc for doc in context_docs if doc.get("distance", 0) < CONFIDENCE_THRESHOLD]

    # Build context from retrieved documents
    if confident_docs:
        context = "\n\n---\n\n".join([doc["content"] for doc in confident_docs])
        context_note = ""
    else:
        # No confident matches - let the model know
        context = "\n\n---\n\n".join([doc["content"] for doc in context_docs[:1]])  # Use top result anyway
        context_note = "\n\n(Note: This context may not be directly relevant.)"

    try:
        client = get_anthropic_client()

        # Build conversation history string for reasoning
        history_str = ""
        if conversation_history and len(conversation_history) > 0:
            recent = conversation_history[-6:]  # Last 3 exchanges
            history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent])

        # Build series reference for the LLM
        series_ref = "\n".join(
            f"- {name}: {', '.join(f.replace('.html', '') for f in files)}"
            for name, files in SERIES_ARTWORK_MAP.items()
        )

        # ============ STEP 1: REASONING ============
        reasoning_prompt = f"""You are analyzing a conversation with a visitor to an art portfolio website.

ARTWORK SERIES REFERENCE:
{series_ref}

CONVERSATION HISTORY:
{history_str if history_str else "(This is the first message)"}

CURRENT QUESTION: "{question}"

AVAILABLE CONTEXT ABOUT ARTWORKS:
{context}{context_note}

Think step by step:
1. INTENT: What is the user actually asking about? If they use words like "it", "that", "this", "them", "both", or short phrases like "yes", "tell me more" - what are they referring to from the conversation history?

2. RELEVANT INFO: What specific information from the context answers their question? Quote the relevant parts.

3. KEY POINTS: What 1-2 key points should be in the response?

Write your analysis concisely:"""

        reasoning_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": reasoning_prompt}],
            max_tokens=300,
        )
        reasoning = reasoning_response.content[0].text
        print(f"[REASONING]: {reasoning[:200]}...")  # Debug log (truncated)

        # ============ STEP 2: FINAL RESPONSE ============
        response_prompt = f"""You are a friendly art assistant for "Inside the Paintbox" by Ragini Chatterjee.

Based on this analysis of what the visitor wants:
---
{reasoning}
---

CONTEXT ABOUT ARTWORKS:
{context}

Now write your response to the visitor. Guidelines:
- Be warm and conversational, like giving a gallery tour
- Keep it concise (1-3 sentences) unless they asked for more detail
- Only include information from the context - don't make things up
- If the context doesn't have the answer, say so honestly
- Only include artwork URLs when the user asks to see, view, or requests a link to an artwork. Don't include links in every response.

Your response:"""

        final_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": response_prompt}],
            max_tokens=250,
        )

        return final_response.content[0].text

    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble responding right now. Please try again in a moment!"


def build_search_query(question: str, conversation_history: List[Dict] = None) -> str:
    """
    Build an enhanced search query that includes conversation context.
    This helps with follow-up questions like "What colors does it have?"
    by including context about what "it" refers to.
    """
    if not conversation_history:
        return question

    # Get the last few exchanges to understand context
    recent_history = conversation_history[-4:]  # Last 2 exchanges

    # Extract key context from recent messages
    context_parts = []
    for msg in recent_history:
        if msg["role"] == "user":
            context_parts.append(msg["content"])

    # Combine current question with recent context for better search
    # Put current question first (most important), then add context
    enhanced_query = question + " " + " ".join(context_parts)

    return enhanced_query


def query_rag(question: str, conversation_history: List[Dict] = None) -> str:
    """Main RAG query function"""
    # 1. Build enhanced search query with conversation context
    search_query = build_search_query(question, conversation_history)

    # 2. Search for relevant documents using enhanced query
    similar_docs = search_similar(search_query, n_results=3)

    # 3. Generate response with context and conversation history
    response = generate_response(question, similar_docs, conversation_history)

    return response


def get_collection_stats():
    """Get statistics about the indexed collection"""
    return {
        "document_count": collection.count(),
        "collection_name": collection.name
    }

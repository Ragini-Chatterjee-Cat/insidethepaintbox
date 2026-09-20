# Inside the Paintbox - Backend

A LangGraph agent that answers visitor questions about Ragini Chatterjee's
artwork, backed by multimodal (text + image) semantic search over the
site's own HTML pages.

This file replaces an older description of the system (single-pass RAG,
ChromaDB + HuggingFace embeddings, Groq/Llama, hosted on Render.com) that
no longer matches the code below.

---

## ARCHITECTURE DIAGRAM

```
                              ART WEBSITE
    +-----------------------------------------------------------------+
    |                                                                 |
    |   +---------------------------------------------------------+  |
    |   |                 NETLIFY (static hosting)                |  |
    |   |                                                          |  |
    |   |   index.html --- css/ --- js/chat.js --- images/        |  |
    |   |                              |                           |  |
    |   +------------------------------|---------------------------+  |
    |                                  |                              |
    +----------------------------------|------------------------------+
                                       |
                                       | HTTPS  POST /chat/v2
                                       | {"message": "...", "thread_id": "..."}
                                       v
    +-----------------------------------------------------------------+
    |                    RAILWAY (Docker, FastAPI)                     |
    |                                                                   |
    |   app.py: rate-limited, validates input, calls chat_with_memory() |
    |                              |                                    |
    |                              v                                    |
    |   agent/graph.py: LangGraph state machine                         |
    |     load_preferences -> classify -> [react_node | general_chat    |
    |       | commission_intake] -> extract_preferences                 |
    |       -> save_preferences -> END                                  |
    |                              |                                    |
    +------------------------------|------------------------------------+
                                   |
              +--------------------+---------------------+----------------+
              |                    |                      |                |
              v                    v                      v                v
    +------------------+ +-------------------+  +------------------+ +-----------+
    |  Anthropic API    | |    Voyage AI       |  |    ChromaDB      | | Postgres  |
    |  Claude Haiku 4.5  | | voyage-multimodal-3|  | (local disk,     | | (or JSON  |
    |  classify/generate | | text+image embed   |  |  persistent      | | fallback) |
    +------------------+ +-------------------+  |  volume)         | | prefs +   |
                                   |               +------------------+ | LangGraph |
                                   v                        ^           | checkpoint|
                          embedding vector -------------------           +-----------+
                                                    queried by react_node's tools
```

The visitor never talks to Voyage, Chroma, Postgres, or Anthropic
directly — every one of those is a hop `agent/nodes.py` or a tool class
makes on the visitor's behalf, inside the single `/chat/v2` request.

---

## WHAT EACH FILE DOES

```
backend/
|
|-- app.py                    FastAPI app - the only HTTP-facing file.
|     POST /chat/v2             one turn of the LangGraph agent
|     GET  /chat/history/{id}   read a thread's message history
|     DELETE /chat/history/{id} clear a thread's saved preferences
|     POST /reindex             force a full reindex of artwork content
|     GET  /health, /, /stats   health checks + collection stats
|
|-- db.py                     Postgres persistence for visitor preferences,
|                              with a JSON-file fallback for local dev.
|                              (LangGraph's own message history is
|                              persisted separately - see agent/graph.py.)
|
|-- document_loader.py        Parses frontend/**/*.html into documents:
|                              title, description, series, image paths,
|                              secret flag. The only file that touches
|                              the site's HTML.
|
|-- rag.py                    Embeds documents/queries via Voyage AI and
|                              stores/searches them in a local ChromaDB
|                              collection. Skips reindexing when artwork
|                              content hasn't changed since last time.
|
|-- agent/
|   |-- __init__.py           Public interface: chat_with_memory(),
|   |                         get_conversation_history(), clear_conversation().
|   |                         app.py only ever calls into agent/ through here.
|   |-- graph.py               Builds the LangGraph state machine and its
|   |                          checkpointer (Postgres if configured, else
|   |                          in-memory).
|   |-- nodes.py                PaintboxAgent - the actual node logic and
|   |                            LLM calls (classify, react_node/tool loop,
|   |                            general_chat, commission_intake,
|   |                            extract_preferences, save_preferences).
|   |-- prompts.py              Every system prompt, one per node.
|   +-- state.py                AgentState - the TypedDict that flows
|                                through the graph.
|
|-- tools/                    LangChain tool classes the agent can call
|   |                         mid-conversation (react_node only):
|   |-- search_artworks.py       semantic search, by mood/theme
|   |-- get_artwork_details.py   look up one artwork by title
|   |-- filter_by_series.py      list a named series
|   |-- recommend_similar.py     "more like this one"
|   |-- commission_info.py       static commission-request text
|   |-- reveal_secret_artwork    hidden pieces, only if explicitly asked
|   +-- base.py                  shared collection/embed_query/constants
|
|-- requirements.txt          Python dependencies
|-- Dockerfile / nixpacks.toml  container build (Railway)
+-- chroma_db/                 vector database storage (persisted via a
                                Railway volume mounted at
                                /app/backend/chroma_db - see below)
```

---

## HOW A CHAT TURN ACTUALLY WORKS

This isn't the classic "embed the question, stuff context into one
prompt, generate an answer" RAG shape. It's closer to **tool-augmented
generation**: the model decides *whether* and *what* to search, turn by
turn, rather than every message automatically getting retrieval results
shoved into it.

1. **Indexing** (once, at boot, and skipped on later boots if nothing
   changed): `document_loader.py` walks every artwork/series/about page,
   pulls out title/description/images, and `rag.py` embeds each one
   (text *and* image together, via `voyage-multimodal-3`) into ChromaDB.

2. **A message arrives** at `POST /chat/v2`. `classify` asks Haiku a
   cheap yes/no-shaped question: is this about a specific **artwork**, a
   **commission**, or **general** chat? Only the "artwork" path touches
   any tool at all.

3. **`react_node`** (the artwork path) gives Haiku up to 5 tool-calling
   iterations against the six tools in `tools/`. Not all of them do
   vector search - `filter_by_series` is a plain metadata filter,
   `get_artwork_details` re-ranks embedding results by literal title
   match, and `reveal_secret_artwork` only ever runs if the visitor
   explicitly asked about something hidden.

4. **Generation** happens inline - whatever a tool returned gets appended
   to the message history, and Haiku writes the final reply directly.
   There's no separate "stuff retrieved context into a template" step.

5. **Memory** on top: after every turn, `extract_preferences` asks Haiku
   to pull out liked series/mentioned artworks from the last few
   messages, and `save_preferences` persists that (via `db.py`) so the
   next turn's `load_preferences` can personalize the reply.

---

## RUNNING LOCALLY

```bash
cd backend
pip install -r requirements.txt
```

Required environment variables (put them in `backend/.env` - it's
gitignored):

| Variable            | Required | Notes                                                        |
| -------------------- | -------- | -------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`  | yes      | Claude Haiku - classification, generation, tool calls.         |
| `VOYAGE_API_KEY`     | yes      | Multimodal (text+image) embeddings for indexing and search.    |
| `POSTGRES_URI`       | no       | Enables persistent conversation memory + prefs across restarts and multiple instances. Falls back to in-memory checkpointing + local JSON files if unset. |
| `CHROMA_DB_PATH`     | no       | Where the vector index lives on disk. Defaults to `./chroma_db`. Needs a persistent volume in production or it's rebuilt on every deploy. |
| `EMBEDDING_MODEL`    | no       | Defaults to `voyage-multimodal-3`. |
| `PREFS_DIR`          | no       | Where the JSON-fallback preference files live. Defaults to `./user_prefs`. |

Then:

```bash
uvicorn app:app --reload --port 8000
```

The first request after startup may see an empty or partial artwork
collection while background indexing finishes - `GET /health` reports
the current document count.

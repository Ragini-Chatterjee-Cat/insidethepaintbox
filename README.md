# Inside the Paintbox
My art portfolio : a website I creqted during my first CS50 course and then improve to add a chatbot using RAG with multi model embeddings to help explain my artwork.

## Structure

- **`frontend/`** - the site itself: static HTML/CSS/JS, one page per
  artwork/series, deployed to Netlify. No build step.
- **`backend/`** - a LangGraph agent (FastAPI, Claude Haiku, Voyage AI
  multimodal embeddings, ChromaDB, optional Postgres for memory),
  deployed to Railway. See **[`backend/README.md`](backend/README.md)**
  for the architecture, what each file does, and how to run it locally.

## Live

- Site: https://insidethepaintbox.netlify.app
- Chat API: https://art-website-production.up.railway.app

# Inside the Paintbox

Ragini Chatterjee's art portfolio - a static site with an AI chat widget
that answers visitor questions about the artwork.

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

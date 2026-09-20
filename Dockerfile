FROM python:3.11-slim

# Pinned uv binary, copied straight from astral's distroless image - no pip
# install step needed just to get the installer itself.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

WORKDIR /app/backend

# Copy only the dependency manifest first so this layer stays cached across
# rebuilds unless pyproject.toml/uv.lock actually change.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Copy entire project (backend needs access to ../frontend for document
# loading). .dockerignore keeps this from pulling in a host-built .venv.
COPY . /app

WORKDIR /app/backend

# Use the venv uv just built without needing `uv run` at container start
ENV PATH="/app/backend/.venv/bin:$PATH"

EXPOSE 8000

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

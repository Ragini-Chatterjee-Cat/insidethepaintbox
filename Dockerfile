FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch FIRST from the CPU wheel index
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install numpy before other deps to avoid version conflicts
RUN pip install --no-cache-dir numpy==1.26.4

# Copy requirements and install dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy entire project (backend needs access to HTML files for document loading)
COPY . .

# Set working directory to backend
WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Run the app
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

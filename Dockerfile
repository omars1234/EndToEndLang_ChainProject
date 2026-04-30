# syntax=docker/dockerfile:1.9
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies needed for unstructured, pymupdf, pdf2image, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

# Install dependencies (this is the heavy part)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# COPY . .

COPY app.py .
COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

# Optional: Use non-root user for better security
# RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
# USER appuser

# Adjust CMD based on your app type:

# 1. If it's a FastAPI app:
#CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]

# 2. If it's a Streamlit app:
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# 3. If it's a simple script:
# CMD ["uv", "run", "python", "main.py"]
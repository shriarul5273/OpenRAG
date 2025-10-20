# syntax=docker/dockerfile:1.6

FROM python:3.10-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends build-essential git curl \ 
       poppler-utils libreoffice pandoc \ 
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \ 
    && pip install .

EXPOSE 8000

ENV OPENRAG_CHROMA_PERSIST_DIR=/app/.chroma

CMD ["uvicorn", "openrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

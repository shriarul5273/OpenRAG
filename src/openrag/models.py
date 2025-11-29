"""Shared domain models used across the OpenRAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata captured for an ingested document."""

    document_id: str
    source_path: str
    media_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    access_label: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """Normalized chunk of document text ready for embedding."""

    chunk_id: str
    text: str
    document_metadata: DocumentMetadata
    order: int
    chunk_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk returned from the vector store during retrieval."""

    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class Answer:
    """Structured answer produced by the LLM with citations."""

    text: str
    citations: Sequence[RetrievedChunk]
    query_id: str
    latency_ms: float
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    trace_id: str | None = None

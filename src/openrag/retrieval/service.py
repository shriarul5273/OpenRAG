"""Retrieval orchestration built on top of embedding stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from openrag.embeddings import EmbeddingStore
from openrag.models import RetrievedChunk


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval."""

    top_k: int = 5


class Retriever(Protocol):
    """Retrieve relevant chunks for a query string."""

    def retrieve(self, query: str, *, top_k: int | None = None) -> Sequence[RetrievedChunk]:
        """Return the top-k retrieved chunks."""


class ChromaRetriever:
    """Retriever backed by a Chroma embedding store."""

    def __init__(self, store: EmbeddingStore, config: RetrievalConfig | None = None) -> None:
        self._store = store
        self._config = config or RetrievalConfig()

    def retrieve(self, query: str, *, top_k: int | None = None) -> Sequence[RetrievedChunk]:
        limit = top_k or self._config.top_k
        return self._store.similarity_search(query, top_k=limit)

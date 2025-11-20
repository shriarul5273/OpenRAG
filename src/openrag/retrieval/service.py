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
    rerank_lexical: bool = False
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_n: int = 10
    cross_encoder_device: str | None = None


class Retriever(Protocol):
    """Retrieve relevant chunks for a query string."""

    def retrieve(self, query: str, *, top_k: int | None = None, dataset_id: str | None = None) -> Sequence[RetrievedChunk]:
        """Return the top-k retrieved chunks."""


class ChromaRetriever:
    """Retriever backed by a Chroma embedding store."""

    def __init__(self, store: EmbeddingStore, config: RetrievalConfig | None = None) -> None:
        self._store = store
        self._config = config or RetrievalConfig()
        self._ce = None
        if self._config.use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._ce = CrossEncoder(self._config.cross_encoder_model, device=self._config.cross_encoder_device)
            except Exception:
                # Fallback if not available; keep running
                self._ce = None

    def retrieve(self, query: str, *, top_k: int | None = None, dataset_id: str | None = None) -> Sequence[RetrievedChunk]:
        limit = top_k or self._config.top_k
        items = list(self._store.similarity_search(query, top_k=limit, dataset_id=dataset_id))
        if self._config.use_cross_encoder and self._ce and items:
            # Rerank top-n retrieved items via CrossEncoder
            top_n = min(self._config.cross_encoder_top_n, len(items))
            pairs = [(query, rc.chunk.text) for rc in items[:top_n]]
            scores = self._ce.predict(pairs)
            scored = list(zip(items[:top_n], scores, strict=False))
            scored.sort(key=lambda x: float(x[1]), reverse=True)
            reranked = [rc for rc, _ in scored] + items[top_n:]
            items = reranked
        elif self._config.rerank_lexical and items:
            tokens = set(query.lower().split())
            items.sort(key=lambda rc: -_token_overlap_score(tokens, rc.chunk.text))
        return items


def _token_overlap_score(query_tokens: set[str], text: str) -> float:
    tokens = set(text.lower().split())
    if not tokens:
        return 0.0
    overlap = len(query_tokens.intersection(tokens))
    return overlap / max(len(query_tokens), 1)

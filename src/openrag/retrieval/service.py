"""Retrieval orchestration built on top of embedding stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from openrag.embeddings import EmbeddingStore
from openrag.models import RetrievedChunk


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval."""

    top_k: int = 5
    rerank_lexical: bool = False
    lexical_blend_weight: float = 0.35
    max_top_k: int | None = 10
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_n: int = 10
    cross_encoder_device: str | None = None


class Retriever(Protocol):
    """Retrieve relevant chunks for a query string."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        dataset_id: str | None = None,
        rerank: Literal["none", "lexical", "cross_encoder"] | None = None,
        lexical_weight: float | None = None,
        cross_encoder_top_n: int | None = None,
        access_labels: Sequence[str] | None = None,
        max_top_k: int | None = None,
    ) -> Sequence[RetrievedChunk]:
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

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        dataset_id: str | None = None,
        rerank: Literal["none", "lexical", "cross_encoder"] | None = None,
        lexical_weight: float | None = None,
        cross_encoder_top_n: int | None = None,
        access_labels: Sequence[str] | None = None,
        max_top_k: int | None = None,
    ) -> Sequence[RetrievedChunk]:
        limit = top_k or self._config.top_k
        effective_max = self._config.max_top_k if max_top_k is None else max_top_k
        if effective_max:
            limit = min(limit, effective_max)
        limit = max(1, limit)
        items = list(self._store.similarity_search(query, top_k=limit, dataset_id=dataset_id, access_labels=access_labels))
        mode = rerank
        if mode is None:
            if self._config.use_cross_encoder:
                mode = "cross_encoder"
            elif self._config.rerank_lexical:
                mode = "lexical"
            else:
                mode = "none"
        weight = self._clamp_weight(lexical_weight if lexical_weight is not None else self._config.lexical_blend_weight)
        ce_top_n = cross_encoder_top_n or self._config.cross_encoder_top_n or len(items)
        if mode == "cross_encoder" and items:
            if self._ensure_cross_encoder(allow_override=True):
                top_n = min(ce_top_n, len(items))
                pairs = [(query, rc.chunk.text) for rc in items[:top_n]]
                scores = self._ce.predict(pairs)
                scored = list(zip(items[:top_n], scores, strict=False))
                scored.sort(key=lambda x: float(x[1]), reverse=True)
                items = [rc for rc, _ in scored] + items[top_n:]
            elif weight and self._config.rerank_lexical:
                mode = "lexical"
            else:
                mode = "none"
        if mode == "lexical" and items:
            # Rerank top-n retrieved items via CrossEncoder
            tokens = set(query.lower().split())
            scored: list[tuple[RetrievedChunk, float]] = []
            for rc in items:
                lexical = _token_overlap_score(tokens, rc.chunk.text)
                blended = (1.0 - weight) * rc.score + weight * lexical
                scored.append((rc, blended))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            items = [pair[0] for pair in scored]
        return items

    def _ensure_cross_encoder(self, allow_override: bool = False) -> bool:
        if self._ce is not None:
            return True
        if not self._config.use_cross_encoder and not allow_override:
            return False
        if not self._config.use_cross_encoder and allow_override:
            self._config.use_cross_encoder = True
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._ce = CrossEncoder(self._config.cross_encoder_model, device=self._config.cross_encoder_device)
            return True
        except Exception:
            self._ce = None
            return False

    @staticmethod
    def _clamp_weight(weight: float) -> float:
        if weight < 0.0:
            return 0.0
        if weight > 1.0:
            return 1.0
        return weight


def _token_overlap_score(query_tokens: set[str], text: str) -> float:
    tokens = set(text.lower().split())
    if not tokens:
        return 0.0
    overlap = len(query_tokens.intersection(tokens))
    return overlap / max(len(query_tokens), 1)

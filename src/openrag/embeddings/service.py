"""Embedding backends for OpenRAG."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings as LangChainEmbeddings

from openrag.models import DocumentChunk

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding backends."""

    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    use_model: bool = False
    device: str | None = None
    normalize: bool = True
    cache_folder: str | None = None


@dataclass(frozen=True)
class Embedding:
    """Vector representation of a document chunk."""

    chunk: DocumentChunk
    vector: Tuple[float, ...]


class EmbeddingBackend(Protocol):
    """Protocol describing embedding behaviour."""

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> Sequence[Embedding]:
        """Return embeddings for the provided chunks."""

    def embed_query(self, query: str) -> Tuple[float, ...]:
        """Return embedding vector for a query string."""


class HashEmbeddingBackend:
    """Deterministic lightweight embedding fallback used for testing."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()

    def _hash_to_vector(self, text: str) -> Tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeat = (self._config.dim + len(digest) - 1) // len(digest)
        raw = (digest * repeat)[: self._config.dim]
        vector = [byte / 255.0 for byte in raw]
        if self._config.normalize:
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vector = [value / norm for value in vector]
        return tuple(vector)

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> Sequence[Embedding]:
        return [Embedding(chunk=chunk, vector=self._hash_to_vector(chunk.text)) for chunk in chunks]

    def embed_query(self, query: str) -> Tuple[float, ...]:
        return self._hash_to_vector(query)


class QwenEmbeddingBackend:
    """Embedding backend that optionally leverages Qwen2.5 models via LangChain."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()
        self._delegate = HashEmbeddingBackend(self._config)
        self._client: LangChainEmbeddings | None = None
        if not self._config.use_model:
            LOGGER.info("QwenEmbeddingBackend running in hash-only mode.")
            return
        try:
            model_kwargs = {"device": self._config.device} if self._config.device else {}
            if self._config.cache_folder:
                model_kwargs["cache_dir"] = self._config.cache_folder
            self._client = HuggingFaceEmbeddings(
                model_name=self._config.model,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": self._config.normalize},
            )
            LOGGER.info("Loaded embedding model %s", self._config.model)
        except Exception as exc:  # pragma: no cover - defensive import/runtime guard
            LOGGER.warning("Falling back to hash embeddings: %s", exc)
            self._client = None

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> Sequence[Embedding]:
        if not chunks:
            return []
        if self._client is None:
            return self._delegate.embed_chunks(chunks)
        vectors = self._client.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            LOGGER.error(
                "Embedding backend returned %d vectors for %d chunks", len(vectors), len(chunks)
            )
            raise ValueError("Mismatch between number of chunks and embedding vectors")
        # Warn if produced dimension differs from configured dim
        if vectors and len(vectors[0]) != self._config.dim:
            LOGGER.warning(
                "Embedding dim mismatch: configured=%d, actual=%d",
                self._config.dim,
                len(vectors[0]),
            )
        return [
            Embedding(chunk=chunk, vector=self._normalize(tuple(vector)))
            for chunk, vector in zip(chunks, vectors)
        ]

    def embed_query(self, query: str) -> Tuple[float, ...]:
        if self._client is None:
            return self._delegate.embed_query(query)
        vector = tuple(self._client.embed_query(query))
        return self._normalize(vector)

    def _normalize(self, vector: Tuple[float, ...]) -> Tuple[float, ...]:
        if not self._config.normalize:
            return vector
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return tuple(value / norm for value in vector)

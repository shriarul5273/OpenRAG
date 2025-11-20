"""Embedding store implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Protocol, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Documents, Embeddings as ChromaEmbeddings, IDs, Metadatas

from openrag.embeddings.service import EmbeddingBackend
from openrag.models import DocumentChunk, DocumentMetadata, RetrievedChunk


class EmbeddingStore(Protocol):
    """Protocol for embedding persistence backends."""

    def upsert(self, chunks: Sequence[DocumentChunk], *, dataset_id: str | None = None) -> Sequence[str]:
        """Persist embeddings for the provided chunks."""

    def similarity_search(self, query: str, *, top_k: int = 5, dataset_id: str | None = None) -> Sequence[RetrievedChunk]:
        """Return the top-k similar chunks for the query string."""

    def reset(self, *, dataset_id: str | None = None) -> None:
        """Remove all stored embeddings."""

    def count(self) -> int:
        """Return total number of stored chunks."""

    def count_by_dataset(self) -> Mapping[str, int]:
        """Return a mapping of dataset_id to chunk count."""


class ChromaEmbeddingStore:
    """Chroma-backed embedding store."""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        collection_name: str = "openrag",
        *,
        client: ClientAPI | None = None,
        persist_directory: str | Path | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._backend = embedding_backend

    def upsert(self, chunks: Sequence[DocumentChunk], *, dataset_id: str | None = None) -> Sequence[str]:
        embeddings = self._backend.embed_chunks(chunks)
        ids: IDs = [embedding.chunk.chunk_id for embedding in embeddings]
        documents: Documents = [embedding.chunk.text for embedding in embeddings]
        metadatas: Metadatas = [
            self._serialize_chunk(embedding.chunk, dataset_id=dataset_id)
            for embedding in embeddings
        ]
        vectors: ChromaEmbeddings = [list(embedding.vector) for embedding in embeddings]
        self._collection.upsert(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
        return list(ids)

    def similarity_search(self, query: str, *, top_k: int = 5, dataset_id: str | None = None) -> Sequence[RetrievedChunk]:
        if top_k <= 0:
            return []
        vector = list(self._backend.embed_query(query))
        where = {"dataset_id": dataset_id} if dataset_id else None
        results = self._collection.query(query_embeddings=[vector], n_results=top_k, where=where)
        return self._deserialize_results(results)

    def reset(self, *, dataset_id: str | None = None) -> None:
        if dataset_id:
            self._collection.delete(where={"dataset_id": dataset_id})
        else:
            self._collection.delete(where={})

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def count_by_dataset(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        try:
            # paginate through metadatas only
            limit = 1000
            offset = 0
            while True:
                batch = self._collection.get(
                    include=["metadatas"],
                    limit=limit,
                    offset=offset,
                )
                metadatas = batch.get("metadatas") or []
                if not metadatas:
                    break
                for md in metadatas:
                    if not isinstance(md, dict):
                        continue
                    ds = str(md.get("dataset_id", "")) or "default"
                    counts[ds] = counts.get(ds, 0) + 1
                # if fewer than limit, last page
                if isinstance(metadatas, list) and len(metadatas) < limit:
                    break
                offset += limit
        except Exception:
            pass
        return counts

    def _serialize_chunk(self, chunk: DocumentChunk, *, dataset_id: str | None = None) -> MutableMapping[str, object]:
        metadata: MutableMapping[str, object] = {
            "document_id": chunk.document_metadata.document_id,
            "source_path": chunk.document_metadata.source_path,
            "media_type": chunk.document_metadata.media_type,
            "order": chunk.order,
            "doc_extra": self._dumps(chunk.document_metadata.extra),
            "chunk_metadata": self._dumps(chunk.chunk_metadata),
        }
        if dataset_id:
            metadata["dataset_id"] = dataset_id
        return metadata

    def _deserialize_results(self, results: Mapping[str, object]) -> Sequence[RetrievedChunk]:
        ids = self._first(results.get("ids", []))
        documents = self._first(results.get("documents", []))
        metadatas = self._first(results.get("metadatas", []))
        distances = self._first(results.get("distances", []))
        retrieved: list[RetrievedChunk] = []
        if not ids or not documents or not metadatas:
            return retrieved
        for idx, doc, metadata, distance in zip(ids, documents, metadatas, distances or [], strict=False):
            retrieved.append(self._deserialize_chunk(idx, doc, metadata, distance))
        # Handle cases when distances missing or shorter than ids
        if len(retrieved) < len(ids):
            for idx, doc, metadata in zip(ids[len(retrieved) :], documents[len(retrieved) :], metadatas[len(retrieved) :], strict=False):
                retrieved.append(self._deserialize_chunk(idx, doc, metadata, distance=None))
        return retrieved

    def _deserialize_chunk(
        self,
        chunk_id: str,
        document: str,
        metadata: Mapping[str, object],
        distance: float | None,
    ) -> RetrievedChunk:
        document_metadata = DocumentMetadata(
            document_id=str(metadata.get("document_id", "")),
            source_path=str(metadata.get("source_path", "")),
            media_type=str(metadata.get("media_type", "")),
            extra=self._loads_dict(metadata.get("doc_extra")),
        )
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=document,
            document_metadata=document_metadata,
            order=int(metadata.get("order", 0)),
            chunk_metadata=self._loads_dict(metadata.get("chunk_metadata")),
        )
        score = 1.0 - float(distance) if distance is not None else 0.0
        return RetrievedChunk(chunk=chunk, score=score)

    @staticmethod
    def _first(value: object) -> Iterable:
        if isinstance(value, list):
            return value[0] if value else []
        return []

    @staticmethod
    def _dumps(value: object) -> str:
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return json.dumps({}, default=str)

    @staticmethod
    def _loads_dict(value: object) -> Dict[str, object]:
        if isinstance(value, str) and value:
            try:
                loaded = json.loads(value)
                if isinstance(loaded, dict):
                    return loaded
            except json.JSONDecodeError:
                return {}
        if isinstance(value, Mapping):
            return dict(value)
        return {}

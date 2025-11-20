from __future__ import annotations

import chromadb

from openrag.embeddings.service import EmbeddingConfig, HashEmbeddingBackend
from openrag.embeddings.store import ChromaEmbeddingStore
from openrag.models import DocumentChunk, DocumentMetadata


def _mk(doc_id: str, txt: str, order: int) -> DocumentChunk:
    meta = DocumentMetadata(document_id=doc_id, source_path=f"/tmp/{doc_id}.txt", media_type="txt")
    return DocumentChunk(chunk_id=f"{doc_id}-{order}", text=txt, document_metadata=meta, order=order)


def test_count_by_dataset():
    store = ChromaEmbeddingStore(HashEmbeddingBackend(EmbeddingConfig(dim=8)), collection_name="stats-test", client=chromadb.EphemeralClient())
    store.reset()
    store.upsert([_mk("a", "alpha", 0), _mk("a", "bravo", 1)], dataset_id="ds1")
    store.upsert([_mk("b", "charlie", 0)], dataset_id="ds2")
    counts = store.count_by_dataset()
    assert counts.get("ds1") == 2
    assert counts.get("ds2") == 1
    assert sum(counts.values()) == store.count()


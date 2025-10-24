from __future__ import annotations

import chromadb

from openrag.embeddings.service import EmbeddingConfig, HashEmbeddingBackend
from openrag.embeddings.store import ChromaEmbeddingStore
from openrag.models import DocumentChunk, DocumentMetadata


def _doc_chunk(doc_id: str, text: str, order: int) -> DocumentChunk:
    meta = DocumentMetadata(document_id=doc_id, source_path=f"/tmp/{doc_id}.txt", media_type="txt")
    return DocumentChunk(chunk_id=f"{doc_id}-{order}", text=text, document_metadata=meta, order=order)


def test_store_upsert_and_similarity_search():
    backend = HashEmbeddingBackend(EmbeddingConfig(dim=16))
    store = ChromaEmbeddingStore(backend, collection_name="test-store", client=chromadb.EphemeralClient())
    chunks = [_doc_chunk("d1", "alpha beta gamma", 0), _doc_chunk("d2", "lorem ipsum", 0)]
    store.reset()
    ids = store.upsert(chunks)
    assert len(ids) == 2
    assert store.count() >= 2
    results = store.similarity_search("alpha", top_k=2)
    assert results
    assert results[0].chunk.document_metadata.document_id in {"d1", "d2"}


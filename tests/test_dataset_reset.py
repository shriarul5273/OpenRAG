from __future__ import annotations

import chromadb

from openrag.embeddings.service import EmbeddingConfig, HashEmbeddingBackend
from openrag.embeddings.store import ChromaEmbeddingStore
from openrag.models import DocumentChunk, DocumentMetadata


def _mk(doc_id: str, txt: str, order: int) -> DocumentChunk:
    meta = DocumentMetadata(document_id=doc_id, source_path=f"/tmp/{doc_id}.txt", media_type="txt")
    return DocumentChunk(chunk_id=f"{doc_id}-{order}", text=txt, document_metadata=meta, order=order)


def test_reset_dataset_only_deletes_matching():
    store = ChromaEmbeddingStore(HashEmbeddingBackend(EmbeddingConfig(dim=8)), collection_name="reset-test", client=chromadb.EphemeralClient())
    store.reset()
    store.upsert([_mk("a", "alpha", 0)], dataset_id="ds1")
    store.upsert([_mk("b", "bravo", 0)], dataset_id="ds2")
    # ds1 returns a result for alpha
    r1 = store.similarity_search("alpha", top_k=5, dataset_id="ds1")
    r2 = store.similarity_search("alpha", top_k=5, dataset_id="ds2")
    assert r1 and not r2
    # reset ds1 only
    store.reset(dataset_id="ds1")
    r1_after = store.similarity_search("alpha", top_k=5, dataset_id="ds1")
    r2_after = store.similarity_search("alpha", top_k=5, dataset_id="ds2")
    assert not r1_after and r2_after


from __future__ import annotations

from openrag.embeddings.service import EmbeddingConfig, HashEmbeddingBackend
from openrag.models import DocumentChunk, DocumentMetadata


def _make_chunk(text: str) -> DocumentChunk:
    meta = DocumentMetadata(document_id="doc-1", source_path="/tmp/doc.txt", media_type="txt")
    return DocumentChunk(chunk_id="c1", text=text, document_metadata=meta, order=0)


def test_hash_embedding_dim_matches_config():
    backend = HashEmbeddingBackend(EmbeddingConfig(dim=64))
    vec = backend.embed_query("hello world")
    assert isinstance(vec, tuple)
    assert len(vec) == 64


def test_hash_embedding_chunks_returns_vectors():
    backend = HashEmbeddingBackend(EmbeddingConfig(dim=32))
    chunks = [_make_chunk("alpha"), _make_chunk("beta")]
    embeddings = backend.embed_chunks(chunks)
    assert len(embeddings) == 2
    assert len(embeddings[0].vector) == 32


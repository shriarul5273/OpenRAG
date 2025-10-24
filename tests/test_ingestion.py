"""Tests for ingestion-related helpers."""

from __future__ import annotations

from pathlib import Path

from openrag.ingestion.service import LangChainDocumentIngestor


def test_langchain_ingestor_records_display_name(tmp_path: Path) -> None:
    document = tmp_path / "example.txt"
    document.write_text("Hello world")

    ingestor = LangChainDocumentIngestor()
    chunks = ingestor.ingest([document])

    assert chunks, "Expected at least one chunk from ingestion"
    for chunk in chunks:
        assert chunk.document_metadata.extra["display_name"] == "example.txt"
        assert chunk.chunk_metadata["display_name"] == "example.txt"

"""Tests for the FastAPI application helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Sequence

from fastapi.testclient import TestClient

from openrag.api.app import AppDependencies, create_app
from openrag.config import Settings
from openrag.ingestion.service import DocumentIngestor
from openrag.models import Answer, DocumentChunk, DocumentMetadata, RetrievedChunk


class StubIngestor(DocumentIngestor):
    def ingest(self, paths: Sequence[Path]) -> Sequence[DocumentChunk]:
        path = Path(paths[0])
        metadata = DocumentMetadata(
            document_id="doc-1",
            source_path=str(path.resolve()),
            media_type="txt",
            extra={"source": str(path)},
        )
        chunk = DocumentChunk(
            chunk_id="doc-1-0",
            text="stub",
            document_metadata=metadata,
            order=0,
            chunk_metadata={"source": str(path)},
        )
        return [chunk]


class StubStore:
    def __init__(self) -> None:
        self.last_upserted: list[DocumentChunk] = []

    def upsert(self, chunks: Sequence[DocumentChunk]) -> None:
        self.last_upserted = list(chunks)

    def count(self) -> int:
        return len(self.last_upserted)

    def reset(self) -> None:
        self.last_upserted = []


class StubQueryService:
    def __init__(self, store: StubStore) -> None:
        self.store = store

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        chunk = self.store.last_upserted[0]
        return Answer(
            text="stub answer",
            citations=[RetrievedChunk(chunk=chunk, score=0.42)],
            query_id="query-1",
            latency_ms=1.0,
        )


def create_test_client(tmp_path: Path) -> TestClient:
    store = StubStore()
    deps = AppDependencies(
        ingestor=StubIngestor(),
        store=store,
        retriever=object(),
        query_service=StubQueryService(store),
    )
    settings = Settings(environment="test")
    app = create_app(settings=settings, dependencies=deps)
    return TestClient(app)


def test_upload_and_query_return_display_name(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    upload_response = client.post(
        "/documents",
        files=[("files", ("report.pdf", BytesIO(b"dummy"), "application/pdf"))],
    )
    assert upload_response.status_code == 201, upload_response.text
    payload = upload_response.json()
    assert payload["documents"][0]["source_path"] == "report.pdf"

    query_response = client.post("/query", json={"question": "What?"})
    assert query_response.status_code == 200, query_response.text
    query_payload = query_response.json()
    assert query_payload["citations"][0]["source_path"] == "report.pdf"

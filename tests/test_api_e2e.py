from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from openrag.api.app import create_app
from openrag.config import Settings


def make_app() -> TestClient:
    settings = Settings(
        environment="test",
        use_model_embeddings=False,
        use_model_generator=False,
        chroma_host=None,
        rate_limit_requests=1000,
        allowed_ingest_domains=(),
    )
    app = create_app(settings=settings)
    return TestClient(app)


def test_health_and_index_flow():
    client = make_app()
    # Health endpoints
    r = client.get("/healthz")
    assert r.status_code == 200
    r = client.head("/healthz")
    assert r.status_code == 200
    r = client.get("/livez")
    assert r.status_code == 200
    r = client.get("/healthz/ready")
    assert r.status_code == 200

    # Upload a small text file
    content = b"OpenRAG is a retrieval augmented generation reference implementation."
    files = {"files": ("readme.txt", BytesIO(content), "text/plain")}
    data = {"dataset_id": "test-ds"}
    r = client.post("/documents", files=files, data=data)
    assert r.status_code == 201, r.text
    dataset_id = r.json()["dataset_id"]
    assert dataset_id == "test-ds"

    # Stats should report at least 1 chunk
    r = client.get("/index/stats")
    assert r.status_code == 200
    stats = r.json()
    total = stats.get("total_chunks") or stats.get("chunks")
    assert total is not None and total >= 1

    # Query
    payload = {"question": "What is OpenRAG?", "top_k": 3, "dataset_id": "test-ds"}
    r = client.post("/query", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["citations"], "Expected at least one citation"


def test_text_ingestion_endpoint():
    client = make_app()
    payload = {"texts": ["alpha beta", "gamma"], "dataset_id": "text-ds"}
    r = client.post("/documents/text", json=payload)
    assert r.status_code == 201, r.text
    assert r.json()["dataset_id"] == "text-ds"

    # Should retrieve from text dataset
    query_payload = {"question": "alpha?", "top_k": 3, "dataset_id": "text-ds"}
    r = client.post("/query", json=query_payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["citations"], "Expected citations from text ingestion"

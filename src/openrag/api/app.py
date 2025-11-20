"""FastAPI application exposing OpenRAG services."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import chromadb

from openrag.api.schemas import (
    CitationModel,
    DocumentIngestionResponse,
    DocumentSummary,
    DatasetStats,
    IndexStatsResponse,
    QueryRequest,
    QueryResponse,
    TextIngestionRequest,
    UrlIngestionRequest,
)
from openrag.config import Settings, get_settings
from openrag.embeddings import ChromaEmbeddingStore, EmbeddingConfig, QwenEmbeddingBackend
from openrag.ingestion import (
    DocumentIngestor,
    IngestionConfig,
    IngestionError,
    LangChainDocumentIngestor,
    UnsupportedFileTypeError,
)
from openrag.models import DocumentChunk
from openrag.retrieval.service import ChromaRetriever, RetrievalConfig, Retriever
from openrag.metrics.observability import bind_correlation_id, clear_correlation_id, configure_logging, get_logger
from openrag.services.generation import GenerationConfig, QwenGenerator, TemplateGenerator
from openrag.services.query import PromptBuilder, QueryService

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass(frozen=True)
class AppDependencies:
    ingestor: DocumentIngestor
    store: ChromaEmbeddingStore
    retriever: Retriever
    query_service: QueryService


def _build_dependencies(settings: Settings) -> AppDependencies:
    ingestion_config = IngestionConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        encoding="utf-8",
        use_token_splitter=settings.use_token_splitter,
        tokens_per_chunk=settings.tokens_per_chunk,
        token_overlap=settings.token_overlap,
    )
    ingestor = LangChainDocumentIngestor(config=ingestion_config)

    embedding_backend = QwenEmbeddingBackend(
        EmbeddingConfig(
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            use_model=settings.use_model_embeddings,
            normalize=True,
        ),
    )
    chroma_client = None
    if settings.chroma_host:
        chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port or 8000,
            ssl=settings.chroma_ssl,
        )
    store = ChromaEmbeddingStore(
        embedding_backend,
        collection_name=settings.chroma_collection,
        client=chroma_client,
        persist_directory=None if chroma_client else settings.chroma_persist_dir,
    )
    retriever = ChromaRetriever(
        store,
        RetrievalConfig(
            top_k=settings.max_chunks,
            rerank_lexical=settings.retrieval_rerank_lexical,
            lexical_blend_weight=settings.retrieval_lexical_blend_weight,
            max_top_k=settings.retrieval_max_top_k,
            use_cross_encoder=settings.cross_encoder_use,
            cross_encoder_model=settings.cross_encoder_model,
            cross_encoder_top_n=settings.cross_encoder_top_n,
            cross_encoder_device=settings.cross_encoder_device,
        ),
    )
    generator = QwenGenerator(
        GenerationConfig(
            model=settings.generator_model,
            max_new_tokens=settings.generator_max_new_tokens,
            temperature=settings.generator_temperature,
            use_model=settings.use_model_generator,
        ),
        fallback=TemplateGenerator(),
    )
    query_service = QueryService(retriever=retriever, generator=generator, prompt_builder=PromptBuilder())
    return AppDependencies(ingestor=ingestor, store=store, retriever=retriever, query_service=query_service)


def create_app(*, settings: Settings | None = None, dependencies: AppDependencies | None = None) -> FastAPI:
    settings = settings or get_settings()
    deps = dependencies or _build_dependencies(settings)

    configure_logging()
    logger = get_logger("api")
    app = FastAPI(title="OpenRAG API", version="0.1.1")
    app.state.dependencies = deps

    # Optional CORS
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=list(settings.cors_allow_methods),
            allow_headers=list(settings.cors_allow_headers),
        )

    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next):  # type: ignore[override]
        correlation_id = request.headers.get("X-Request-ID", uuid4().hex)
        request.state.correlation_id = correlation_id
        bind_correlation_id(correlation_id)
        try:
            response = await call_next(request)
        finally:
            clear_correlation_id()
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    # Security dependencies
    def require_api_key(request: Request) -> None:
        expected = settings.api_key
        if not expected:
            return
        provided = request.headers.get("X-API-Key")
        if provided != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    class RateLimiter:
        def __init__(self, requests: int, window_seconds: int) -> None:
            self.requests = requests
            self.window = window_seconds
            self._buckets: dict[str, list[float]] = {}

        def __call__(self, request: Request) -> None:
            import time as _t

            client_ip = request.headers.get("x-forwarded-for") or request.client.host
            key = f"{client_ip}:{request.url.path}"
            now = _t.time()
            bucket = self._buckets.setdefault(key, [])
            # Drop old entries
            cutoff = now - self.window
            while bucket and bucket[0] < cutoff:
                bucket.pop(0)
            if len(bucket) >= self.requests:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
            bucket.append(now)

    rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)

    @app.exception_handler(IngestionError)
    async def handle_ingestion_error(request: Request, exc: IngestionError) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", uuid4().hex)
        logger.error("ingestion.error", correlation_id=correlation_id, detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "correlation_id": correlation_id},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", uuid4().hex)
        logger.error("unhandled.error", correlation_id=correlation_id, detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error", "correlation_id": correlation_id},
        )

    def get_dependencies(request: Request) -> AppDependencies:
        return request.app.state.dependencies

    def get_ingestor(dep: AppDependencies = Depends(get_dependencies)) -> DocumentIngestor:
        return dep.ingestor

    def get_store(dep: AppDependencies = Depends(get_dependencies)) -> ChromaEmbeddingStore:
        return dep.store

    def get_query_service(dep: AppDependencies = Depends(get_dependencies)) -> QueryService:
        return dep.query_service

    @app.post("/documents", response_model=DocumentIngestionResponse, status_code=status.HTTP_201_CREATED)
    async def upload_documents(
        request: Request,
        files: Sequence[UploadFile] = File(...),
        dataset_id: str | None = Form(default=None),
        ingestor: DocumentIngestor = Depends(get_ingestor),
        store: ChromaEmbeddingStore = Depends(get_store),
        _auth: None = Depends(require_api_key),
        _rl: None = Depends(rate_limiter),
    ) -> DocumentIngestionResponse:
        if not files:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")
        if len(files) > settings.max_files:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Too many files")

        # Check total payload if provided
        total_len = request.headers.get("content-length")
        if total_len and int(total_len) > settings.max_total_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Payload too large")
        effective_dataset_id = (dataset_id or "").strip() or settings.default_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths: list[Path] = []
            for upload in files:
                filename = upload.filename or f"upload-{uuid4().hex}"
                suffix = Path(filename).suffix.lower()
                allowed = set(settings.allowed_extensions_tuple) or ALLOWED_EXTENSIONS
                if suffix not in allowed:
                    await upload.close()
                    msg = f"Unsupported file type: {suffix or 'unknown'}"
                    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=msg)
                destination = Path(tmpdir) / filename
                # Stream copy to avoid loading entire file into memory
                bytes_written = 0
                with destination.open("wb") as out_f:
                    while True:
                        chunk = await upload.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        out_f.write(chunk)
                        bytes_written += len(chunk)
                        if bytes_written > settings.max_upload_size_mb * 1024 * 1024:
                            await upload.close()
                            # Remove partial file
                            try:
                                destination.unlink(missing_ok=True)
                            except Exception:
                                pass
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"File too large (>{settings.max_upload_size_mb}MB): {filename}",
                            )
                await upload.close()
                # Optional magic MIME validation
                try:
                    import magic  # type: ignore

                    mime = magic.Magic(mime=True)
                    detected = str(mime.from_file(str(destination)))
                    allowed_mimes = settings.allowed_mime_map
                    if suffix in allowed_mimes and detected not in allowed_mimes[suffix]:
                        # Clean up and reject
                        try:
                            destination.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise HTTPException(
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=f"MIME mismatch for {filename}: {detected}",
                        )
                except ImportError:
                    # magic is optional; skip if unavailable
                    pass
                if bytes_written == 0:
                    # Empty file; reject to avoid useless ingest
                    try:
                        destination.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File is empty: {filename}",
                    )
                saved_paths.append(destination)
            try:
                chunks = ingestor.ingest(saved_paths)
            except UnsupportedFileTypeError as exc:
                raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
        dataset_documents = _build_document_summaries(chunks)
        store.upsert(chunks, dataset_id=effective_dataset_id)
        return DocumentIngestionResponse(dataset_id=effective_dataset_id, documents=dataset_documents)

    @app.post("/documents/text", response_model=DocumentIngestionResponse, status_code=status.HTTP_201_CREATED)
    async def ingest_raw_text(
        payload: TextIngestionRequest,
        ingestor: DocumentIngestor = Depends(get_ingestor),
        store: ChromaEmbeddingStore = Depends(get_store),
        _auth: None = Depends(require_api_key),
        _rl: None = Depends(rate_limiter),
    ) -> DocumentIngestionResponse:
        if not payload.texts:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No text provided")
        effective_dataset_id = (payload.dataset_id or "").strip() or settings.default_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[Path] = []
            for index, text in enumerate(payload.texts):
                normalized = (text or "").strip()
                if not normalized:
                    continue
                path = Path(tmpdir) / f"text-{index}.txt"
                path.write_text(normalized, encoding="utf-8")
                paths.append(path)
            if not paths:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No non-empty text provided")
            chunks = ingestor.ingest(paths)
        dataset_documents = _build_document_summaries(chunks)
        store.upsert(chunks, dataset_id=effective_dataset_id)
        return DocumentIngestionResponse(dataset_id=effective_dataset_id, documents=dataset_documents)

    @app.post("/query", response_model=QueryResponse)
    async def query_documents(
        payload: QueryRequest,
        service: QueryService = Depends(get_query_service),
        _auth: None = Depends(require_api_key),
        _rl: None = Depends(rate_limiter),
    ) -> QueryResponse:
        answer = service.answer(payload.question, top_k=payload.top_k, dataset_id=payload.dataset_id or settings.default_dataset)
        if not answer.citations:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No relevant documents found")
        citations = [
            CitationModel(
                chunk_id=citation.chunk.chunk_id,
                document_id=citation.chunk.document_metadata.document_id,
                source_path=citation.chunk.document_metadata.source_path,
                media_type=citation.chunk.document_metadata.media_type,
                score=citation.score,
                text=citation.chunk.text,
            )
            for citation in answer.citations
        ]
        return QueryResponse(
            query_id=answer.query_id,
            answer=answer.text,
            citations=citations,
            latency_ms=answer.latency_ms,
        )

    @app.post("/documents/url", response_model=DocumentIngestionResponse, status_code=status.HTTP_201_CREATED)
    async def ingest_from_urls(
        payload: UrlIngestionRequest,
        request: Request,
        ingestor: DocumentIngestor = Depends(get_ingestor),
        store: ChromaEmbeddingStore = Depends(get_store),
        _auth: None = Depends(require_api_key),
        _rl: None = Depends(rate_limiter),
    ) -> DocumentIngestionResponse:
        if not payload.urls:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No URLs provided")
        allowed = settings.allowed_ingest_domains
        allowed_t = allowed if isinstance(allowed, tuple) else tuple([d.strip() for d in allowed.split(',') if d.strip()])
        import httpx
        dataset_id = payload.dataset_id or settings.default_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[Path] = []
            for url in payload.urls:
                try:
                    host = httpx.URL(url).host or ""
                except Exception:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid URL: {url}")
                if not allowed_t or host not in allowed_t:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Domain not allowed: {host}")
                dest = Path(tmpdir) / (Path(url).name or f"download-{uuid4().hex}.bin")
                size_limit = settings.max_download_size_mb * 1024 * 1024
                try:
                    with httpx.stream("GET", url, timeout=30.0, follow_redirects=True) as resp:
                        if resp.status_code >= 400:
                            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Download failed: {url}")
                        bytes_written = 0
                        with dest.open("wb") as f:
                            for chunk in resp.iter_bytes():
                                if chunk:
                                    f.write(chunk)
                                    bytes_written += len(chunk)
                                    if bytes_written > size_limit:
                                        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Download too large: {url}")
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed downloading {url}: {exc}")
                paths.append(dest)
            chunks = ingestor.ingest(paths)
            dataset_documents = _build_document_summaries(chunks)
            store.upsert(chunks, dataset_id=dataset_id)
            return DocumentIngestionResponse(dataset_id=dataset_id, documents=dataset_documents)

    @app.get("/metrics")
    async def metrics() -> Response:
        payload = generate_latest()
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        from openrag import __version__

        return {"status": "ok", "version": __version__, "environment": settings.environment}

    @app.head("/healthz")
    async def healthcheck_head() -> Response:
        return Response(status_code=status.HTTP_200_OK)

    @app.get("/livez")
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/healthz/ready")
    async def readiness(store: ChromaEmbeddingStore = Depends(get_store)) -> dict[str, str]:
        try:
            _ = store.count()
            return {"status": "ready"}
        except Exception as exc:  # pragma: no cover - defensive
            return {"status": "error", "detail": str(exc)}

    @app.delete("/index", status_code=status.HTTP_204_NO_CONTENT)
    async def reset_index(
        store: ChromaEmbeddingStore = Depends(get_store),
        dataset_id: str | None = None,
        _auth: None = Depends(require_api_key),
    ) -> Response:
        store.reset(dataset_id=dataset_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/index/stats", response_model=IndexStatsResponse)
    async def index_stats(store: ChromaEmbeddingStore = Depends(get_store)) -> IndexStatsResponse:
        total = store.count()
        by_ds = store.count_by_dataset()
        datasets = [DatasetStats(dataset_id=k, chunks=v) for k, v in sorted(by_ds.items())]
        # Update Prometheus gauge per dataset
        from openrag.metrics.observability import PipelineMetrics

        for k, v in by_ds.items():
            PipelineMetrics.dataset_chunk_count.labels(dataset_id=k).set(v)
        return IndexStatsResponse(
            collection=settings.chroma_collection,
            total_chunks=total,
            chunks=total,
            datasets=datasets,
        )

    @app.post("/query/stream")
    async def query_stream(
        payload: QueryRequest,
        service: QueryService = Depends(get_query_service),
        _auth: None = Depends(require_api_key),
        _rl: None = Depends(rate_limiter),
    ) -> Response:
        # Generate full answer then stream in small chunks as SSE with heartbeats
        answer = service.answer(payload.question, top_k=payload.top_k, dataset_id=payload.dataset_id or settings.default_dataset)
        import json as _json
        text = answer.text
        citations = [
            {
                "chunk_id": c.chunk.chunk_id,
                "document_id": c.chunk.document_metadata.document_id,
                "source_path": c.chunk.document_metadata.source_path,
                "media_type": c.chunk.document_metadata.media_type,
                "score": c.score,
                "text": c.chunk.text,
            }
            for c in answer.citations
        ]

        def iter_sse():
            # Initial heartbeat to keep idle proxies open
            yield ": heartbeat\n\n"
            chunk_size = 128
            for i in range(0, len(text), chunk_size):
                yield f"data: {text[i:i+chunk_size]}\n\n"
            yield f"event: citations\ndata: {_json.dumps(citations)}\n\n"
            yield ": heartbeat\n\n"

        return Response(iter_sse(), media_type="text/event-stream")

    return app


def _build_document_summaries(chunks: Iterable[DocumentChunk]) -> list[DocumentSummary]:
    counts: defaultdict[str, int] = defaultdict(int)
    metadata: dict[str, DocumentChunk] = {}
    for chunk in chunks:
        doc_id = chunk.document_metadata.document_id
        counts[doc_id] += 1
        metadata.setdefault(doc_id, chunk)
    summaries: list[DocumentSummary] = []
    for doc_id, count in counts.items():
        chunk = metadata[doc_id]
        summaries.append(
            DocumentSummary(
                document_id=doc_id,
                source_path=chunk.document_metadata.source_path,
                media_type=chunk.document_metadata.media_type,
                chunk_count=count,
            ),
        )
    return summaries


app = create_app()

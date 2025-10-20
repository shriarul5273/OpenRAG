"""FastAPI application exposing OpenRAG services."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import chromadb

from openrag.api.schemas import (
    CitationModel,
    DocumentIngestionResponse,
    DocumentSummary,
    QueryRequest,
    QueryResponse,
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
from openrag.metrics.observability import bind_correlation_id, clear_correlation_id, configure_logging
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
    retriever = ChromaRetriever(store, RetrievalConfig(top_k=settings.max_chunks))
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
    app = FastAPI(title="OpenRAG API", version="0.1.0")
    app.state.dependencies = deps

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

    @app.exception_handler(IngestionError)
    async def handle_ingestion_error(request: Request, exc: IngestionError) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", uuid4().hex)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "correlation_id": correlation_id},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", uuid4().hex)
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
        files: Sequence[UploadFile] = File(...),
        ingestor: DocumentIngestor = Depends(get_ingestor),
        store: ChromaEmbeddingStore = Depends(get_store),
    ) -> DocumentIngestionResponse:
        if not files:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")
        dataset_id = uuid4().hex
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths: list[Path] = []
            for upload in files:
                filename = upload.filename or f"upload-{uuid4().hex}"
                suffix = Path(filename).suffix.lower()
                if suffix not in ALLOWED_EXTENSIONS:
                    await upload.close()
                    msg = f"Unsupported file type: {suffix or 'unknown'}"
                    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=msg)
                destination = Path(tmpdir) / filename
                async_data = await upload.read()
                destination.write_bytes(async_data)
                await upload.close()
                saved_paths.append(destination)
            try:
                chunks = ingestor.ingest(saved_paths)
            except UnsupportedFileTypeError as exc:
                raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
        dataset_documents = _build_document_summaries(chunks)
        store.upsert(chunks)
        return DocumentIngestionResponse(dataset_id=dataset_id, documents=dataset_documents)

    @app.post("/query", response_model=QueryResponse)
    async def query_documents(
        payload: QueryRequest,
        service: QueryService = Depends(get_query_service),
    ) -> QueryResponse:
        answer = service.answer(payload.question, top_k=payload.top_k)
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

    @app.get("/metrics")
    async def metrics() -> Response:
        payload = generate_latest()
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

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

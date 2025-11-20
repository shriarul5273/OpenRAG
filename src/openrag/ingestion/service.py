"""Document ingestion service for OpenRAG."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_community.document_loaders import BSHTMLLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from openrag.metrics.observability import PipelineMetrics, get_logger
from openrag.models import DocumentChunk, DocumentMetadata


class IngestionError(RuntimeError):
    """Raised when ingestion fails for a particular document."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a document extension is not supported by the ingestor."""


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for document ingestion."""

    chunk_size: int = 600
    chunk_overlap: int = 100
    encoding: str = "utf-8"
    use_token_splitter: bool = False
    tokens_per_chunk: int = 256
    token_overlap: int = 50


class DocumentIngestor(Protocol):
    """Protocol for ingestion implementations."""

    def ingest(self, paths: Sequence[Path]) -> Sequence[DocumentChunk]:
        """Ingest the given document paths into normalized chunks."""


def _normalize_text(raw: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


class LangChainDocumentIngestor:
    """Ingest documents via LangChain loaders and chunk splitter."""

    _LOADERS: Mapping[str, type[BaseLoader]] = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".html": BSHTMLLoader,
        ".htm": BSHTMLLoader,
    }

    _logger = get_logger("ingestion")

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self._config = config or IngestionConfig()
        if self._config.use_token_splitter:
            self._splitter = TokenTextSplitter(
                chunk_size=self._config.tokens_per_chunk,
                chunk_overlap=self._config.token_overlap,
            )
        else:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
                add_start_index=True,
            )

    def ingest(self, paths: Sequence[Path]) -> Sequence[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for path in paths:
            chunks.extend(self._ingest_single(path))
        return chunks

    def _ingest_single(self, path: Path) -> Sequence[DocumentChunk]:
        suffix = path.suffix.lower()
        loader_cls = self._LOADERS.get(suffix)
        if loader_cls is None:
            raise UnsupportedFileTypeError(f"Unsupported document type: {suffix or '<none>'}")

        start = time.perf_counter()
        try:
            loader = self._build_loader(loader_cls, path)
            documents = loader.load()
        except Exception as exc:  # pragma: no cover - loader specific errors
            raise IngestionError(f"Failed to load {path}: {exc}") from exc

        normalized_docs = self._normalize_documents(documents, path)
        split_docs = self._splitter.split_documents(normalized_docs)
        doc_id = uuid5(NAMESPACE_URL, str(path.resolve())).hex
        document_metadata = DocumentMetadata(
            document_id=doc_id,
            source_path=str(path.resolve()),
            media_type=suffix.lstrip("."),
            extra={"source": str(path)},
        )

        chunks: List[DocumentChunk] = []
        for order, doc in enumerate(split_docs):
            chunk_metadata: Dict[str, object] = dict(doc.metadata)
            chunk_metadata.setdefault("source", str(path))
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}-{order}",
                text=_normalize_text(doc.page_content),
                document_metadata=document_metadata,
                order=order,
                chunk_metadata=chunk_metadata,
            )
            chunks.append(chunk)

        duration = time.perf_counter() - start
        PipelineMetrics.observe_ingestion(duration, len(chunks))
        self._logger.info(
            "ingestion.complete",
            path=str(path),
            chunk_count=len(chunks),
            duration_seconds=duration,
        )
        return chunks

    def _build_loader(self, loader_cls: type[BaseLoader], path: Path) -> BaseLoader:
        if loader_cls is TextLoader:
            return loader_cls(str(path), encoding=self._config.encoding)
        return loader_cls(str(path))

    def _normalize_documents(self, documents: Sequence[LCDocument], path: Path) -> Sequence[LCDocument]:
        normalized: List[LCDocument] = []
        for document in documents:
            metadata: Dict[str, object] = dict(document.metadata)
            metadata.setdefault("source", str(path))
            normalized.append(
                LCDocument(
                    page_content=_normalize_text(document.page_content),
                    metadata=metadata,
                ),
            )
        return normalized


def ingest_paths(*paths: Path, config: IngestionConfig | None = None) -> Sequence[DocumentChunk]:
    """Convenience helper for tests and ad-hoc ingestion."""

    ingestor = LangChainDocumentIngestor(config=config)
    return ingestor.ingest(list(paths))

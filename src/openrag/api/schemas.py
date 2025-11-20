"""Pydantic models for the OpenRAG API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentSummary(BaseModel):
    document_id: str = Field(..., description="Stable identifier for the ingested document")
    source_path: str = Field(..., description="Absolute path of the stored document")
    media_type: str = Field(..., description="Original media type (pdf, docx, txt)")
    chunk_count: int = Field(..., ge=0, description="Number of chunks created for the document")


class DocumentIngestionResponse(BaseModel):
    dataset_id: str = Field(..., description="Identifier representing this ingestion batch")
    documents: List[DocumentSummary]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="End-user question to answer")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="Override the number of retrieved chunks")
    dataset_id: Optional[str] = Field(default=None, description="Optional dataset/namespace to search within")


class CitationModel(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    media_type: str
    score: float
    text: str


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: List[CitationModel]
    latency_ms: float


class UrlIngestionRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to ingest")
    dataset_id: Optional[str] = Field(default=None)

class IndexStats(BaseModel):
    collection: str
    chunks: int

class DatasetStats(BaseModel):
    dataset_id: str
    chunks: int

class IndexStatsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    collection: str
    total_chunks: int
    chunks: int | None = Field(
        default=None,
        description="Legacy duplicate of total_chunks for clients expecting `chunks`",
    )
    datasets: List[DatasetStats]


class TextIngestionRequest(BaseModel):
    """Payload for ingesting raw text content."""

    texts: List[str] = Field(..., description="List of raw text snippets to ingest")
    dataset_id: Optional[str] = Field(default=None)

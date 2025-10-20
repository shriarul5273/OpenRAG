"""Pydantic models for the OpenRAG API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


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

"""Query orchestration combining retrieval and generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from openrag.metrics.observability import PipelineMetrics, get_logger
from openrag.models import Answer, RetrievedChunk
from openrag.retrieval.service import Retriever
from openrag.services.generation import GenerationBackend, TemplateGenerator


@dataclass(frozen=True)
class PromptBuilderConfig:
    """Configuration for prompt construction."""

    citation_prefix: str = "["
    citation_suffix: str = "]"


class PromptBuilder:
    """Builds prompts for the generation backend."""

    def __init__(self, config: PromptBuilderConfig | None = None) -> None:
        self._config = config or PromptBuilderConfig()

    def build_context(self, citations: Sequence[RetrievedChunk]) -> str:
        if not citations:
            return ""
        lines = []
        for index, citation in enumerate(citations, start=1):
            prefix = f"{self._config.citation_prefix}{index}{self._config.citation_suffix}"
            source = citation.chunk.document_metadata.source_path
            lines.append(f"{prefix} {citation.chunk.text}\nSource: {source}")
        return "\n\n".join(lines)


class QueryService:
    """Orchestrates retrieval and generation for incoming questions."""

    def __init__(
        self,
        retriever: Retriever,
        generator: GenerationBackend | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator or TemplateGenerator()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._logger = get_logger("query")

    def answer(self, question: str, *, top_k: int | None = None, dataset_id: str | None = None) -> Answer:
        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        retrieved = self._retriever.retrieve(question, top_k=top_k, dataset_id=dataset_id)
        retrieval_duration = time.perf_counter() - retrieval_start
        PipelineMetrics.observe_retrieval(
            retrieval_duration,
            len(retrieved),
            (citation.score for citation in retrieved),
        )
        self._logger.info(
            "retrieval.complete",
            question=question,
            chunk_count=len(retrieved),
            duration_seconds=retrieval_duration,
            top_k=top_k,
        )
        deduped = self._dedupe_citations(retrieved)
        context = self._prompt_builder.build_context(deduped)
        generation_start = time.perf_counter()
        text = self._generator.generate(question=question, context=context, citations=deduped)
        generation_duration = time.perf_counter() - generation_start
        PipelineMetrics.observe_generation(generation_duration)
        self._logger.info(
            "generation.complete",
            question=question,
            duration_seconds=generation_duration,
            citation_count=len(deduped),
        )
        latency_ms = (time.perf_counter() - start) * 1000
        query_id = uuid5(NAMESPACE_URL, question).hex
        return Answer(text=text, citations=deduped, query_id=query_id, latency_ms=latency_ms)

    @staticmethod
    def _dedupe_citations(citations: Sequence[RetrievedChunk]) -> Sequence[RetrievedChunk]:
        seen: set[str] = set()
        ordered: list[RetrievedChunk] = []
        for citation in citations:
            doc_id = citation.chunk.document_metadata.document_id
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ordered.append(citation)
        return ordered

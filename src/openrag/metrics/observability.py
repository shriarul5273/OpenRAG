"""Observability helpers for OpenRAG."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Iterable, Sequence

import structlog
from prometheus_client import Gauge, Histogram

_logger_configured = False
_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


def configure_logging(level: int = logging.INFO) -> None:
    global _logger_configured  # noqa: PLW0603 - module-level guard
    if _logger_configured:
        return
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _logger_configured = True


def bind_correlation_id(correlation_id: str) -> None:
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    _correlation_id_var.set(correlation_id)


def clear_correlation_id() -> None:
    structlog.contextvars.clear_contextvars()
    _correlation_id_var.set("-")


def get_correlation_id() -> str:
    return _correlation_id_var.get()


def get_logger(name: str = "openrag") -> structlog.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)


def _clamp_score(score: float) -> float:
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


class PipelineMetrics:
    """Prometheus metrics for pipeline stages."""

    ingestion_latency = Histogram(
        "openrag_ingestion_duration_seconds",
        "Time spent ingesting documents.",
        buckets=(0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    ingestion_chunks = Histogram(
        "openrag_ingestion_chunk_count",
        "Chunks produced per ingestion batch.",
        buckets=(0, 1, 5, 10, 20, 40, 80),
    )
    retrieval_latency = Histogram(
        "openrag_retrieval_duration_seconds",
        "Time spent retrieving context chunks.",
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
    )
    retrieved_chunk_count = Histogram(
        "openrag_retrieved_chunk_count",
        "Number of chunks returned by retrieval.",
        buckets=(0, 1, 2, 3, 5, 8, 13),
    )
    grounding_score = Histogram(
        "openrag_grounding_score",
        "Similarity/hallucination proxy score for retrieved chunks.",
        buckets=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    generation_latency = Histogram(
        "openrag_generation_duration_seconds",
        "Time spent generating answers.",
        buckets=(0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
    )
    dataset_chunk_count = Gauge(
        "openrag_dataset_chunk_count",
        "Number of chunks per dataset.",
        ["dataset_id"],
    )

    @classmethod
    def observe_ingestion(cls, duration_seconds: float, chunk_count: int) -> None:
        cls.ingestion_latency.observe(duration_seconds)
        cls.ingestion_chunks.observe(chunk_count)

    @classmethod
    def observe_retrieval(
        cls,
        duration_seconds: float,
        chunk_count: int,
        scores: Iterable[float],
    ) -> None:
        cls.retrieval_latency.observe(duration_seconds)
        cls.retrieved_chunk_count.observe(chunk_count)
        for score in scores:
            cls.grounding_score.observe(_clamp_score(score))

    @classmethod
    def observe_generation(cls, duration_seconds: float) -> None:
        cls.generation_latency.observe(duration_seconds)


class TimedSection:
    """Context manager capturing elapsed time for metrics."""

    def __init__(self, callback) -> None:
        self._callback = callback
        self._start = 0.0

    def __enter__(self) -> "TimedSection":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        duration = time.perf_counter() - self._start
        self._callback(duration)


__all__ = [
    "PipelineMetrics",
    "TimedSection",
    "bind_correlation_id",
    "clear_correlation_id",
    "configure_logging",
    "get_correlation_id",
    "get_logger",
]

"""CLI for evaluating OpenRAG retrieval accuracy."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import chromadb

from openrag.config import Settings, get_settings
from openrag.embeddings import ChromaEmbeddingStore, EmbeddingConfig, HashEmbeddingBackend
from openrag.ingestion import IngestionConfig, LangChainDocumentIngestor
from openrag.retrieval.service import ChromaRetriever, RetrievalConfig
from openrag.services.generation import TemplateGenerator
from openrag.services.query import PromptBuilder, QueryService


@dataclass(frozen=True)
class DocumentFixture:
    id: str
    title: str
    content: str


@dataclass(frozen=True)
class QueryFixture:
    question: str
    relevant_document_ids: Sequence[str]
    expected_answer: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    total_queries: int
    hits: int
    recall_at_k: float
    mean_reciprocal_rank: float
    average_latency_ms: float
    details: List[dict]

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "hits": self.hits,
            "recall_at_k": self.recall_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "average_latency_ms": self.average_latency_ms,
            "details": self.details,
        }


def load_dataset(path: Path) -> tuple[list[DocumentFixture], list[QueryFixture]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = [
        DocumentFixture(id=item["id"], title=item.get("title", ""), content=item["content"])
        for item in data["documents"]
    ]
    queries = [
        QueryFixture(
            question=item["question"],
            relevant_document_ids=item.get("relevant_document_ids", []),
            expected_answer=item.get("expected_answer"),
        )
        for item in data["queries"]
    ]
    return documents, queries


def _write_documents(temp_dir: Path, fixtures: Sequence[DocumentFixture]) -> list[Path]:
    paths: list[Path] = []
    for fixture in fixtures:
        path = temp_dir / f"{fixture.id}.txt"
        path.write_text(fixture.content, encoding="utf-8")
        paths.append(path)
    return paths


def _map_source_to_fixture(paths: Sequence[Path], fixtures: Sequence[DocumentFixture]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for path, fixture in zip(paths, fixtures, strict=True):
        lookup[str(path.resolve())] = fixture.id
    return lookup


def run_evaluation(
    dataset_path: Path,
    *,
    top_k: int = 3,
    settings: Settings | None = None,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
    dataset_id: str | None = None,
) -> EvaluationResult:
    settings = settings or get_settings()
    documents, queries = load_dataset(dataset_path)
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        doc_paths = _write_documents(tmpdir, documents)
        source_lookup = _map_source_to_fixture(doc_paths, documents)

        backend = HashEmbeddingBackend(EmbeddingConfig(dim=settings.embedding_dim))
        store = ChromaEmbeddingStore(
            backend,
            collection_name="evaluation",
            client=chromadb.EphemeralClient(),
        )
        ingestor = LangChainDocumentIngestor(
            IngestionConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ),
        )

        chunks = ingestor.ingest(doc_paths)
        store.reset()
        store.upsert(chunks, dataset_id=dataset_id or settings.default_dataset)

        service = QueryService(
            ChromaRetriever(store, RetrievalConfig(top_k=top_k)),
            generator=TemplateGenerator(),
            prompt_builder=PromptBuilder(),
        )

        hits = 0
        reciprocal_ranks: list[float] = []
        latencies: list[float] = []
        details: list[dict] = []

        for query in queries:
            answer = service.answer(query.question, top_k=top_k, dataset_id=dataset_id or settings.default_dataset)
            latencies.append(answer.latency_ms)
            retrieved_ids = [
                source_lookup.get(chunk.chunk.document_metadata.source_path)
                for chunk in answer.citations
            ]
            retrieved_ids = [doc_id for doc_id in retrieved_ids if doc_id]
            relevant_set = set(query.relevant_document_ids)
            rank = None
            for index, doc_id in enumerate(retrieved_ids, start=1):
                if doc_id in relevant_set:
                    rank = index
                    break
            if rank is not None:
                hits += 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0.0)
            details.append(
                {
                    "question": query.question,
                    "retrieved": retrieved_ids,
                    "relevant": query.relevant_document_ids,
                    "latency_ms": answer.latency_ms,
                    "answer": answer.text,
                },
            )

    total = len(queries)
    recall = hits / total if total else 0.0
    mrr = statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_latency = statistics.fmean(latencies) if latencies else 0.0
    result = EvaluationResult(
        total_queries=total,
        hits=hits,
        recall_at_k=recall,
        mean_reciprocal_rank=mrr,
        average_latency_ms=avg_latency,
        details=details,
    )

    if json_out:
        json_out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if markdown_out:
        markdown_out.write_text(_format_markdown(result), encoding="utf-8")
    return result


def _format_markdown(result: EvaluationResult) -> str:
    lines = [
        "# OpenRAG Evaluation Report",
        "",
        f"- Total queries: {result.total_queries}",
        f"- Hits: {result.hits}",
        f"- Recall@k: {result.recall_at_k:.2f}",
        f"- MRR: {result.mean_reciprocal_rank:.2f}",
        f"- Avg latency (ms): {result.average_latency_ms:.2f}",
        "",
        "| Question | Retrieved | Relevant |",
        "| --- | --- | --- |",
    ]
    for item in result.details:
        retrieved = ", ".join(item["retrieved"]) if item["retrieved"] else "-"
        relevant = ", ".join(item["relevant"]) if item["relevant"] else "-"
        lines.append(f"| {item['question']} | {retrieved} | {relevant} |")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenRAG retrieval accuracy.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluations/fixtures/sample.json"),
        help="Path to evaluation dataset JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Retriever top-k value to evaluate")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON report")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Optional path to write Markdown report")
    parser.add_argument("--min-recall", type=float, default=None, help="Override recall threshold")
    parser.add_argument("--min-mrr", type=float, default=None, help="Override MRR threshold")
    parser.add_argument("--dataset-id", type=str, default=None, help="Optional dataset namespace to use")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    settings = get_settings()
    min_recall = args.min_recall if args.min_recall is not None else settings.evaluation_min_recall
    min_mrr = args.min_mrr if args.min_mrr is not None else settings.evaluation_min_mrr

    result = run_evaluation(
        args.dataset,
        top_k=args.top_k,
        settings=settings,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
        dataset_id=args.dataset_id,
    )
    print(json.dumps(result.to_dict(), indent=2))

    if result.recall_at_k < min_recall or result.mean_reciprocal_rank < min_mrr:
        print(
            f"Evaluation failed thresholds (recall {result.recall_at_k:.2f} vs {min_recall}, "
            f"MRR {result.mean_reciprocal_rank:.2f} vs {min_mrr})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())

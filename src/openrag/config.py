"""Runtime configuration for the OpenRAG services."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed configuration model."""

    model_config = SettingsConfigDict(env_prefix="openrag_", env_file=".env", case_sensitive=False)

    environment: Literal["dev", "test", "prod"] = "dev"
    data_dir: Path = Path("./data")

    # Dataset / namespace
    default_dataset: str = "default"

    chroma_persist_dir: Path = Path("./.chroma")
    chroma_collection: str = "openrag-default"
    chroma_host: str | None = None
    chroma_port: int | None = None
    chroma_ssl: bool = False

    # Use a sentence-embedding model by default and align dim
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    generator_model: str = "Qwen/Qwen2.5-1.8B-Instruct"
    generator_max_new_tokens: int = 512
    generator_temperature: float = 0.3

    max_chunks: int = 5
    chunk_size: int = 600
    chunk_overlap: int = 100
    use_token_splitter: bool = False
    tokens_per_chunk: int = 256
    token_overlap: int = 50
    retrieval_rerank_lexical: bool = False
    # Cross-encoder reranker (optional, requires sentence-transformers)
    cross_encoder_use: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_n: int = 10
    cross_encoder_device: str | None = None

    use_model_embeddings: bool = False
    use_model_generator: bool = False

    evaluation_min_recall: float = 0.5
    evaluation_min_mrr: float = 0.5

    # API & upload safety
    allowed_extensions: tuple[str, ...] | str = (".pdf", ".docx", ".txt", ".md")
    max_files: int = 12
    max_upload_size_mb: int = 25  # per file
    max_total_upload_mb: int = 100

    # CORS
    cors_allow_origins: tuple[str, ...] = ()  # e.g., ("*") to allow all
    cors_allow_credentials: bool = False
    cors_allow_methods: tuple[str, ...] = ("GET", "POST", "DELETE", "OPTIONS")
    cors_allow_headers: tuple[str, ...] = ("*",)

    # Security
    api_key: str | None = None  # if set, required in X-API-Key header
    rate_limit_requests: int = 120  # per window per client
    rate_limit_window_seconds: int = 60

    # Remote ingestion
    allowed_ingest_domains: tuple[str, ...] | str = ()  # empty means block external URLs by default
    max_download_size_mb: int = 25

    # MIME allowlist (JSON mapping from extension to list of allowed mime types)
    allowed_mime_json: str | None = None

    @property
    def is_test(self) -> bool:
        return self.environment == "test"

    @property
    def allowed_extensions_tuple(self) -> tuple[str, ...]:
        value = self.allowed_extensions
        if isinstance(value, tuple):
            return value
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return tuple(parts) if parts else (".pdf", ".docx", ".txt")
        return (".pdf", ".docx", ".txt")

    @property
    def allowed_mime_map(self) -> dict[str, set[str]]:
        default: dict[str, set[str]] = {
            ".pdf": {"application/pdf"},
            ".txt": {"text/plain", "application/octet-stream"},
            ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
            ".md": {"text/markdown", "text/plain", "application/octet-stream"},
        }
        if not self.allowed_mime_json:
            return default
        try:
            import json

            parsed = json.loads(self.allowed_mime_json)
            if not isinstance(parsed, dict):
                return default
            mapped: dict[str, set[str]] = {}
            for k, v in parsed.items():
                if isinstance(k, str) and isinstance(v, (list, tuple)):
                    mapped[k] = {str(x) for x in v}
            return mapped or default
        except Exception:
            return default


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings(override: Optional[dict[str, object]] = None) -> Settings:
    """Return settings, optionally overriding values without mutating cache."""

    if override:
        return Settings(**override)
    return _cached_settings()

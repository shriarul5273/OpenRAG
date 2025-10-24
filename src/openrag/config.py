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

    use_model_embeddings: bool = False
    use_model_generator: bool = False

    evaluation_min_recall: float = 0.5
    evaluation_min_mrr: float = 0.5

    # API & upload safety
    allowed_extensions: tuple[str, ...] | str = (".pdf", ".docx", ".txt")
    max_files: int = 12
    max_upload_size_mb: int = 25  # per file
    max_total_upload_mb: int = 100

    # CORS
    cors_allow_origins: tuple[str, ...] = ()  # e.g., ("*") to allow all
    cors_allow_credentials: bool = False
    cors_allow_methods: tuple[str, ...] = ("GET", "POST", "DELETE", "OPTIONS")
    cors_allow_headers: tuple[str, ...] = ("*",)

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


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings(override: Optional[dict[str, object]] = None) -> Settings:
    """Return settings, optionally overriding values without mutating cache."""

    if override:
        return Settings(**override)
    return _cached_settings()

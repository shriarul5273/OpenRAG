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

    embedding_model: str = "Qwen/Qwen2.5-0.5B"
    embedding_dim: int = 1536

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

    @property
    def is_test(self) -> bool:
        return self.environment == "test"


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings(override: Optional[dict[str, object]] = None) -> Settings:
    """Return settings, optionally overriding values without mutating cache."""

    if override:
        return Settings(**override)
    return _cached_settings()

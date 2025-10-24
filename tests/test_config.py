from __future__ import annotations

from openrag.config import get_settings


def test_defaults_embedding_model_and_dim():
    settings = get_settings({})
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dim == 384


def test_upload_limits_defaults():
    settings = get_settings({})
    assert settings.max_files >= 1
    assert settings.max_upload_size_mb >= 1
    assert settings.max_total_upload_mb >= settings.max_upload_size_mb


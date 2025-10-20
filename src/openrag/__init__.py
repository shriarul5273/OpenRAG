"""OpenRAG: Retrieval-Augmented Generation reference implementation."""

from importlib import metadata


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return metadata.version("openrag")
        except metadata.PackageNotFoundError:  # pragma: no cover - during local dev w/out install
            return "0.0.0"
    raise AttributeError(name)


__all__ = ["__version__"]

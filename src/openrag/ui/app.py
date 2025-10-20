"""Gradio-based chat interface for OpenRAG."""

from __future__ import annotations

import mimetypes
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _ensure_hf_folder() -> None:
    """Gradio still expects `huggingface_hub.HfFolder`; reintroduce it if removed."""

    try:
        import huggingface_hub  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return

    if hasattr(huggingface_hub, "HfFolder"):
        return

    from huggingface_hub import constants
    from huggingface_hub.utils import get_token

    class HfFolder:  # pragma: no cover - mirrors upstream compatibility shim
        path_token = Path(constants.HF_TOKEN_PATH)
        _old_path_token = Path(getattr(constants, "_OLD_HF_TOKEN_PATH", constants.HF_TOKEN_PATH))

        @classmethod
        def save_token(cls, token: str) -> None:
            cls.path_token.parent.mkdir(parents=True, exist_ok=True)
            cls.path_token.write_text(token)

        @classmethod
        def get_token(cls):
            try:
                cls._copy_to_new_path_and_warn()
            except Exception:
                pass
            return get_token()

        @classmethod
        def delete_token(cls) -> None:
            for target in (cls.path_token, cls._old_path_token):
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass

        @classmethod
        def _copy_to_new_path_and_warn(cls) -> None:
            if cls._old_path_token.exists() and not cls.path_token.exists():
                cls.save_token(cls._old_path_token.read_text())
                warnings.warn(
                    "Copied legacy Hugging Face token to new location. You can now delete the old file manually.",
                )

    huggingface_hub.HfFolder = HfFolder


_ensure_hf_folder()

import gradio as gr
import httpx

from openrag.api.schemas import DocumentIngestionResponse, QueryResponse

DEFAULT_API_URL = os.getenv("OPENRAG_API_URL", "http://localhost:8000")


class APIError(RuntimeError):
    """Raised when communication with the OpenRAG API fails."""


@dataclass
class OpenRAGClient:
    """HTTPX-based client for the OpenRAG FastAPI service."""

    base_url: str = DEFAULT_API_URL
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def upload_documents(self, paths: Sequence[Path]) -> DocumentIngestionResponse:
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for path in paths:
            mime, _ = mimetypes.guess_type(path.name)
            data = path.read_bytes()
            files.append(("files", (path.name, data, mime or "application/octet-stream")))
        response = self._client.post("/documents", files=files)
        if response.status_code >= 400:
            raise APIError(f"Upload failed ({response.status_code}): {response.text}")
        return DocumentIngestionResponse.model_validate(response.json())

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        payload = {"question": question}
        if top_k:
            payload["top_k"] = top_k
        response = self._client.post("/query", json=payload)
        if response.status_code == 404:
            raise APIError("No relevant documents found. Upload content before querying.")
        if response.status_code >= 400:
            raise APIError(f"Query failed ({response.status_code}): {response.text}")
        return QueryResponse.model_validate(response.json())

    def close(self) -> None:
        self._client.close()


def _normalize_paths(files: Iterable[object]) -> list[Path]:
    normalized: list[Path] = []
    for file in files or []:
        if isinstance(file, Path):
            normalized.append(file)
        elif isinstance(file, str):
            normalized.append(Path(file))
        elif hasattr(file, "name"):
            normalized.append(Path(getattr(file, "name")))
    return normalized


def _format_citations(citations: Sequence[dict]) -> str:
    if not citations:
        return "Sources: none"
    lines = []
    for index, citation in enumerate(citations, start=1):
        source = citation.get("source_path", "unknown")
        score = citation.get("score", 0.0)
        lines.append(f"[{index}] {source} (score {score:.2f})")
    return "Sources:\n" + "\n".join(lines)



def create_upload_handler(client: OpenRAGClient):
    def handle_upload(files: list[object]) -> str:
        paths = _normalize_paths(files)
        if not paths:
            return "⚠️ Please choose PDF, DOCX, or TXT files to upload."
        try:
            response = client.upload_documents(paths)
        except APIError as exc:
            return f"⚠️ Upload failed: {exc}"
        summary = [f"{doc.source_path} ({doc.chunk_count} chunks)" for doc in response.documents]
        return "✅ Indexed documents:\n" + "\n".join(summary)

    return handle_upload


def create_query_handler(client: OpenRAGClient):
    def handle_query(message: str, history: list, top_k_value: int) -> str:  # noqa: ARG001 - history handled by Gradio
        if not message.strip():
            return "⚠️ Enter a question."
        try:
            response = client.query(message, top_k=top_k_value)
        except APIError as exc:
            return f"⚠️ {exc}"
        citations = [citation.model_dump() for citation in response.citations]
        formatted_sources = _format_citations(citations)
        return f"{response.answer}\n\n{formatted_sources}"

    return handle_query


def build_interface(base_url: str | None = None, client: OpenRAGClient | None = None) -> gr.Blocks:
    api_client = client or OpenRAGClient(base_url=base_url or DEFAULT_API_URL)
    handle_upload = create_upload_handler(api_client)
    handle_query = create_query_handler(api_client)

    with gr.Blocks(title="OpenRAG Chat") as demo:
        gr.Markdown("## OpenRAG Chat Interface")
        with gr.Row():
            with gr.Column(scale=1):
                upload_input = gr.File(label="Upload documents", file_count="multiple", file_types=[".pdf", ".docx", ".txt"])
                upload_status = gr.Markdown("Ready to ingest documents.")
                upload_button = gr.Button("Sync Documents", variant="primary")
                upload_button.click(fn=handle_upload, inputs=upload_input, outputs=upload_status)
            with gr.Column(scale=2):
                top_k_slider = gr.Slider(
                    label="Top-k Chunks",
                    value=3,
                    minimum=1,
                    maximum=10,
                    step=1,
                )
                chat = gr.ChatInterface(
                    fn=handle_query,
                    additional_inputs=[top_k_slider],
                    chatbot=gr.Chatbot(height=420),
                    textbox=gr.Textbox(placeholder="Ask a question about your documents..."),
                    retry_btn=None,
                    undo_btn=None,
                    clear_btn="Clear",
                )
        gr.Markdown(
            "Tip: set `OPENRAG_API_URL` before launching to point the UI at a remote backend."
        )

    return demo


def launch(*, base_url: str | None = None, share: bool = False) -> None:
    """Launch the Gradio interface."""

    demo = build_interface(base_url=base_url)
    demo.launch(share=share)


if __name__ == "__main__":
    launch()

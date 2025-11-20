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

    def upload_documents(self, paths: Sequence[Path], *, dataset_id: str | None = None, api_key: str | None = None) -> DocumentIngestionResponse:
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for path in paths:
            mime, _ = mimetypes.guess_type(path.name)
            data = path.read_bytes()
            files.append(("files", (path.name, data, mime or "application/octet-stream")))
        data = {"dataset_id": dataset_id} if dataset_id else None
        headers = {"X-API-Key": api_key} if api_key else None
        response = self._client.post("/documents", files=files, data=data, headers=headers)
        if response.status_code >= 400:
            cid = response.headers.get("X-Correlation-ID", "-")
            raise APIError(f"Upload failed ({response.status_code}) [cid={cid}]: {response.text}")
        return DocumentIngestionResponse.model_validate(response.json())

    def query(self, question: str, top_k: int | None = None, *, dataset_id: str | None = None, api_key: str | None = None) -> QueryResponse:
        payload = {"question": question}
        if top_k:
            payload["top_k"] = top_k
        if dataset_id:
            payload["dataset_id"] = dataset_id
        headers = {"X-API-Key": api_key} if api_key else None
        response = self._client.post("/query", json=payload, headers=headers)
        if response.status_code == 404:
            cid = response.headers.get("X-Correlation-ID", "-")
            raise APIError(f"No relevant documents found [cid={cid}]. Upload content before querying.")
        if response.status_code >= 400:
            cid = response.headers.get("X-Correlation-ID", "-")
            raise APIError(f"Query failed ({response.status_code}) [cid={cid}]: {response.text}")
        return QueryResponse.model_validate(response.json())

    def stream_query(self, question: str, top_k: int | None = None, *, dataset_id: str | None = None, api_key: str | None = None):
        payload: dict[str, object] = {"question": question}
        if top_k:
            payload["top_k"] = top_k
        if dataset_id:
            payload["dataset_id"] = dataset_id
        headers = {"Accept": "text/event-stream"}
        if api_key:
            headers["X-API-Key"] = api_key
        with self._client.stream("POST", "/query/stream", json=payload, headers=headers) as r:
            if r.status_code >= 400:
                cid = r.headers.get("X-Correlation-ID", "-")
                raise APIError(f"Stream failed ({r.status_code}) [cid={cid}]")
            buffer = ""
            for byte_chunk in r.iter_bytes():
                if not byte_chunk:
                    continue
                buffer += byte_chunk.decode("utf-8", errors="ignore")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    if event.startswith("data: "):
                        yield event[len("data: ") :]
                    # ignore named events here; UI adds sources at the end

    def close(self) -> None:
        self._client.close()

    def index_stats(self, api_key: str | None = None) -> dict:
        headers = {"X-API-Key": api_key} if api_key else None
        r = self._client.get("/index/stats", headers=headers)
        if r.status_code >= 400:
            raise APIError(f"Stats failed ({r.status_code}): {r.text}")
        return r.json()

    def reset_index(self, dataset_id: str | None = None, api_key: str | None = None) -> None:
        headers = {"X-API-Key": api_key} if api_key else None
        params = {"dataset_id": dataset_id} if dataset_id else None
        r = self._client.delete("/index", headers=headers, params=params)
        if r.status_code >= 400:
            raise APIError(f"Reset failed ({r.status_code}): {r.text}")


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
    def handle_upload(files: list[object], dataset: str | None = None, api_key: str | None = None) -> str:
        paths = _normalize_paths(files)
        if not paths:
            return "⚠️ Please choose PDF, DOCX, or TXT files to upload."
        try:
            ds = (dataset or "").strip() or None
            response = client.upload_documents(paths, dataset_id=ds, api_key=(api_key or None))
        except APIError as exc:
            return f"⚠️ Upload failed: {exc}"
        summary = [f"{doc.source_path} ({doc.chunk_count} chunks)" for doc in response.documents]
        return "✅ Indexed documents:\n" + "\n".join(summary)

    return handle_upload


def create_query_handler(client: OpenRAGClient):
    def handle_query(message: str, history: list, top_k_value: int, dataset: str | None = None, stream: bool = True, api_key: str | None = None):  # noqa: ARG001 - history handled by Gradio
        if not message.strip():
            return "⚠️ Enter a question."
        if stream:
            try:
                ds = (dataset or "").strip() or None
                for chunk in client.stream_query(message, top_k=top_k_value, dataset_id=ds, api_key=(api_key or None)):
                    yield chunk
                # After stream ends, fetch full response to show sources
                response = client.query(message, top_k=top_k_value, dataset_id=ds, api_key=(api_key or None))
                citations = [citation.model_dump() for citation in response.citations]
                formatted_sources = _format_citations(citations)
                yield "\n\n" + formatted_sources
            except APIError as exc:
                yield f"⚠️ {exc}"
        else:
            try:
                ds = (dataset or "").strip() or None
                response = client.query(message, top_k=top_k_value, dataset_id=ds, api_key=(api_key or None))
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
            with gr.Column(scale=2):
                top_k_slider = gr.Slider(
                    label="Top-k Chunks",
                    value=3,
                    minimum=1,
                    maximum=10,
                    step=1,
                )
                dataset_box = gr.Dropdown(
                    label="Dataset (optional)",
                    choices=["default"],
                    allow_custom_value=True,
                    value="default",
                    interactive=True,
                )
                stream_checkbox = gr.Checkbox(label="Stream responses", value=True)
                api_key_box = gr.Textbox(label="API Key (optional)", type="password")
                # Wire upload to include dataset
                upload_button.click(fn=handle_upload, inputs=[upload_input, dataset_box, api_key_box], outputs=upload_status)
                chat = gr.ChatInterface(
                    fn=handle_query,
                    additional_inputs=[top_k_slider, dataset_box, stream_checkbox, api_key_box],
                    chatbot=gr.Chatbot(height=420),
                    textbox=gr.Textbox(placeholder="Ask a question about your documents..."),
                    retry_btn=None,
                    undo_btn=None,
                    clear_btn="Clear",
                )
        gr.Markdown(
            "Tip: set `OPENRAG_API_URL` before launching to point the UI at a remote backend."
        )

        gr.Markdown("### Admin")
        with gr.Row():
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh Stats")
                reset_btn = gr.Button("Reset Dataset")
            with gr.Column(scale=2):
                stats_md = gr.Markdown("(stats will appear here)")

        def _format_stats(api_key: str | None, dataset: str | None):  # noqa: ARG001 - dataset present for symmetry
            try:
                stats = api_client.index_stats(api_key=api_key or None)
            except APIError as exc:
                return gr.Dropdown.update(), f"⚠️ {exc}"
            lines = [
                f"Collection: {stats.get('collection')}",
                f"Total chunks: {stats.get('total_chunks')}",
                "Datasets:",
            ]
            choices: list[str] = []
            for item in stats.get("datasets", []):
                lines.append(f"- {item.get('dataset_id')}: {item.get('chunks')} chunks")
                if ds := item.get("dataset_id"):
                    choices.append(str(ds))
            if "default" not in choices:
                choices.append("default")
            return gr.Dropdown.update(choices=choices, value=(dataset if dataset in choices else "default")), "\n".join(lines)

        def _reset_dataset(dataset: str | None, api_key: str | None):
            try:
                ds = (dataset or "").strip() or None
                api_client.reset_index(dataset_id=ds, api_key=api_key or None)
            except APIError as exc:
                return f"⚠️ {exc}"
            return "✅ Reset complete"

        refresh_btn.click(_format_stats, inputs=[api_key_box, dataset_box], outputs=[dataset_box, stats_md])
        reset_btn.click(_reset_dataset, inputs=[dataset_box, api_key_box], outputs=stats_md)

    return demo


def launch(*, base_url: str | None = None, share: bool = False) -> None:
    """Launch the Gradio interface."""

    demo = build_interface(base_url=base_url)
    demo.launch(share=share)


if __name__ == "__main__":
    launch()

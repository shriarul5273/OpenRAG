"""Generation backends for OpenRAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from openrag.models import RetrievedChunk

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for answer generation."""

    model: str = "Qwen/Qwen2.5-1.8B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.3
    use_model: bool = False
    device: str | None = None


class GenerationBackend(Protocol):
    """Protocol describing generation behaviour."""

    def generate(self, *, question: str, context: str, citations: Sequence[RetrievedChunk]) -> str:
        """Return a grounded answer for the supplied question and context."""


class TemplateGenerator:
    """Simple deterministic generator used for tests and offline environments."""

    def generate(self, *, question: str, context: str, citations: Sequence[RetrievedChunk]) -> str:
        if not citations:
            return "I do not have enough relevant context to answer that question."
        summary = citations[0].chunk.text
        sources = "\n".join(
            f"[{index + 1}] {c.chunk.document_metadata.source_path} (chunk {c.chunk.order})"
            for index, c in enumerate(citations)
        )
        return (
            f"Summary: {summary}\n\n"
            f"Answer: Based on the provided documents, here is the best match for your question '{question}'.\n"
            f"Sources:\n{sources}"
        )


class QwenGenerator:
    """Generator that optionally calls into Qwen models via Transformers."""

    def __init__(self, config: GenerationConfig | None = None, fallback: GenerationBackend | None = None) -> None:
        self._config = config or GenerationConfig()
        self._fallback = fallback or TemplateGenerator()
        self._tokenizer = None
        self._model = None
        if not self._config.use_model:
            LOGGER.info("QwenGenerator running in template-only mode.")
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            if getattr(self._model.config, "pad_token_id", None) is None and self._tokenizer.pad_token_id is not None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id
            if self._config.device:
                self._model.to(self._config.device)
            LOGGER.info("Loaded generation model %s", self._config.model)
        except Exception as exc:  # pragma: no cover - defensive import
            LOGGER.warning("Falling back to template generator: %s", exc)
            self._tokenizer = None
            self._model = None

    def generate(self, *, question: str, context: str, citations: Sequence[RetrievedChunk]) -> str:
        if self._tokenizer is None or self._model is None:
            return self._fallback.generate(question=question, context=context, citations=citations)
        messages = self._build_messages(question=question, context=context)
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = self._build_prompt(question=question, context=context)
        import torch

        tokenized = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        prompt_length = input_ids.shape[1]
        if self._config.device:
            input_ids = input_ids.to(self._config.device)
            attention_mask = attention_mask.to(self._config.device)
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
            )
        generated_tokens = output[0][prompt_length:]
        generated = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated.strip()

    def _build_prompt(self, *, question: str, context: str) -> str:
        context_block = f"Context:\n{context}\n\n" if context else ""
        return (
            "You are a retrieval augmented assistant. Answer the question using the provided context.\n"
            f"{context_block}"
            f"Question: {question}\nAnswer with citations in the form [index]."
        )

    def _build_messages(self, *, question: str, context: str) -> list[dict[str, str]]:
        if not hasattr(self._tokenizer, "apply_chat_template"):
            return []
        system_context = (
            "You are a retrieval augmented assistant. Use the provided context snippets to answer the user. "
            "Cite sources using [index] references. If the context is empty, say you do not have enough information."
        )
        if context:
            system_context += f"\n\nContext:\n{context}"
        return [
            {"role": "system", "content": system_context},
            {"role": "user", "content": question},
        ]

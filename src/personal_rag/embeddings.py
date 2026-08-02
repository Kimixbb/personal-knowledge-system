from __future__ import annotations

import threading
from typing import Any

from langchain_core.embeddings import Embeddings

from personal_rag.config import DEFAULT_EMBEDDING_REVISION


QUERY_INSTRUCTION = (
    "Given a user question, retrieve relevant passages from a personal knowledge "
    "archive containing Chinese and English materials about work, school, books, "
    "and everyday life."
)


class Qwen3Embeddings(Embeddings):
    """Lazy, normalized Qwen3 embeddings with separate query/document behavior."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        revision: str = DEFAULT_EMBEDDING_REVISION,
        dimensions: int = 1024,
        device: str = "cuda",
        batch_size: int = 16,
        query_instruction: str = QUERY_INSTRUCTION,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.dimensions = dimensions
        self.requested_device = device
        self.batch_size = batch_size
        self.query_instruction = query_instruction.strip()
        self._model: Any | None = None
        self._device: str | None = None
        self._load_lock = threading.Lock()

    @property
    def active_device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self) -> str:
        if self.requested_device != "cuda":
            return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except (ImportError, RuntimeError):
            return "cpu"

    def _get_model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_name,
                        revision=self.revision,
                        device=self.active_device,
                        trust_remote_code=True,
                        truncate_dim=self.dimensions,
                    )
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        values = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        return [[float(component) for component in vector] for vector in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        instructed = f"Instruct: {self.query_instruction}\nQuery:{text}"
        return self._encode([instructed])[0]

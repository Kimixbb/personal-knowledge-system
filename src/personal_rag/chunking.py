from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from personal_rag.config import DEFAULT_EMBEDDING_REVISION


MULTILINGUAL_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ".",
    "!",
    "?",
    ";",
    " ",
    "",
]


class TokenizerLength:
    """Lazy Hugging Face token counting for the configured embedding model."""

    def __init__(self, model_name: str, revision: str) -> None:
        self.model_name = model_name
        self.revision = revision

    @cached_property
    def tokenizer(self):  # type: ignore[no-untyped-def]
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=True,
        )

    def __call__(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class MultilingualChunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        *,
        model_name: str | None = None,
        model_revision: str = DEFAULT_EMBEDDING_REVISION,
        length_function: Callable[[str], int] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if length_function is None:
            if not model_name:
                raise ValueError("model_name is required when no length function is supplied")
            length_function = TokenizerLength(model_name, model_revision)
        self._splitter = RecursiveCharacterTextSplitter(
            separators=MULTILINGUAL_SEPARATORS,
            keep_separator="end",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            add_start_index=True,
            strip_whitespace=True,
        )

    def split(self, documents: Sequence[Document]) -> list[Document]:
        """Split each loader unit independently, then number chunks per source file."""

        chunks: list[Document] = []
        for document in documents:
            if not document.page_content.strip():
                continue
            page_chunks = self._splitter.create_documents(
                [document.page_content], [document.metadata]
            )
            chunks.extend(chunk for chunk in page_chunks if chunk.page_content.strip())
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
        return chunks

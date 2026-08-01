from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from personal_rag.vector_store import ChromaVectorStore


class TinyEmbeddings(Embeddings):
    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_chunk(text: str, document_id: str, chunk_id: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "relative_path": f"library/{document_id}.txt",
            "chunk_id": chunk_id,
        },
    )


def test_persistent_chroma_replace_search_delete_and_rebuild(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    store = ChromaVectorStore(persist_dir, TinyEmbeddings(), "test_collection")
    apple = make_chunk("apple fact", "doc-1", "doc-1:old:0")
    store.replace_document("doc-1", [apple], ["doc-1:old:0"])

    # Retrying the same deterministic ID is idempotent.
    store.replace_document("doc-1", [apple], ["doc-1:old:0"])
    assert store.count_documents() == 1
    assert store.count_chunks() == 1

    reopened = ChromaVectorStore(persist_dir, TinyEmbeddings(), "test_collection")
    results = reopened.similarity_search_with_scores("apple", k=1)
    assert results[0][0].page_content == "apple fact"
    assert results[0][1] == 1.0

    orange = make_chunk("orange fact", "doc-1", "doc-1:new:0")
    reopened.replace_document("doc-1", [orange], ["doc-1:new:0"])
    assert reopened.count_chunks() == 1
    assert reopened.similarity_search_with_scores("orange", k=1)[0][0].page_content == (
        "orange fact"
    )

    reopened.delete_document("doc-1")
    assert reopened.count_chunks() == 0

    reopened.replace_document("doc-1", [apple], ["doc-1:old:0"])
    reopened.rebuild_collection()
    assert reopened.count_chunks() == 0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: Path,
        embedding_function: Embeddings,
        collection_name: str,
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _ids_for_document(self, document_id: str) -> list[str]:
        result = self._store.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        return [str(value) for value in result.get("ids", [])]

    def add_chunks(self, chunks: Sequence[Document], chunk_ids: Sequence[str]) -> None:
        if len(chunks) != len(chunk_ids):
            raise ValueError("Each chunk must have exactly one deterministic ID")
        if chunks:
            self._store.add_documents(list(chunks), ids=list(chunk_ids))

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Document],
        chunk_ids: Sequence[str],
    ) -> None:
        """Embed/add first so an embedding failure leaves the old document intact."""

        old_ids = self._ids_for_document(document_id)
        old_id_set = set(old_ids)
        missing_chunks: list[Document] = []
        missing_ids: list[str] = []
        for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
            if chunk_id not in old_id_set:
                missing_chunks.append(chunk)
                missing_ids.append(chunk_id)
        self.add_chunks(missing_chunks, missing_ids)
        new_ids = set(chunk_ids)
        obsolete = [chunk_id for chunk_id in old_ids if chunk_id not in new_ids]
        if obsolete:
            self._store.delete(ids=obsolete)

    def delete_document(self, document_id: str) -> None:
        chunk_ids = self._ids_for_document(document_id)
        if chunk_ids:
            self._store.delete(ids=chunk_ids)

    def similarity_search_with_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def rebuild_collection(self) -> None:
        self._store.reset_collection()

    def count_chunks(self) -> int:
        return int(self._store._collection.count())

    def count_documents(self) -> int:
        result = self._store.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return len(
            {
                str(metadata["document_id"])
                for metadata in metadatas
                if metadata and "document_id" in metadata
            }
        )

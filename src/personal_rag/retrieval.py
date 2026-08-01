from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.documents import Document


class SearchableVectorStore(Protocol):
    def similarity_search_with_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]: ...


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    source_id: str
    rank: int
    score: float
    chunk_id: str
    content: str
    relative_path: str
    page: int | None
    metadata: dict[str, Any]


class Retriever:
    def __init__(
        self,
        vector_store: SearchableVectorStore,
        *,
        top_k: int = 5,
        minimum_score: float = 0.2,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.minimum_score = minimum_score

    def retrieve(self, question: str) -> list[RetrievedPassage]:
        results = self.vector_store.similarity_search_with_scores(
            question, k=self.top_k
        )
        passages: list[RetrievedPassage] = []
        for document, score in results:
            numeric_score = float(score)
            if numeric_score < self.minimum_score:
                continue
            metadata = dict(document.metadata)
            page_value = metadata.get("page")
            page = int(page_value) if page_value is not None else None
            rank = len(passages) + 1
            passages.append(
                RetrievedPassage(
                    source_id=f"S{rank}",
                    rank=rank,
                    score=numeric_score,
                    chunk_id=str(metadata.get("chunk_id", "")),
                    content=document.page_content,
                    relative_path=str(metadata.get("relative_path", "")),
                    page=page,
                    metadata=metadata,
                )
            )
        return passages

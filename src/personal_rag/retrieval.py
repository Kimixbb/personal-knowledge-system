from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.documents import Document


class SearchableVectorStore(Protocol):
    def similarity_search_with_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]: ...

    def keyword_search_with_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]: ...


@dataclass(slots=True)
class _HybridCandidate:
    document: Document
    dense_score: float = 0.0
    keyword_score: float = 0.0


def _candidate_key(document: Document) -> str:
    metadata = document.metadata
    chunk_id = str(metadata.get("chunk_id", ""))
    if chunk_id:
        return chunk_id
    return "\n".join(
        (
            str(metadata.get("relative_path", "")),
            str(metadata.get("page", "")),
            document.page_content,
        )
    )


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
        dense_weight: float = 0.5,
        keyword_weight: float = 0.5,
        candidate_multiplier: int = 5,
    ) -> None:
        if dense_weight < 0 or keyword_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if dense_weight + keyword_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least one")
        self.vector_store = vector_store
        self.top_k = top_k
        self.minimum_score = minimum_score
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, question: str) -> list[RetrievedPassage]:
        candidate_k = self.top_k * self.candidate_multiplier
        dense_results = self.vector_store.similarity_search_with_scores(
            question, k=candidate_k
        )
        keyword_results = self.vector_store.keyword_search_with_scores(
            question, k=candidate_k
        )
        candidates: dict[str, _HybridCandidate] = {}
        for document, score in dense_results:
            numeric_score = float(score)
            if numeric_score < self.minimum_score:
                continue
            key = _candidate_key(document)
            candidate = candidates.setdefault(key, _HybridCandidate(document))
            candidate.dense_score = max(
                candidate.dense_score, min(max(numeric_score, 0.0), 1.0)
            )

        maximum_keyword_score = max(
            (float(score) for _, score in keyword_results), default=0.0
        )
        if maximum_keyword_score > 0:
            for document, score in keyword_results:
                numeric_score = max(float(score), 0.0) / maximum_keyword_score
                key = _candidate_key(document)
                candidate = candidates.setdefault(key, _HybridCandidate(document))
                candidate.keyword_score = max(candidate.keyword_score, numeric_score)

        dense_available = any(
            candidate.dense_score > 0 for candidate in candidates.values()
        )
        keyword_available = any(
            candidate.keyword_score > 0 for candidate in candidates.values()
        )
        active_weight = (
            (self.dense_weight if dense_available else 0.0)
            + (self.keyword_weight if keyword_available else 0.0)
        )
        if active_weight == 0:
            return []

        scored_candidates = [
            (
                candidate,
                (
                    self.dense_weight * candidate.dense_score
                    + self.keyword_weight * candidate.keyword_score
                )
                / active_weight,
            )
            for candidate in candidates.values()
        ]
        scored_candidates.sort(
            key=lambda item: (
                -item[1],
                -item[0].keyword_score,
                -item[0].dense_score,
                _candidate_key(item[0].document),
            )
        )
        passages: list[RetrievedPassage] = []
        for candidate, hybrid_score in scored_candidates[: self.top_k]:
            document = candidate.document
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "dense_score": candidate.dense_score,
                    "keyword_score": candidate.keyword_score,
                    "hybrid_score": hybrid_score,
                }
            )
            page_value = metadata.get("page")
            page = int(page_value) if page_value is not None else None
            rank = len(passages) + 1
            passages.append(
                RetrievedPassage(
                    source_id=f"S{rank}",
                    rank=rank,
                    score=hybrid_score,
                    chunk_id=str(metadata.get("chunk_id", "")),
                    content=document.page_content,
                    relative_path=str(metadata.get("relative_path", "")),
                    page=page,
                    metadata=metadata,
                )
            )
        return passages

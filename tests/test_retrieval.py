from __future__ import annotations

from langchain_core.documents import Document

from personal_rag.retrieval import Retriever


class FakeSearchStore:
    def similarity_search_with_scores(self, query: str, k: int):  # type: ignore[no-untyped-def]
        assert query == "生日是哪天？"
        assert k == 5
        return [
            (
                Document(
                    page_content="生日是八月七日。",
                    metadata={
                        "chunk_id": "doc:hash:0",
                        "relative_path": "library/inbox/friends.md",
                        "page": 2,
                    },
                ),
                0.91,
            ),
            (
                Document(
                    page_content="irrelevant",
                    metadata={
                        "chunk_id": "doc:hash:1",
                        "relative_path": "library/other.txt",
                    },
                ),
                0.05,
            ),
        ]


def test_retrieval_preserves_score_metadata_and_assigns_source_ids() -> None:
    retriever = Retriever(FakeSearchStore(), top_k=5, minimum_score=0.2)

    passages = retriever.retrieve("生日是哪天？")

    assert len(passages) == 1
    assert passages[0].source_id == "S1"
    assert passages[0].rank == 1
    assert passages[0].score == 0.91
    assert passages[0].page == 2
    assert passages[0].chunk_id == "doc:hash:0"

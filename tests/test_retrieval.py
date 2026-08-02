from __future__ import annotations

from langchain_core.documents import Document

from personal_rag.retrieval import Retriever


class FakeSearchStore:
    def similarity_search_with_scores(self, query: str, k: int):  # type: ignore[no-untyped-def]
        assert query == "生日是哪天？"
        assert k == 25
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

    def keyword_search_with_scores(self, query: str, k: int):  # type: ignore[no-untyped-def]
        assert query == "生日是哪天？"
        assert k == 25
        return []


class FakeHybridSearchStore:
    def similarity_search_with_scores(self, query: str, k: int):  # type: ignore[no-untyped-def]
        assert query == "Which launch used the codename Juniper?"
        assert k == 10
        return [
            (
                Document(
                    page_content="General launch planning notes.",
                    metadata={
                        "chunk_id": "dense",
                        "relative_path": "library/planning.md",
                    },
                ),
                0.8,
            )
        ]

    def keyword_search_with_scores(self, query: str, k: int):  # type: ignore[no-untyped-def]
        assert query == "Which launch used the codename Juniper?"
        assert k == 10
        return [
            (
                Document(
                    page_content="The product launch codename was Juniper.",
                    metadata={
                        "chunk_id": "keyword",
                        "relative_path": "library/codenames.md",
                    },
                ),
                7.5,
            )
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
    assert passages[0].metadata["dense_score"] == 0.91
    assert passages[0].metadata["keyword_score"] == 0.0


def test_hybrid_retrieval_can_surface_a_keyword_only_candidate() -> None:
    retriever = Retriever(FakeHybridSearchStore(), top_k=2, minimum_score=0.2)

    passages = retriever.retrieve("Which launch used the codename Juniper?")

    assert [passage.chunk_id for passage in passages] == ["keyword", "dense"]
    assert passages[0].score == 0.5
    assert passages[0].metadata["dense_score"] == 0.0
    assert passages[0].metadata["keyword_score"] == 1.0
    assert passages[1].score == 0.4

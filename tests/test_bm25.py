from __future__ import annotations

from langchain_core.documents import Document

from personal_rag.bm25 import BM25Index


def test_bm25_ranks_the_chunk_with_distinctive_query_terms_first() -> None:
    target = Document(
        page_content=(
            "The perfume brand Charlie used a masculine name and pantsuit ads."
        ),
        metadata={"chunk_id": "target"},
    )
    generic = Document(
        page_content="A brand name gives customers a memorable identity.",
        metadata={"chunk_id": "generic"},
    )
    index = BM25Index([generic, target])

    results = index.search("masculine perfume brand", k=2)

    assert results[0][0].metadata["chunk_id"] == "target"
    assert results[0][1] > results[1][1] > 0


def test_bm25_returns_no_candidates_without_shared_terms() -> None:
    index = BM25Index(
        [Document(page_content="orchids need filtered light", metadata={})]
    )

    assert index.search("quarterly finance report", k=5) == []


def test_bm25_matches_cjk_character_bigrams() -> None:
    birthday = Document(
        page_content="生日是八月七日。",
        metadata={"chunk_id": "birthday"},
    )
    meeting = Document(
        page_content="会议安排在星期三。",
        metadata={"chunk_id": "meeting"},
    )

    results = BM25Index([meeting, birthday]).search("生日是哪天？", k=2)

    assert results[0][0].metadata["chunk_id"] == "birthday"

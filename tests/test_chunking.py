from __future__ import annotations

from langchain_core.documents import Document

from personal_rag.chunking import MultilingualChunker


def character_length(text: str) -> int:
    return len(text)


def test_chunks_pages_independently_and_preserves_metadata() -> None:
    chunker = MultilingualChunker(
        chunk_size=12,
        chunk_overlap=3,
        length_function=character_length,
    )
    documents = [
        Document(
            page_content="第一句。第二句。第三句。",
            metadata={"relative_path": "library/book.pdf", "file_type": "pdf", "page": 42},
        ),
        Document(
            page_content="English one. English two.",
            metadata={"relative_path": "library/book.pdf", "file_type": "pdf", "page": 43},
        ),
    ]

    chunks = chunker.split(documents)

    assert len(chunks) >= 3
    assert {chunk.metadata["page"] for chunk in chunks} == {42, 43}
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert all(isinstance(chunk.metadata["start_index"], int) for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_empty_documents_do_not_create_chunks() -> None:
    chunker = MultilingualChunker(
        chunk_size=10,
        chunk_overlap=2,
        length_function=character_length,
    )

    chunks = chunker.split(
        [Document(page_content="   \n", metadata={"relative_path": "library/a.txt"})]
    )

    assert chunks == []

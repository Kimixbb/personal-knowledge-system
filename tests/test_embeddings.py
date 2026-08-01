from __future__ import annotations

from personal_rag.embeddings import Qwen3Embeddings


class FakeArray(list):
    def tolist(self):  # type: ignore[no-untyped-def]
        return list(self)


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> FakeArray:
        self.calls.append((texts, kwargs))
        return FakeArray([[1.0, 0.0] for _ in texts])


def test_documents_are_raw_but_queries_receive_qwen_instruction() -> None:
    embeddings = Qwen3Embeddings(device="cpu", dimensions=2)
    fake = FakeSentenceTransformer()
    embeddings._model = fake

    assert embeddings.embed_documents(["document text"]) == [[1.0, 0.0]]
    assert embeddings.embed_query("question text") == [1.0, 0.0]

    assert fake.calls[0][0] == ["document text"]
    assert fake.calls[1][0][0].startswith("Instruct: ")
    assert fake.calls[1][0][0].endswith("Query:question text")
    assert fake.calls[0][1]["normalize_embeddings"] is True
    assert fake.calls[0][1]["batch_size"] == 16

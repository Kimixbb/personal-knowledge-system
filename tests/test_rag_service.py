from __future__ import annotations

from langchain_core.messages import AIMessage

from personal_rag.rag_service import RAGService
from personal_rag.retrieval import RetrievedPassage


class FakeSynchronizer:
    def __init__(self) -> None:
        self.calls = 0

    def sync_library(self, *, raise_on_errors: bool = False):  # type: ignore[no-untyped-def]
        assert raise_on_errors is True
        self.calls += 1
        return object()


class QuestionRetriever:
    def retrieve(self, question: str) -> list[RetrievedPassage]:
        return [
            RetrievedPassage(
                source_id="S1",
                rank=1,
                score=0.9,
                chunk_id=f"chunk-{question}",
                content=f"Evidence for {question}",
                relative_path="library/inbox/notes.md",
                page=None,
                metadata={},
            )
        ]


class RecordingLLM:
    def __init__(self, answer: str = "Grounded [S1]") -> None:
        self.answer = answer
        self.calls: list[list[object]] = []

    def invoke(self, messages):  # type: ignore[no-untyped-def]
        self.calls.append(list(messages))
        return AIMessage(content=self.answer)


class EmptyRetriever:
    def retrieve(self, question: str) -> list[RetrievedPassage]:
        return []


class FailingLLM:
    def invoke(self, messages):  # type: ignore[no-untyped-def]
        raise RuntimeError("secret provider detail")


def test_each_question_synchronizes_and_sends_no_previous_history() -> None:
    synchronizer = FakeSynchronizer()
    llm = RecordingLLM()
    service = RAGService(synchronizer, QuestionRetriever())

    service.ask("first question", llm, provider="deepseek", model="chat")
    service.ask("second question", llm, provider="deepseek", model="chat")

    assert synchronizer.calls == 2
    assert len(llm.calls) == 2
    second_text = "\n".join(str(message.content) for message in llm.calls[1])
    assert "second question" in second_text
    assert "first question" not in second_text
    assert len(llm.calls[1]) == 2


def test_no_useful_retrieval_refuses_locally_in_question_language() -> None:
    synchronizer = FakeSynchronizer()
    llm = RecordingLLM()
    service = RAGService(synchronizer, EmptyRetriever())

    result = service.ask("我的生日是哪天？", llm, provider="deepseek", model="chat")

    assert "没有找到足够的信息" in result.answer
    assert llm.calls == []
    assert result.exact_hosted_context == ""


def test_invalid_model_citations_are_removed_and_reported() -> None:
    service = RAGService(FakeSynchronizer(), QuestionRetriever())

    result = service.ask(
        "question",
        RecordingLLM("Claim [S1] and invented [S9]."),
        provider="deepseek",
        model="chat",
    )

    assert "[S1]" in result.answer
    assert "[S9]" not in result.answer
    assert result.invalid_citations == ("S9",)


def test_hosted_failure_retains_retrieval_debug_without_leaking_error_text() -> None:
    service = RAGService(FakeSynchronizer(), QuestionRetriever())

    result = service.ask(
        "question", FailingLLM(), provider="deepseek", model="chat"
    )

    assert result.answer == ""
    assert result.hosted_error == "RuntimeError"
    assert result.passages[0].content == "Evidence for question"
    assert "Evidence for question" in result.exact_hosted_context
    assert "secret provider detail" not in result.hosted_error


def test_uncited_hosted_answer_is_not_displayed_as_grounded() -> None:
    service = RAGService(FakeSynchronizer(), QuestionRetriever())

    result = service.ask(
        "question",
        RecordingLLM("An answer without a source."),
        provider="deepseek",
        model="chat",
    )

    assert result.answer == ""
    assert result.hosted_error == "MissingCitation"
    assert result.passages

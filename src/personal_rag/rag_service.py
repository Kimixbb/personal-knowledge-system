from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from personal_rag.prompting import build_grounded_prompt
from personal_rag.providers import normalize_response_text
from personal_rag.retrieval import RetrievedPassage


class SyncBeforeQuestion(Protocol):
    def sync_library(self, *, raise_on_errors: bool = False) -> Any: ...


class PassageRetriever(Protocol):
    def retrieve(self, question: str) -> list[RetrievedPassage]: ...


class ChatModel(Protocol):
    def invoke(self, messages: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class RAGResult:
    question: str
    answer: str
    provider: str
    model: str
    passages: tuple[RetrievedPassage, ...] = ()
    exact_hosted_context: str = ""
    invalid_citations: tuple[str, ...] = ()
    hosted_error: str | None = None
    usage_metadata: dict[str, Any] = field(default_factory=dict)


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def _insufficient_answer(question: str) -> str:
    if _is_chinese(question):
        return "我在知识库中没有找到足够的信息来回答这个问题。"
    return "I could not find enough information in the knowledge archive to answer this question."


def _remove_invalid_citations(
    answer: str, allowed_source_ids: set[str]
) -> tuple[str, tuple[str, ...]]:
    found = set(re.findall(r"\[(S\d+)\]", answer))
    invalid = tuple(sorted(found - allowed_source_ids))
    for source_id in invalid:
        answer = answer.replace(f"[{source_id}]", "")
    answer = re.sub(r"[ \t]+([.,;:!?。！？；])", r"\1", answer)
    return answer.strip(), invalid


class RAGService:
    def __init__(
        self, synchronizer: SyncBeforeQuestion, retriever: PassageRetriever
    ) -> None:
        self.synchronizer = synchronizer
        self.retriever = retriever

    def ask(
        self,
        question: str,
        llm: ChatModel,
        *,
        provider: str,
        model: str,
    ) -> RAGResult:
        question = question.strip()
        if not question:
            return RAGResult(
                question="", answer="", provider=provider, model=model
            )

        self.synchronizer.sync_library(raise_on_errors=True)
        passages = tuple(self.retriever.retrieve(question))
        if not passages:
            return RAGResult(
                question=question,
                answer=_insufficient_answer(question),
                provider=provider,
                model=model,
            )

        prompt = build_grounded_prompt(question, passages)
        try:
            response = llm.invoke(list(prompt.messages))
        except Exception as exc:
            return RAGResult(
                question=question,
                answer="",
                provider=provider,
                model=model,
                passages=passages,
                exact_hosted_context=prompt.exact_hosted_context,
                hosted_error=type(exc).__name__,
            )

        answer = normalize_response_text(response)
        answer, invalid_citations = _remove_invalid_citations(
            answer, {passage.source_id for passage in passages}
        )
        valid_citations = set(re.findall(r"\[(S\d+)\]", answer))
        if answer and not valid_citations:
            return RAGResult(
                question=question,
                answer="",
                provider=provider,
                model=model,
                passages=passages,
                exact_hosted_context=prompt.exact_hosted_context,
                invalid_citations=invalid_citations,
                hosted_error="MissingCitation",
            )
        usage = getattr(response, "usage_metadata", None)
        usage_metadata = dict(usage) if isinstance(usage, dict) else {}
        return RAGResult(
            question=question,
            answer=answer,
            provider=provider,
            model=model,
            passages=passages,
            exact_hosted_context=prompt.exact_hosted_context,
            invalid_citations=invalid_citations,
            usage_metadata=usage_metadata,
        )

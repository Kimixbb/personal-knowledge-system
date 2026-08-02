from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from personal_rag.prompting import INSUFFICIENT_EVIDENCE_TOKEN, build_grounded_prompt
from personal_rag.providers import normalize_response_text
from personal_rag.retrieval import RetrievedPassage


class SyncBeforeQuestion(Protocol):
    def sync_library(self, *, raise_on_errors: bool = False) -> Any: ...


class PassageRetriever(Protocol):
    def retrieve(self, question: str) -> list[RetrievedPassage]: ...


class ChatModel(Protocol):
    def invoke(self, messages: Any) -> Any: ...


class QuestionStage(StrEnum):
    SYNCHRONIZING = "Checking the vault for file changes…"
    RETRIEVING = "Running semantic and keyword retrieval…"
    PREPARING_CONTEXT = "Building the grounded source context…"
    REQUESTING_ANSWER = "Waiting for the hosted model to answer…"
    VALIDATING_CITATIONS = "Validating the answer's source citations…"


class RAGResultKind(StrEnum):
    ANSWERED = "answered"
    NO_RELEVANT_PASSAGES = "no_relevant_passages"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    HOSTED_REFUSAL = "hosted_refusal"
    HOSTED_ERROR = "hosted_error"
    CITATION_ERROR = "citation_error"


StageCallback = Callable[[QuestionStage], None]


@dataclass(frozen=True, slots=True)
class RAGResult:
    question: str
    answer: str
    provider: str
    model: str
    kind: RAGResultKind = RAGResultKind.ANSWERED
    passages: tuple[RetrievedPassage, ...] = ()
    exact_hosted_context: str = ""
    hosted_response_text: str = ""
    invalid_citations: tuple[str, ...] = ()
    hosted_error: str | None = None
    usage_metadata: dict[str, Any] = field(default_factory=dict)


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def _no_relevant_passages_answer(question: str) -> str:
    if _is_chinese(question):
        return "知识库中没有检索到与这个问题相关的段落。"
    return "I could not find any relevant passages in the knowledge archive."


def _insufficient_evidence_answer(question: str) -> str:
    if _is_chinese(question):
        return "我找到了可能相关的段落，但其中没有足够的依据来回答这个问题。"
    return (
        "I found potentially relevant passages, but they did not contain enough "
        "evidence to answer this question."
    )


def _extract_model_refusal(response: Any) -> str:
    for metadata_name in ("additional_kwargs", "response_metadata"):
        metadata = getattr(response, metadata_name, None)
        if isinstance(metadata, dict) and isinstance(metadata.get("refusal"), str):
            refusal = metadata["refusal"].strip()
            if refusal:
                return refusal
    content = getattr(response, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "refusal":
                refusal = block.get("refusal") or block.get("text")
                if isinstance(refusal, str) and refusal.strip():
                    return refusal.strip()
    return ""


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
        on_stage: StageCallback | None = None,
    ) -> RAGResult:
        question = question.strip()
        if not question:
            return RAGResult(
                question="", answer="", provider=provider, model=model
            )

        if on_stage is not None:
            on_stage(QuestionStage.SYNCHRONIZING)
        self.synchronizer.sync_library(raise_on_errors=True)

        if on_stage is not None:
            on_stage(QuestionStage.RETRIEVING)
        passages = tuple(self.retriever.retrieve(question))
        if not passages:
            return RAGResult(
                question=question,
                answer=_no_relevant_passages_answer(question),
                provider=provider,
                model=model,
                kind=RAGResultKind.NO_RELEVANT_PASSAGES,
            )

        if on_stage is not None:
            on_stage(QuestionStage.PREPARING_CONTEXT)
        prompt = build_grounded_prompt(question, passages)

        if on_stage is not None:
            on_stage(QuestionStage.REQUESTING_ANSWER)
        try:
            response = llm.invoke(list(prompt.messages))
        except Exception as exc:
            return RAGResult(
                question=question,
                answer="",
                provider=provider,
                model=model,
                kind=RAGResultKind.HOSTED_ERROR,
                passages=passages,
                exact_hosted_context=prompt.exact_hosted_context,
                hosted_error=type(exc).__name__,
            )

        if on_stage is not None:
            on_stage(QuestionStage.VALIDATING_CITATIONS)
        model_refusal = _extract_model_refusal(response)
        hosted_response_text = model_refusal or normalize_response_text(response)
        if model_refusal:
            return RAGResult(
                question=question,
                answer="",
                provider=provider,
                model=model,
                kind=RAGResultKind.HOSTED_REFUSAL,
                passages=passages,
                exact_hosted_context=prompt.exact_hosted_context,
                hosted_response_text=hosted_response_text,
                hosted_error="ModelRefusal",
            )
        if hosted_response_text == INSUFFICIENT_EVIDENCE_TOKEN:
            return RAGResult(
                question=question,
                answer=_insufficient_evidence_answer(question),
                provider=provider,
                model=model,
                kind=RAGResultKind.INSUFFICIENT_EVIDENCE,
                passages=passages,
                exact_hosted_context=prompt.exact_hosted_context,
                hosted_response_text=hosted_response_text,
            )
        answer = hosted_response_text
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
                kind=RAGResultKind.CITATION_ERROR,
                passages=passages,
                exact_hosted_context=prompt.exact_hosted_context,
                hosted_response_text=hosted_response_text,
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
            hosted_response_text=hosted_response_text,
            invalid_citations=invalid_citations,
            usage_metadata=usage_metadata,
        )

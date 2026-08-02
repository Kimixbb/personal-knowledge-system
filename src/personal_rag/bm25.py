from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from math import log

from langchain_core.documents import Document


_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def tokenize(text: str) -> list[str]:
    """Tokenize Latin text as words and CJK text as characters plus bigrams."""

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.casefold()):
        token = match.group(0)
        if token.endswith(("'s", "’s")):
            token = token[:-2]
        if not token:
            continue
        if _CJK_PATTERN.fullmatch(token):
            characters = list(token)
            tokens.extend(characters)
            tokens.extend(
                characters[index] + characters[index + 1]
                for index in range(len(characters) - 1)
            )
        elif len(token) > 1 or token.isdigit():
            tokens.append(token)
    return tokens


class BM25Index:
    """Small in-memory Okapi BM25 index over the authoritative Chroma chunks."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = tuple(documents)
        self.k1 = k1
        self.b = b
        self._term_frequencies = tuple(
            Counter(tokenize(document.page_content)) for document in self.documents
        )
        self._document_lengths = tuple(
            sum(frequencies.values()) for frequencies in self._term_frequencies
        )
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequencies: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            self._document_frequencies.update(frequencies.keys())

    def _score(self, query_frequencies: Counter[str], document_index: int) -> float:
        if not self.documents or self._average_document_length == 0:
            return 0.0
        frequencies = self._term_frequencies[document_index]
        document_length = self._document_lengths[document_index]
        score = 0.0
        for term, query_frequency in query_frequencies.items():
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = log(
                1
                + (len(self.documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_normalization = self.k1 * (
                1
                - self.b
                + self.b * document_length / self._average_document_length
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * term_frequency
                * (self.k1 + 1)
                / (term_frequency + length_normalization)
            )
        return score

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if k <= 0:
            return []
        query_frequencies = Counter(tokenize(query))
        if not query_frequencies:
            return []
        scored = [
            (self.documents[index], self._score(query_frequencies, index), index)
            for index in range(len(self.documents))
        ]
        matches = [item for item in scored if item[1] > 0]
        matches.sort(key=lambda item: (-item[1], item[2]))
        return [(document, score) for document, score, _ in matches[:k]]

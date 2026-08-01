from __future__ import annotations

from pathlib import Path
import unicodedata

from charset_normalizer import from_bytes
from langchain_core.documents import Document
from pypdf import PdfReader

from personal_rag.vault import SUPPORTED_EXTENSIONS


class DocumentLoadError(RuntimeError):
    """A safe, UI-facing document extraction failure."""


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        matches = list(from_bytes(data))
        if not matches:
            raise DocumentLoadError("Text encoding could not be detected")
        # Very short Western text is inherently ambiguous to statistical
        # detectors. Prefer candidates whose letters consistently belong to
        # the Latin and/or CJK scripts and avoid presentation/control glyphs.
        def quality(match) -> tuple[float, float, float]:  # type: ignore[no-untyped-def]
            text = str(match)
            letters = [character for character in text if character.isalpha()]
            recognized = 0
            unsafe = 0
            for character in text:
                name = unicodedata.name(character, "")
                if character.isalpha() and (
                    "LATIN" in name
                    or "CJK" in name
                    or "IDEOGRAPH" in name
                    or "HIRAGANA" in name
                    or "KATAKANA" in name
                ):
                    recognized += 1
                category = unicodedata.category(character)
                if category.startswith("C") or 0xFB50 <= ord(character) <= 0xFEFF:
                    unsafe += 1
            script_ratio = recognized / max(len(letters), 1)
            return (-float(unsafe), script_ratio, float(match.coherence))

        return str(max(matches, key=quality))


def _load_text(path: Path, relative_path: str, file_type: str) -> list[Document]:
    try:
        text = _decode_text(path.read_bytes())
    except (OSError, UnicodeError) as exc:
        raise DocumentLoadError(f"{type(exc).__name__} while reading text") from exc
    return [
        Document(
            page_content=text,
            metadata={"relative_path": relative_path, "file_type": file_type},
        )
    ]


def _load_pdf(path: Path, relative_path: str) -> list[Document]:
    try:
        reader = PdfReader(path)
        documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "relative_path": relative_path,
                        "file_type": "pdf",
                        "page": page_number,
                    },
                )
            )
        return documents
    except Exception as exc:
        raise DocumentLoadError(f"{type(exc).__name__} while extracting PDF") from exc


def load_document(path: Path, relative_path: str) -> list[Document]:
    """Extract a supported file without including its absolute path in metadata."""

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(f"Unsupported file extension: {extension or '(none)'}")
    if extension == ".pdf":
        return _load_pdf(path, relative_path)
    return _load_text(path, relative_path, extension.removeprefix("."))

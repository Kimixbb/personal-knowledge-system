from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from personal_rag.loaders import DocumentLoadError, load_document


def test_loads_utf8_bom_markdown_with_relative_metadata(tmp_path: Path) -> None:
    source = tmp_path / "笔记.MD"
    source.write_bytes("标题\n生日是八月七日。".encode("utf-8-sig"))

    documents = load_document(source, "library/inbox/笔记.MD")

    assert [document.page_content for document in documents] == [
        "标题\n生日是八月七日。"
    ]
    assert documents[0].metadata == {
        "relative_path": "library/inbox/笔记.MD",
        "file_type": "md",
    }


def test_uses_encoding_detection_after_utf8_fails(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes("café résumé".encode("cp1252"))

    documents = load_document(source, "library/notes.txt")

    assert documents[0].page_content == "café résumé"


def test_empty_pdf_produces_no_documents(tmp_path: Path) -> None:
    source = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with source.open("wb") as handle:
        writer.write(handle)

    assert load_document(source, "library/blank.pdf") == []


def test_pdf_preserves_one_based_pages_and_skips_blank_pages(tmp_path: Path) -> None:
    source = tmp_path / "pages.pdf"
    writer = PdfWriter()
    first = writer.add_blank_page(width=300, height=300)
    writer.add_blank_page(width=300, height=300)
    third = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    for page, text in ((first, "First page"), (third, "Third page")):
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 72 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with source.open("wb") as handle:
        writer.write(handle)

    documents = load_document(source, "library/pages.pdf")

    assert [document.metadata["page"] for document in documents] == [1, 3]
    assert [document.page_content.strip() for document in documents] == [
        "First page",
        "Third page",
    ]


def test_corrupted_pdf_raises_safe_loader_error(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a pdf")

    with pytest.raises(DocumentLoadError):
        load_document(source, "library/broken.pdf")


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "secrets.env"
    source.write_text("KEY=value", encoding="utf-8")

    with pytest.raises(DocumentLoadError):
        load_document(source, "library/secrets.env")

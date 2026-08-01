from __future__ import annotations

from pathlib import Path

import pytest

from personal_rag.file_actions import (
    DuplicateFileError,
    UnsafePathError,
    UnsupportedUploadError,
    open_vault_file,
    save_upload,
    sanitize_upload_name,
)


def test_sanitizes_both_windows_and_posix_traversal_names() -> None:
    assert sanitize_upload_name("../../notes.md") == "notes.md"
    assert sanitize_upload_name(r"..\..\book.PDF") == "book.PDF"


def test_rejects_unsupported_upload() -> None:
    with pytest.raises(UnsupportedUploadError):
        sanitize_upload_name("archive.epub")


def test_upload_never_silently_replaces_existing_file(tmp_path: Path) -> None:
    inbox = tmp_path / "vault" / "library" / "inbox"
    temp = tmp_path / "vault" / ".rag" / "temp"
    inbox.mkdir(parents=True)
    temp.mkdir(parents=True)
    destination = inbox / "notes.md"
    destination.write_text("old", encoding="utf-8")

    with pytest.raises(DuplicateFileError):
        save_upload(b"new", "notes.md", inbox, temp)

    assert destination.read_text(encoding="utf-8") == "old"


def test_confirmed_upload_atomically_replaces_file_and_cleans_temp(tmp_path: Path) -> None:
    inbox = tmp_path / "vault" / "library" / "inbox"
    temp = tmp_path / "vault" / ".rag" / "temp"
    inbox.mkdir(parents=True)
    temp.mkdir(parents=True)
    destination = inbox / "notes.md"
    destination.write_text("old", encoding="utf-8")

    result = save_upload(b"new", "notes.md", inbox, temp, replace=True)

    assert result.destination == destination
    assert result.replaced is True
    assert destination.read_bytes() == b"new"
    assert list(temp.iterdir()) == []


def test_open_file_rejects_absolute_and_outside_paths(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "library").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    with pytest.raises(UnsafePathError):
        open_vault_file(str(outside), vault, opener=lambda _: None)
    with pytest.raises(UnsafePathError):
        open_vault_file("../outside.txt", vault, opener=lambda _: None)


def test_open_file_resolves_citation_inside_library(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    source = vault / "library" / "notes.txt"
    source.parent.mkdir(parents=True)
    source.write_text("hello", encoding="utf-8")
    opened: list[Path] = []

    resolved = open_vault_file(
        "library/notes.txt", vault, opener=lambda path: opened.append(path)
    )

    assert resolved == source.resolve()
    assert opened == [source.resolve()]

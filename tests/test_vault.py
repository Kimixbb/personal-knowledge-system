from __future__ import annotations

from pathlib import Path

from personal_rag.vault import iter_library_files


def test_scanner_is_recursive_case_insensitive_and_ignores_hidden_files(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    library = vault / "library"
    nested = library / "projects"
    nested.mkdir(parents=True)
    (library / "root.TXT").write_text("root", encoding="utf-8")
    (nested / "notes.MD").write_text("nested", encoding="utf-8")
    (nested / "ignore.epub").write_text("unsupported", encoding="utf-8")
    (nested / ".hidden.md").write_text("hidden", encoding="utf-8")

    paths = list(iter_library_files(library, vault))

    assert {path.name for path in paths} == {"root.TXT", "notes.MD"}

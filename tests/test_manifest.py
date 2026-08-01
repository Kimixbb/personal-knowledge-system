from __future__ import annotations

from pathlib import Path

from personal_rag.manifest import DocumentRecord, Manifest


def make_record(**overrides: object) -> DocumentRecord:
    values: dict[str, object] = {
        "document_id": "doc-1",
        "relative_path": "library/inbox/notes.md",
        "extension": ".md",
        "file_size": 10,
        "modified_time_ns": 100,
        "content_hash": "abc",
        "status": "indexed",
        "chunk_count": 2,
        "indexed_at": "2026-08-01T00:00:00+00:00",
        "last_error": None,
    }
    values.update(overrides)
    return DocumentRecord(**values)  # type: ignore[arg-type]


def test_manifest_round_trip_and_counts(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite3")
    manifest.upsert(make_record())
    manifest.upsert(
        make_record(
            document_id="doc-2",
            relative_path="library/blank.pdf",
            status="empty",
            chunk_count=0,
        )
    )

    assert manifest.get("doc-1") == make_record()
    assert manifest.count_documents() == 2
    assert manifest.count_chunks() == 2


def test_upsert_replaces_fingerprint_and_metadata_is_atomic(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite3")
    manifest.upsert(make_record())
    changed = make_record(file_size=20, modified_time_ns=200, content_hash="def")
    manifest.upsert(changed)
    manifest.replace_index_metadata({"model_name": "qwen", "dimensions": "1024"})

    assert manifest.get_by_path(changed.relative_path) == changed
    assert manifest.get_index_metadata() == {
        "model_name": "qwen",
        "dimensions": "1024",
    }


def test_delete_and_state(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.sqlite3")
    manifest.upsert(make_record())
    manifest.set_state("last_sync", "now")

    manifest.delete("doc-1")

    assert manifest.all_documents() == []
    assert manifest.get_state("last_sync") == "now"

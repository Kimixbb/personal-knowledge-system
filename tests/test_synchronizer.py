from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

from personal_rag.config import Settings, ensure_vault
from personal_rag.loaders import DocumentLoadError
from personal_rag.manifest import Manifest
from personal_rag.synchronizer import (
    IndexConfigurationMismatch,
    SyncProgress,
    SyncStage,
    SynchronizationError,
    Synchronizer,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: dict[str, list[Document]] = {}
        self.chunk_ids: dict[str, list[str]] = {}

    def replace_document(
        self, document_id: str, chunks: list[Document], chunk_ids: list[str]
    ) -> None:
        self.documents[document_id] = list(chunks)
        self.chunk_ids[document_id] = list(chunk_ids)

    def delete_document(self, document_id: str) -> None:
        self.documents.pop(document_id, None)
        self.chunk_ids.pop(document_id, None)

    def rebuild_collection(self) -> None:
        self.documents.clear()
        self.chunk_ids.clear()


class IdentityChunker:
    def split(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for index, document in enumerate(documents):
            metadata = dict(document.metadata)
            metadata.update({"chunk_index": index, "start_index": 0})
            chunks.append(Document(page_content=document.page_content, metadata=metadata))
        return chunks


def text_loader(path: Path, relative_path: str) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    if text == "broken":
        raise DocumentLoadError("broken test file")
    return [
        Document(
            page_content=text,
            metadata={"relative_path": relative_path, "file_type": path.suffix[1:]},
        )
    ]


def make_synchronizer(tmp_path: Path):
    settings = Settings(vault_path=tmp_path / "vault")
    ensure_vault(settings)
    manifest = Manifest(settings.manifest_path)
    vectors = FakeVectorStore()
    synchronizer = Synchronizer(
        settings,
        manifest,
        vectors,
        IdentityChunker(),
        loader=text_loader,
    )
    return settings, manifest, vectors, synchronizer


def rewrite_with_new_mtime(path: Path, text: str) -> None:
    old_mtime = path.stat().st_mtime_ns
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))


def test_indexes_new_file_then_skips_unchanged_file(tmp_path: Path) -> None:
    settings, manifest, vectors, synchronizer = make_synchronizer(tmp_path)
    source = settings.inbox_dir / "notes.md"
    source.write_text("Birthday is August 1.", encoding="utf-8")

    first = synchronizer.sync_library()
    second = synchronizer.sync_library()

    assert first.added == 1
    assert second.skipped == 1
    assert manifest.count_documents() == 1
    assert manifest.count_chunks() == 1
    assert next(iter(vectors.documents.values()))[0].page_content.endswith("August 1.")


def test_sync_progress_reports_real_file_processing_stages_in_order(
    tmp_path: Path,
) -> None:
    settings, _, _, synchronizer = make_synchronizer(tmp_path)
    source = settings.inbox_dir / "notes.md"
    source.write_text("Birthday is August 1.", encoding="utf-8")
    progress: list[SyncProgress] = []

    synchronizer.sync_library(on_progress=progress.append)

    assert [item.stage for item in progress] == [
        SyncStage.VERIFYING_INDEX,
        SyncStage.SCANNING,
        SyncStage.CHECKING,
        SyncStage.HASHING,
        SyncStage.EXTRACTING,
        SyncStage.CHUNKING,
        SyncStage.INDEXING,
        SyncStage.RECORDING,
        SyncStage.FINALIZING,
    ]
    file_progress = progress[2:-1]
    assert all(item.relative_path == "library/inbox/notes.md" for item in file_progress)
    assert all(item.file_index == 1 for item in file_progress)
    assert all(item.file_count == 1 for item in file_progress)
    assert progress[6].chunk_count == 1


def test_sync_progress_skips_expensive_stages_for_unchanged_files(
    tmp_path: Path,
) -> None:
    settings, _, _, synchronizer = make_synchronizer(tmp_path)
    source = settings.inbox_dir / "notes.md"
    source.write_text("unchanged", encoding="utf-8")
    synchronizer.sync_library()
    progress: list[SyncProgress] = []

    synchronizer.sync_library(on_progress=progress.append)

    assert [item.stage for item in progress] == [
        SyncStage.VERIFYING_INDEX,
        SyncStage.SCANNING,
        SyncStage.CHECKING,
        SyncStage.FINALIZING,
    ]


def test_birthday_change_replaces_all_stale_chunks(tmp_path: Path) -> None:
    settings, manifest, vectors, synchronizer = make_synchronizer(tmp_path)
    source = settings.inbox_dir / "birthday.md"
    source.write_text("The birthday is August 1.", encoding="utf-8")
    synchronizer.sync_library()
    old_ids = list(next(iter(vectors.chunk_ids.values())))

    rewrite_with_new_mtime(source, "The birthday is August 7.")
    result = synchronizer.sync_library()

    passages = [
        chunk.page_content
        for document_chunks in vectors.documents.values()
        for chunk in document_chunks
    ]
    assert result.changed == 1
    assert passages == ["The birthday is August 7."]
    assert "August 1" not in " ".join(passages)
    assert next(iter(vectors.chunk_ids.values())) != old_ids
    assert manifest.all_documents()[0].content_hash[:16] in next(
        iter(vectors.chunk_ids.values())
    )[0]


def test_deleted_file_removes_vectors_and_manifest_entry(tmp_path: Path) -> None:
    settings, manifest, vectors, synchronizer = make_synchronizer(tmp_path)
    source = settings.inbox_dir / "delete-me.txt"
    source.write_text("temporary", encoding="utf-8")
    synchronizer.sync_library()

    source.unlink()
    result = synchronizer.sync_library()

    assert result.deleted == 1
    assert manifest.all_documents() == []
    assert vectors.documents == {}


def test_sync_progress_reports_deleted_document_removal(tmp_path: Path) -> None:
    settings, _, _, synchronizer = make_synchronizer(tmp_path)
    source = settings.inbox_dir / "delete-me.txt"
    source.write_text("temporary", encoding="utf-8")
    synchronizer.sync_library()
    source.unlink()
    progress: list[SyncProgress] = []

    synchronizer.sync_library(on_progress=progress.append)

    removal = next(item for item in progress if item.stage is SyncStage.REMOVING)
    assert removal.relative_path == "library/inbox/delete-me.txt"


def test_rebuild_progress_starts_with_collection_reset(tmp_path: Path) -> None:
    settings, _, _, synchronizer = make_synchronizer(tmp_path)
    (settings.inbox_dir / "notes.md").write_text("hello", encoding="utf-8")
    progress: list[SyncProgress] = []

    synchronizer.rebuild_index(on_progress=progress.append)

    assert progress[0].stage is SyncStage.RESETTING
    assert progress[1].stage is SyncStage.VERIFYING_INDEX


def test_changed_document_failure_keeps_old_vectors_and_blocks_question(
    tmp_path: Path,
) -> None:
    settings, manifest, vectors, synchronizer = make_synchronizer(tmp_path)
    source = settings.inbox_dir / "fragile.txt"
    source.write_text("known good", encoding="utf-8")
    synchronizer.sync_library()
    old_passage = next(iter(vectors.documents.values()))[0].page_content

    rewrite_with_new_mtime(source, "broken")
    with pytest.raises(SynchronizationError):
        synchronizer.sync_library(raise_on_errors=True)

    assert next(iter(vectors.documents.values()))[0].page_content == old_passage
    assert manifest.all_documents()[0].status == "failed"


def test_incompatible_index_requires_explicit_rebuild(tmp_path: Path) -> None:
    settings, manifest, vectors, synchronizer = make_synchronizer(tmp_path)
    source = settings.inbox_dir / "notes.md"
    source.write_text("hello", encoding="utf-8")
    synchronizer.sync_library()

    incompatible = Settings(vault_path=settings.vault_path, embedding_dimensions=768)
    changed_sync = Synchronizer(
        incompatible,
        manifest,
        vectors,
        IdentityChunker(),
        loader=text_loader,
    )

    with pytest.raises(IndexConfigurationMismatch):
        changed_sync.sync_library()

    result = changed_sync.rebuild_index()
    assert result.added == 1
    assert manifest.get_index_metadata()["dimensions"] == "768"

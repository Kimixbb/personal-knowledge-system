from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from personal_rag.config import Settings
from personal_rag.loaders import load_document
from personal_rag.manifest import DocumentRecord, Manifest
from personal_rag.vault import (
    content_hash,
    iter_library_files,
    relative_source_path,
    stable_document_id,
)


class Chunker(Protocol):
    def split(self, documents: Sequence[Document]) -> list[Document]: ...


class MutableVectorStore(Protocol):
    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Document],
        chunk_ids: Sequence[str],
    ) -> None: ...

    def delete_document(self, document_id: str) -> None: ...

    def rebuild_collection(self) -> None: ...


class SyncStage(StrEnum):
    VERIFYING_INDEX = "verifying_index"
    SCANNING = "scanning"
    CHECKING = "checking"
    HASHING = "hashing"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    RECORDING = "recording"
    REMOVING = "removing"
    FINALIZING = "finalizing"
    RESETTING = "resetting"


@dataclass(frozen=True, slots=True)
class SyncProgress:
    stage: SyncStage
    relative_path: str | None = None
    file_index: int | None = None
    file_count: int | None = None
    chunk_count: int | None = None


SyncProgressCallback = Callable[[SyncProgress], None]


@dataclass(slots=True)
class SyncResult:
    added: int = 0
    changed: int = 0
    deleted: int = 0
    skipped: int = 0
    empty: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    completed_at: str | None = None

    @property
    def successful(self) -> bool:
        return not self.errors


class SynchronizationError(RuntimeError):
    def __init__(self, result: SyncResult) -> None:
        super().__init__("Library synchronization failed; answering is blocked")
        self.result = result


class IndexConfigurationMismatch(RuntimeError):
    """Raised before search when incompatible vectors may exist."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Synchronizer:
    def __init__(
        self,
        settings: Settings,
        manifest: Manifest,
        vector_store: MutableVectorStore,
        chunker: Chunker,
        *,
        loader: Callable[[Path, str], list[Document]] = load_document,
    ) -> None:
        self.settings = settings
        self.manifest = manifest
        self.vector_store = vector_store
        self.chunker = chunker
        self.loader = loader

    def _ensure_compatible_index(self) -> None:
        expected = self.settings.index_metadata
        actual = self.manifest.get_index_metadata()
        if not actual:
            count_method = getattr(self.vector_store, "count_chunks", None)
            if callable(count_method) and count_method() > 0:
                raise IndexConfigurationMismatch(
                    "The vector collection has no configuration metadata; rebuild it"
                )
            self.manifest.replace_index_metadata(expected)
            return
        mismatches = {
            key: (actual.get(key), expected_value)
            for key, expected_value in expected.items()
            if actual.get(key) != expected_value
        }
        if mismatches:
            names = ", ".join(sorted(mismatches))
            raise IndexConfigurationMismatch(
                f"Index configuration changed ({names}); rebuild the index"
            )

    def _failure_record(
        self,
        path: Path,
        relative_path: str,
        document_id: str,
        existing: DocumentRecord | None,
        error: Exception,
        known_hash: str,
    ) -> DocumentRecord:
        try:
            stat = path.stat()
            size = stat.st_size
            modified = stat.st_mtime_ns
        except OSError:
            size = existing.file_size if existing else 0
            modified = existing.modified_time_ns if existing else 0
        return DocumentRecord(
            document_id=document_id,
            relative_path=relative_path,
            extension=path.suffix.lower(),
            file_size=size,
            modified_time_ns=modified,
            content_hash=known_hash or (existing.content_hash if existing else ""),
            status="failed",
            chunk_count=existing.chunk_count if existing else 0,
            indexed_at=existing.indexed_at if existing else None,
            last_error=type(error).__name__,
        )

    def _process_file(
        self,
        path: Path,
        relative_path: str,
        existing: DocumentRecord | None,
        result: SyncResult,
        on_progress: SyncProgressCallback | None,
        file_index: int,
        file_count: int,
    ) -> None:
        document_id = stable_document_id(relative_path)
        known_hash = ""

        def report(stage: SyncStage, *, chunk_count: int | None = None) -> None:
            if on_progress is not None:
                on_progress(
                    SyncProgress(
                        stage=stage,
                        relative_path=relative_path,
                        file_index=file_index,
                        file_count=file_count,
                        chunk_count=chunk_count,
                    )
                )

        try:
            stat = path.stat()
            if (
                existing is not None
                and existing.status in {"indexed", "empty"}
                and existing.file_size == stat.st_size
                and existing.modified_time_ns == stat.st_mtime_ns
            ):
                result.skipped += 1
                return

            report(SyncStage.HASHING)
            known_hash = content_hash(path)
            if (
                existing is not None
                and existing.status in {"indexed", "empty"}
                and existing.content_hash == known_hash
            ):
                self.manifest.upsert(
                    DocumentRecord(
                        document_id=document_id,
                        relative_path=relative_path,
                        extension=path.suffix.lower(),
                        file_size=stat.st_size,
                        modified_time_ns=stat.st_mtime_ns,
                        content_hash=known_hash,
                        status=existing.status,
                        chunk_count=existing.chunk_count,
                        indexed_at=existing.indexed_at,
                        last_error=None,
                    )
                )
                result.skipped += 1
                return

            report(SyncStage.EXTRACTING)
            extracted = self.loader(path, relative_path)
            report(SyncStage.CHUNKING)
            chunks = self.chunker.split(extracted)
            chunk_ids: list[str] = []
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{document_id}:{known_hash[:16]}:{chunk_index}"
                chunk.metadata.update(
                    {
                        "document_id": document_id,
                        "relative_path": relative_path,
                        "file_type": path.suffix.lower().removeprefix("."),
                        "chunk_index": chunk_index,
                        "content_hash": known_hash,
                        "chunk_id": chunk_id,
                    }
                )
                chunk_ids.append(chunk_id)

            report(SyncStage.INDEXING, chunk_count=len(chunks))
            self.vector_store.replace_document(document_id, chunks, chunk_ids)
            status = "indexed" if chunks else "empty"
            indexed_at = _now()
            report(SyncStage.RECORDING, chunk_count=len(chunks))
            self.manifest.upsert(
                DocumentRecord(
                    document_id=document_id,
                    relative_path=relative_path,
                    extension=path.suffix.lower(),
                    file_size=stat.st_size,
                    modified_time_ns=stat.st_mtime_ns,
                    content_hash=known_hash,
                    status=status,
                    chunk_count=len(chunks),
                    indexed_at=indexed_at,
                    last_error=None,
                )
            )
            if existing is None:
                result.added += 1
            else:
                result.changed += 1
            if not chunks:
                result.empty += 1
        except Exception as exc:
            self.manifest.upsert(
                self._failure_record(
                    path,
                    relative_path,
                    document_id,
                    existing,
                    exc,
                    known_hash,
                )
            )
            result.failed += 1
            result.errors.append(f"{relative_path}: {type(exc).__name__}")

    def sync_library(
        self,
        *,
        raise_on_errors: bool = False,
        on_progress: SyncProgressCallback | None = None,
    ) -> SyncResult:
        if on_progress is not None:
            on_progress(SyncProgress(stage=SyncStage.VERIFYING_INDEX))
        self._ensure_compatible_index()
        result = SyncResult()
        existing_by_path = {
            record.relative_path: record for record in self.manifest.all_documents()
        }
        current_paths: set[str] = set()
        current_document_ids: set[str] = set()

        try:
            if on_progress is not None:
                on_progress(SyncProgress(stage=SyncStage.SCANNING))
            files = sorted(
                iter_library_files(self.settings.library_dir, self.settings.vault_path),
                key=lambda path: str(path).casefold(),
            )
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"library scan: {type(exc).__name__}")
            if raise_on_errors:
                raise SynchronizationError(result) from exc
            return result

        file_count = len(files)
        for file_index, path in enumerate(files, start=1):
            try:
                relative_path = relative_source_path(path, self.settings.vault_path)
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"unsafe source: {type(exc).__name__}")
                continue
            current_paths.add(relative_path)
            current_document_ids.add(stable_document_id(relative_path))
            if on_progress is not None:
                on_progress(
                    SyncProgress(
                        stage=SyncStage.CHECKING,
                        relative_path=relative_path,
                        file_index=file_index,
                        file_count=file_count,
                    )
                )
            self._process_file(
                path,
                relative_path,
                existing_by_path.get(relative_path),
                result,
                on_progress,
                file_index,
                file_count,
            )

        for relative_path, record in existing_by_path.items():
            if (
                relative_path in current_paths
                or record.document_id in current_document_ids
            ):
                continue
            try:
                if on_progress is not None:
                    on_progress(
                        SyncProgress(
                            stage=SyncStage.REMOVING,
                            relative_path=relative_path,
                        )
                    )
                self.vector_store.delete_document(record.document_id)
                self.manifest.delete(record.document_id)
                result.deleted += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{relative_path}: {type(exc).__name__}")

        if on_progress is not None:
            on_progress(SyncProgress(stage=SyncStage.FINALIZING))
        result.completed_at = _now()
        if result.successful:
            self.manifest.set_state("last_sync", result.completed_at)
        if result.errors and raise_on_errors:
            raise SynchronizationError(result)
        return result

    def rebuild_index(
        self, *, on_progress: SyncProgressCallback | None = None
    ) -> SyncResult:
        if on_progress is not None:
            on_progress(SyncProgress(stage=SyncStage.RESETTING))
        self.vector_store.rebuild_collection()
        self.manifest.clear_documents()
        self.manifest.replace_index_metadata(self.settings.index_metadata)
        return self.sync_library(raise_on_errors=False, on_progress=on_progress)

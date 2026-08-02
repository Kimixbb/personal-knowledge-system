from __future__ import annotations

from pathlib import PurePosixPath

from personal_rag.synchronizer import SyncProgress, SyncStage


def _file_prefix(progress: SyncProgress) -> tuple[str, str]:
    filename = (
        PurePosixPath(progress.relative_path).name if progress.relative_path else "file"
    )
    if progress.file_index is not None and progress.file_count is not None:
        return f"File {progress.file_index} of {progress.file_count}: ", filename
    return "", filename


def sync_progress_label(
    progress: SyncProgress,
    *,
    embedding_device: str,
    model_loaded: bool,
) -> str:
    """Describe the exact synchronous operation represented by a progress event."""

    prefix, filename = _file_prefix(progress)
    if progress.stage is SyncStage.VERIFYING_INDEX:
        return "Checking that the local index configuration is compatible…"
    if progress.stage is SyncStage.RESETTING:
        return "Removing existing vectors and resetting the local index…"
    if progress.stage is SyncStage.SCANNING:
        return "Scanning the vault for PDF, TXT, and Markdown files…"
    if progress.stage is SyncStage.CHECKING:
        return f"{prefix}checking {filename} for changes…"
    if progress.stage is SyncStage.HASHING:
        return f"{prefix}reading and fingerprinting {filename} on the CPU…"
    if progress.stage is SyncStage.EXTRACTING:
        return f"{prefix}extracting text from {filename} on the CPU…"
    if progress.stage is SyncStage.CHUNKING:
        return f"{prefix}tokenizing and splitting {filename} into chunks on the CPU…"
    if progress.stage is SyncStage.INDEXING:
        chunk_count = progress.chunk_count or 0
        if chunk_count == 0:
            return (
                f"{prefix}no extractable text found in {filename}; "
                "updating its index state…"
            )
        device = "the CUDA GPU" if embedding_device == "cuda" else "the CPU"
        if not model_loaded:
            return (
                f"{prefix}loading the embedding model onto {device}, embedding "
                f"{chunk_count} chunks from {filename}, then writing the vectors locally…"
            )
        return (
            f"{prefix}embedding {chunk_count} chunks from {filename} on {device}, "
            "then writing the vectors locally…"
        )
    if progress.stage is SyncStage.RECORDING:
        return f"{prefix}recording {filename} in the synchronization manifest…"
    if progress.stage is SyncStage.REMOVING:
        return f"Removing deleted document {filename} from the local index…"
    if progress.stage is SyncStage.FINALIZING:
        return "Saving the final synchronization state…"
    raise ValueError(f"Unsupported synchronization stage: {progress.stage}")

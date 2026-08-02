from __future__ import annotations

from personal_rag.synchronizer import SyncProgress, SyncStage
from personal_rag.ui_progress import sync_progress_label


def test_indexing_label_distinguishes_lazy_model_loading_from_gpu_work() -> None:
    progress = SyncProgress(
        stage=SyncStage.INDEXING,
        relative_path="library/inbox/book.pdf",
        file_index=2,
        file_count=5,
        chunk_count=605,
    )

    assert sync_progress_label(
        progress, embedding_device="cuda", model_loaded=False
    ) == (
        "File 2 of 5: loading the embedding model onto the CUDA GPU, embedding "
        "605 chunks from book.pdf, then writing the vectors locally…"
    )
    assert sync_progress_label(
        progress, embedding_device="cuda", model_loaded=True
    ) == (
        "File 2 of 5: embedding 605 chunks from book.pdf on the CUDA GPU, "
        "then writing the vectors locally…"
    )


def test_cpu_preparation_and_empty_document_labels_are_explicit() -> None:
    chunking = SyncProgress(
        stage=SyncStage.CHUNKING,
        relative_path="library/inbox/notes.md",
        file_index=1,
        file_count=1,
    )
    empty = SyncProgress(
        stage=SyncStage.INDEXING,
        relative_path="library/inbox/blank.pdf",
        file_index=1,
        file_count=1,
        chunk_count=0,
    )

    assert sync_progress_label(
        chunking, embedding_device="cuda", model_loaded=False
    ) == "File 1 of 1: tokenizing and splitting notes.md into chunks on the CPU…"
    assert sync_progress_label(
        empty, embedding_device="cuda", model_loaded=False
    ) == "File 1 of 1: no extractable text found in blank.pdf; updating its index state…"

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from personal_rag.vault import SUPPORTED_EXTENSIONS, is_within


class UploadError(RuntimeError):
    pass


class UnsupportedUploadError(UploadError):
    pass


class DuplicateFileError(UploadError):
    def __init__(self, filename: str) -> None:
        super().__init__(f"{filename} already exists")
        self.filename = filename


class UnsafePathError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ImportResult:
    filename: str
    destination: Path
    replaced: bool
    byte_count: int


def sanitize_upload_name(original_name: str) -> str:
    normalized = original_name.replace("\\", "/")
    filename = PurePosixPath(normalized).name
    filename = PureWindowsPath(filename).name
    if (
        not filename
        or filename in {".", ".."}
        or any(ord(character) < 32 for character in filename)
    ):
        raise UnsafePathError("Upload filename is invalid")
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedUploadError(
            f"Unsupported file extension: {Path(filename).suffix or '(none)'}"
        )
    return filename


def save_upload(
    data: bytes,
    original_name: str,
    inbox_dir: Path,
    temp_dir: Path,
    *,
    replace: bool = False,
) -> ImportResult:
    """Stage, verify, and atomically place an upload without implicit overwrite."""

    filename = sanitize_upload_name(original_name)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    destination = inbox_dir / filename
    if not is_within(destination, inbox_dir):
        raise UnsafePathError("Upload destination is outside the inbox")
    if destination.exists() and not replace:
        raise DuplicateFileError(filename)

    staged = temp_dir / f"upload-{uuid.uuid4().hex}.tmp"
    try:
        with staged.open("xb") as handle:
            written = handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if written != len(data) or staged.stat().st_size != len(data):
            raise UploadError("Upload staging verification failed")

        if replace:
            os.replace(staged, destination)
        else:
            try:
                os.link(staged, destination)
            except FileExistsError as exc:
                raise DuplicateFileError(filename) from exc
            staged.unlink()

        if destination.stat().st_size != len(data):
            raise UploadError("Final upload verification failed")
        return ImportResult(
            filename=filename,
            destination=destination,
            replaced=replace,
            byte_count=len(data),
        )
    finally:
        if staged.exists():
            staged.unlink()


def _resolve_library_file(relative_path: str, vault_root: Path) -> Path:
    normalized = relative_path.replace("\\", "/")
    windows = PureWindowsPath(relative_path)
    posix = PurePosixPath(normalized)
    if windows.is_absolute() or posix.is_absolute() or ".." in posix.parts:
        raise UnsafePathError("File path must be relative to the vault")
    target = (vault_root / Path(*posix.parts)).resolve(strict=True)
    library = (vault_root / "library").resolve(strict=True)
    if not target.is_file() or not is_within(target, library):
        raise UnsafePathError("File is outside the vault library")
    return target


def open_vault_file(
    relative_path: str,
    vault_root: Path,
    *,
    opener: Callable[[Path], None] | None = None,
) -> Path:
    target = _resolve_library_file(relative_path, vault_root)
    if opener is None:
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise OSError("Opening files is supported only on Windows")
        opener = startfile
    opener(target)
    return target

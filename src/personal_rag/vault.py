from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Iterator


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def relative_source_path(path: Path, vault_root: Path) -> str:
    resolved = path.resolve(strict=True)
    root = vault_root.resolve(strict=True)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Source path is outside the configured vault") from exc


def stable_document_id(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_hash(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _is_hidden(path: Path) -> bool:
    if path.name.startswith("."):
        return True
    try:
        attributes = getattr(path.stat(), "st_file_attributes", 0)
        hidden_flag = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0)
        return bool(attributes & hidden_flag)
    except OSError:
        return False


def iter_library_files(library_dir: Path, vault_root: Path) -> Iterator[Path]:
    """Yield safe supported files without following symlinks or junction escapes."""

    library_real = library_dir.resolve(strict=True)
    vault_real = vault_root.resolve(strict=True)
    if not is_within(library_real, vault_real):
        raise ValueError("Library directory must be inside the vault")

    for current, directories, filenames in os.walk(library_real, followlinks=False):
        current_path = Path(current)
        safe_directories: list[str] = []
        for name in directories:
            candidate = current_path / name
            if _is_hidden(candidate) or candidate.is_symlink():
                continue
            if not is_within(candidate, library_real):
                continue
            safe_directories.append(name)
        directories[:] = safe_directories

        for name in filenames:
            candidate = current_path / name
            if _is_hidden(candidate) or candidate.is_symlink() or not is_supported(candidate):
                continue
            if is_within(candidate, library_real):
                yield candidate

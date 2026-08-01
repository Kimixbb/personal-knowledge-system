from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    relative_path: str
    extension: str
    file_size: int
    modified_time_ns: int
    content_hash: str
    status: str
    chunk_count: int
    indexed_at: str | None
    last_error: str | None


class Manifest:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL UNIQUE,
                    extension TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    modified_time_ns INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('indexed', 'empty', 'failed')),
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    indexed_at TEXT,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row | None) -> DocumentRecord | None:
        return DocumentRecord(**dict(row)) if row is not None else None

    def get_by_path(self, relative_path: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE relative_path = ?", (relative_path,)
            ).fetchone()
        return self._record(row)

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._record(row)

    def all_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY relative_path"
            ).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def upsert(self, record: DocumentRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, relative_path, extension, file_size,
                    modified_time_ns, content_hash, status, chunk_count,
                    indexed_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    extension=excluded.extension,
                    file_size=excluded.file_size,
                    modified_time_ns=excluded.modified_time_ns,
                    content_hash=excluded.content_hash,
                    status=excluded.status,
                    chunk_count=excluded.chunk_count,
                    indexed_at=excluded.indexed_at,
                    last_error=excluded.last_error
                """,
                (
                    record.document_id,
                    record.relative_path,
                    record.extension,
                    record.file_size,
                    record.modified_time_ns,
                    record.content_hash,
                    record.status,
                    record.chunk_count,
                    record.indexed_at,
                    record.last_error,
                ),
            )

    def delete(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )

    def clear_documents(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")

    def count_documents(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM documents WHERE status IN ('indexed', 'empty')"
            ).fetchone()
        return int(row["count"])

    def count_chunks(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) AS count FROM documents WHERE status = 'indexed'"
            ).fetchone()
        return int(row["count"])

    def get_index_metadata(self) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM index_metadata").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def replace_index_metadata(self, values: Mapping[str, str]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM index_metadata")
            connection.executemany(
                "INSERT INTO index_metadata(key, value) VALUES (?, ?)", values.items()
            )

    def get_state(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_state WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else str(row["value"])

    def set_state(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_state(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )

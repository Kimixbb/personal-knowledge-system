from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """Raised when local configuration is missing or unsafe."""


DEFAULT_EMBEDDING_REVISION = "d43997c8a1046d1734f8d519effbb424a832a0a2"


def _as_positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _as_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    vault_path: Path
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_revision: str = DEFAULT_EMBEDDING_REVISION
    embedding_dimensions: int = 1024
    embedding_device: str = "cuda"
    embedding_batch_size: int = 16
    default_llm_provider: str = "deepseek"
    default_llm_model: str = "deepseek-chat"
    deepseek_api_key: str = field(default="", repr=False)
    moonshot_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)
    openai_compatible_api_key: str = field(default="", repr=False)
    openai_compatible_base_url: str = ""
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.2
    chunk_size: int = 700
    chunk_overlap: int = 100
    collection_name: str = "personal_rag_qwen3_06b_v1"
    chunking_version: str = "multilingual_qwen_tokens_v1"

    @classmethod
    def from_env(
        cls,
        env_file: Path | str = ".env",
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        file_values = {
            key: value
            for key, value in dotenv_values(env_file).items()
            if value is not None
        }
        merged = {**file_values, **dict(os.environ if environ is None else environ)}
        raw_vault = merged.get("KNOWLEDGE_VAULT", "").strip()
        if not raw_vault:
            raise ConfigurationError("KNOWLEDGE_VAULT is required in .env")

        settings = cls(
            vault_path=Path(raw_vault).expanduser().resolve(),
            embedding_model=merged.get(
                "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
            ).strip(),
            embedding_revision=merged.get(
                "EMBEDDING_REVISION", DEFAULT_EMBEDDING_REVISION
            ).strip(),
            embedding_dimensions=_as_positive_int(
                merged, "EMBEDDING_DIMENSIONS", 1024
            ),
            embedding_device=merged.get("EMBEDDING_DEVICE", "cuda").strip().lower(),
            embedding_batch_size=_as_positive_int(
                merged, "EMBEDDING_BATCH_SIZE", 16
            ),
            default_llm_provider=merged.get(
                "DEFAULT_LLM_PROVIDER", "deepseek"
            ).strip().lower(),
            default_llm_model=merged.get(
                "DEFAULT_LLM_MODEL", "deepseek-chat"
            ).strip(),
            deepseek_api_key=merged.get("DEEPSEEK_API_KEY", "").strip(),
            moonshot_api_key=merged.get("MOONSHOT_API_KEY", "").strip(),
            openai_api_key=merged.get("OPENAI_API_KEY", "").strip(),
            openai_compatible_api_key=merged.get(
                "OPENAI_COMPATIBLE_API_KEY", ""
            ).strip(),
            openai_compatible_base_url=merged.get(
                "OPENAI_COMPATIBLE_BASE_URL", ""
            ).strip(),
            retrieval_top_k=_as_positive_int(merged, "RETRIEVAL_TOP_K", 5),
            retrieval_min_score=_as_float(merged, "RETRIEVAL_MIN_SCORE", 0.2),
            chunk_size=_as_positive_int(merged, "CHUNK_SIZE", 700),
            chunk_overlap=_as_positive_int(merged, "CHUNK_OVERLAP", 100),
        )
        settings.validate()
        repository_root = Path(env_file).resolve(strict=False).parent
        try:
            settings.vault_path.relative_to(repository_root)
        except ValueError:
            pass
        else:
            raise ConfigurationError(
                "KNOWLEDGE_VAULT must be outside the source-code repository"
            )
        return settings

    def validate(self) -> None:
        if self.embedding_device not in {"cuda", "cpu"}:
            raise ConfigurationError("EMBEDDING_DEVICE must be 'cuda' or 'cpu'")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not -1.0 <= self.retrieval_min_score <= 1.0:
            raise ConfigurationError("RETRIEVAL_MIN_SCORE must be between -1 and 1")
        if self.default_llm_provider not in {
            "deepseek",
            "kimi",
            "openai",
            "openai-compatible",
        }:
            raise ConfigurationError("DEFAULT_LLM_PROVIDER is not supported")
        if not self.embedding_model:
            raise ConfigurationError("EMBEDDING_MODEL cannot be empty")

    @property
    def library_dir(self) -> Path:
        return self.vault_path / "library"

    @property
    def inbox_dir(self) -> Path:
        return self.library_dir / "inbox"

    @property
    def rag_dir(self) -> Path:
        return self.vault_path / ".rag"

    @property
    def chroma_dir(self) -> Path:
        return self.rag_dir / "chroma"

    @property
    def manifest_path(self) -> Path:
        return self.rag_dir / "manifest.sqlite3"

    @property
    def temp_dir(self) -> Path:
        return self.rag_dir / "temp"

    @property
    def logs_dir(self) -> Path:
        return self.rag_dir / "logs"

    @property
    def index_metadata(self) -> dict[str, str]:
        return {
            "model_name": self.embedding_model,
            "model_revision": self.embedding_revision,
            "dimensions": str(self.embedding_dimensions),
            "chunking_version": self.chunking_version,
            "chunk_size": str(self.chunk_size),
            "chunk_overlap": str(self.chunk_overlap),
            "collection_name": self.collection_name,
        }


def ensure_vault(settings: Settings) -> None:
    """Create only the documented application and library directories."""

    for directory in (
        settings.inbox_dir,
        settings.chroma_dir,
        settings.temp_dir,
        settings.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

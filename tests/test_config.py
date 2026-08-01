from __future__ import annotations

from pathlib import Path

import pytest

from personal_rag.config import ConfigurationError, Settings


def test_env_loader_rejects_vault_inside_repository(tmp_path: Path) -> None:
    env_file = tmp_path / "repo" / ".env"
    env_file.parent.mkdir()

    with pytest.raises(ConfigurationError, match="outside"):
        Settings.from_env(
            env_file,
            environ={"KNOWLEDGE_VAULT": str(env_file.parent / "vault")},
        )


def test_env_loader_accepts_external_vault_and_hides_keys_from_repr(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    settings = Settings.from_env(
        repository / ".env",
        environ={
            "KNOWLEDGE_VAULT": str(tmp_path / "external-vault"),
            "DEEPSEEK_API_KEY": "do-not-display",
        },
    )

    assert settings.vault_path == (tmp_path / "external-vault").resolve()
    assert "do-not-display" not in repr(settings)

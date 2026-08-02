from __future__ import annotations

import pytest

from personal_rag.config import Settings
from personal_rag.providers import (
    ProviderConfigurationError,
    available_providers,
    create_chat_model,
    normalize_response_text,
)


class FakeChatModel:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def settings_with_keys(tmp_path, **overrides):  # type: ignore[no-untyped-def]
    values = {
        "vault_path": tmp_path / "vault",
        "deepseek_api_key": "deep-key",
        "moonshot_api_key": "moon-key",
        "openai_api_key": "open-key",
        "openai_compatible_api_key": "custom-key",
        "openai_compatible_base_url": "https://llm.example/v1",
    }
    values.update(overrides)
    return Settings(**values)


def test_only_fully_configured_providers_are_available(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        vault_path=tmp_path / "vault",
        deepseek_api_key="deep-key",
        openai_compatible_api_key="custom-key",
    )

    availability = {item.provider: item.configured for item in available_providers(settings)}

    assert availability == {
        "deepseek": True,
        "kimi": False,
        "openai": False,
        "openai-compatible": False,
    }


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("deepseek", "deepseek-chat", {"temperature": 0}),
        ("kimi", "kimi-k2.5", {"thinking": False}),
        ("openai", "gpt-4.1-mini", {"temperature": 0}),
        (
            "openai-compatible",
            "custom-chat",
            {"temperature": 0, "base_url": "https://llm.example/v1"},
        ),
    ],
)
def test_provider_specific_options_stay_inside_factory(
    tmp_path, provider: str, model: str, expected: dict[str, object]  # type: ignore[no-untyped-def]
) -> None:
    resolver = lambda _: FakeChatModel

    chat = create_chat_model(
        provider,
        model,
        settings_with_keys(tmp_path),
        class_resolver=resolver,
    )

    assert chat.kwargs["model"] == model
    for key, value in expected.items():
        assert chat.kwargs[key] == value


@pytest.mark.parametrize(
    ("model", "expected", "absent"),
    [
        ("kimi-k2.5", {"thinking": False}, {"temperature"}),
        ("kimi-k2.6", {"thinking": False}, {"temperature"}),
        ("kimi-k3", {}, {"thinking", "temperature"}),
    ],
)
def test_kimi_request_options_follow_model_family(
    tmp_path,  # type: ignore[no-untyped-def]
    model: str,
    expected: dict[str, object],
    absent: set[str],
) -> None:
    chat = create_chat_model(
        "kimi",
        model,
        settings_with_keys(tmp_path),
        class_resolver=lambda _: FakeChatModel,
    )

    for key, value in expected.items():
        assert chat.kwargs[key] == value
    assert absent.isdisjoint(chat.kwargs)


def test_missing_key_is_rejected_before_importing_provider(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ProviderConfigurationError):
        create_chat_model(
            "deepseek",
            "deepseek-chat",
            Settings(vault_path=tmp_path / "vault"),
            class_resolver=lambda _: FakeChatModel,
        )


def test_normalizes_text_and_content_blocks() -> None:
    assert normalize_response_text(type("R", (), {"content": "answer"})()) == "answer"
    response = type(
        "R",
        (),
        {"content": [{"type": "text", "text": "first"}, {"text": "second"}]},
    )()
    assert normalize_response_text(response) == "first\nsecond"


def test_real_provider_integrations_construct_without_calling_network(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = settings_with_keys(tmp_path)

    models = [
        create_chat_model("deepseek", "deepseek-chat", settings),
        create_chat_model("kimi", "kimi-k2.5", settings),
        create_chat_model("kimi", "kimi-k2.6", settings),
        create_chat_model("kimi", "kimi-k3", settings),
        create_chat_model("openai", "gpt-4.1-mini", settings),
        create_chat_model("openai-compatible", "custom-chat", settings),
    ]

    assert [type(model).__name__ for model in models] == [
        "ChatDeepSeek",
        "ChatMoonshot",
        "ChatMoonshot",
        "ChatMoonshot",
        "ChatOpenAI",
        "ChatOpenAI",
    ]
    assert models[1].thinking is False
    assert models[1].temperature is None
    assert models[2].thinking is False
    assert models[2].temperature is None
    assert models[3].thinking is None
    assert models[3].temperature is None

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from personal_rag.config import Settings


PROVIDERS = ("deepseek", "kimi", "openai", "openai-compatible")


@dataclass(frozen=True, slots=True)
class ProviderAvailability:
    provider: str
    configured: bool
    reason: str


class ProviderConfigurationError(ValueError):
    pass


def available_providers(settings: Settings) -> list[ProviderAvailability]:
    states = {
        "deepseek": (
            bool(settings.deepseek_api_key),
            "DEEPSEEK_API_KEY is not configured",
        ),
        "kimi": (
            bool(settings.moonshot_api_key),
            "MOONSHOT_API_KEY is not configured",
        ),
        "openai": (
            bool(settings.openai_api_key),
            "OPENAI_API_KEY is not configured",
        ),
        "openai-compatible": (
            bool(
                settings.openai_compatible_api_key
                and settings.openai_compatible_base_url
            ),
            "OPENAI_COMPATIBLE_API_KEY and OPENAI_COMPATIBLE_BASE_URL are required",
        ),
    }
    return [
        ProviderAvailability(
            provider=provider,
            configured=states[provider][0],
            reason="" if states[provider][0] else states[provider][1],
        )
        for provider in PROVIDERS
    ]


def _resolve_chat_class(provider: str):  # type: ignore[no-untyped-def]
    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek
    if provider == "kimi":
        from langchain_moonshot import ChatMoonshot

        return ChatMoonshot
    if provider in {"openai", "openai-compatible"}:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI
    raise ProviderConfigurationError(f"Unsupported provider: {provider}")


def create_chat_model(
    provider: str,
    model: str,
    settings: Settings,
    *,
    class_resolver: Callable[[str], Any] = _resolve_chat_class,
):  # type: ignore[no-untyped-def]
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ProviderConfigurationError(f"Unsupported provider: {provider}")
    if not model.strip():
        raise ProviderConfigurationError("Model name cannot be empty")
    availability = {item.provider: item for item in available_providers(settings)}
    if not availability[provider].configured:
        raise ProviderConfigurationError(availability[provider].reason)

    chat_class = class_resolver(provider)
    model_name = model.strip()
    common: dict[str, Any] = {"model": model_name, "max_retries": 2}
    if provider == "deepseek":
        return chat_class(
            **common,
            api_key=settings.deepseek_api_key,
            temperature=0,
        )
    if provider == "kimi":
        kimi_options: dict[str, Any] = {}
        if model_name.lower().startswith(("kimi-k2.5", "kimi-k2.6")):
            kimi_options["thinking"] = False
        return chat_class(
            **common,
            **kimi_options,
            api_key=settings.moonshot_api_key,
        )
    if provider == "openai":
        return chat_class(
            **common,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    return chat_class(
        **common,
        api_key=settings.openai_compatible_api_key,
        base_url=settings.openai_compatible_base_url,
        temperature=0,
    )


def normalize_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()

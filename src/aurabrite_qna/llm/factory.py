"""Create the right LLM client from Settings.

All non-mock providers are optional. If the SDK is not installed or the API
key is empty, we degrade gracefully to the mock client instead of crashing —
this makes the system usable in air-gapped CI and demos.
"""

from __future__ import annotations

import logging

from ..config import SETTINGS, Settings
from .base import LLMClient
from .mock import MockLLMClient

log = logging.getLogger(__name__)


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or SETTINGS
    provider = settings.llm_provider

    if provider == "mock":
        return MockLLMClient(model=settings.llm_model)

    if provider == "litellm":
        client = _try_litellm(settings)
        if client:
            return client

    if provider == "openai":
        client = _try_openai(settings)
        if client:
            return client

    if provider == "anthropic":
        client = _try_anthropic(settings)
        if client:
            return client

    log.warning(
        "Requested LLM provider %r is unavailable — falling back to MockLLMClient.",
        provider,
    )
    return MockLLMClient(model=settings.llm_model)


def _try_litellm(settings: Settings) -> LLMClient | None:
    try:
        import litellm  # type: ignore
    except ImportError:
        return None
    from .providers import LiteLLMClient

    return LiteLLMClient(model=settings.llm_model, temperature=settings.llm_temperature)


def _try_openai(settings: Settings) -> LLMClient | None:
    if not settings.openai_api_key:
        return None
    try:
        import openai  # type: ignore  # noqa: F401
    except ImportError:
        return None
    from .providers import OpenAIClient

    return OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def _try_anthropic(settings: Settings) -> LLMClient | None:
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic  # type: ignore  # noqa: F401
    except ImportError:
        return None
    from .providers import AnthropicClient

    return AnthropicClient(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )

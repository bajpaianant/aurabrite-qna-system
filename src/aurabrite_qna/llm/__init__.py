"""LLM abstraction layer.

Every agent talks to LLMs through the tiny `LLMClient` interface defined
here. Concrete providers:

* ``MockLLMClient``    — deterministic, offline, keyword-driven. Used for
  evaluation and CI. It is *not* a random generator: it inspects the prompt
  and returns purposeful JSON so the supervisor/worker graph exercises real
  code paths.
* ``LiteLLMClient``    — thin wrapper around `litellm.completion` for any
  hosted model (OpenAI, Anthropic, Bedrock, Groq, etc.).
* ``OpenAIClient``     — direct OpenAI SDK client (falls back gracefully).
* ``AnthropicClient``  — direct Anthropic SDK client (falls back gracefully).

Selection is driven by ``SETTINGS.llm_provider``.
"""

from .base import ChatMessage, LLMClient, LLMError, LLMResponse, coerce_dict, extract_json
from .factory import build_llm_client
from .mock import MockLLMClient

__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMResponse",
    "LLMError",
    "MockLLMClient",
    "build_llm_client",
    "coerce_dict",
    "extract_json",
]

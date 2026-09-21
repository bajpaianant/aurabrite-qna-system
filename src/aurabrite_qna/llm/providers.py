"""Real LLM providers (all optional).

Each class implements the ``LLMClient`` protocol from ``base.py``. Nothing
in the rest of the codebase imports this module directly — the factory
does the wiring.
"""

from __future__ import annotations

import time

from .base import ChatMessage, LLMError, LLMResponse


class LiteLLMClient:
    provider = "litellm"

    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        import litellm  # type: ignore

        t0 = time.perf_counter()
        try:
            resp = litellm.completion(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                stop=stop,
                # LiteLLM auto-retries transient 5xx / 429 with exponential
                # backoff. Gemini in particular emits 503s from its alpha
                # channels for a few seconds under load.
                num_retries=3,
            )
        except Exception as e:  # pragma: no cover
            # Rewrap common failure modes with actionable hints so users
            # do not have to dig through a 40-line vendor traceback.
            msg = str(e)
            hint = ""
            low = msg.lower()
            if "not found" in low and "gemini" in low:
                hint = (
                    "\n\nHint: Google's `generativelanguage.googleapis.com` "
                    "endpoint returned 404 for this model. Common causes:\n"
                    "  1. The model name is not in your API tier. Try "
                    "`gemini/gemini-2.0-flash-exp` (free-tier friendly) "
                    "or `gemini/gemini-1.5-flash-002`.\n"
                    "  2. Your `GEMINI_API_KEY` is an OAuth token / "
                    "Vertex credential (starts with `AQ.` or `ya29.`), "
                    "not a Google AI Studio API key. Generate a proper "
                    "one at https://aistudio.google.com/apikey — it "
                    "starts with `AIza...`."
                )
            elif "unauthenticated" in low or "api key not valid" in low:
                hint = (
                    "\n\nHint: Your GEMINI_API_KEY was rejected. Confirm "
                    "you copied a Google AI Studio key (starts with "
                    "`AIza...`) from https://aistudio.google.com/apikey."
                )
            elif "quota" in low or "rate limit" in low or "429" in msg:
                hint = (
                    "\n\nHint: You hit a rate limit or quota. Slow down "
                    "requests, or upgrade the key at "
                    "https://aistudio.google.com/apikey."
                )
            raise LLMError(f"LiteLLM call failed: {e}{hint}") from e

        text = resp["choices"][0]["message"]["content"]
        usage = getattr(resp, "usage", {}) or {}
        return LLMResponse(
            text=text or "",
            provider=self.provider,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            raw=resp if isinstance(resp, dict) else {},
        )


class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key: str, model: str, temperature: float = 0.1):
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        except Exception as e:  # pragma: no cover
            raise LLMError(f"OpenAI call failed: {e}") from e
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            provider=self.provider,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_in=getattr(usage, "prompt_tokens", 0),
            tokens_out=getattr(usage, "completion_tokens", 0),
        )


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str, model: str, temperature: float = 0.1):
        import anthropic  # type: ignore

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        # Anthropic uses a system parameter + user/assistant messages.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        try:
            resp = self._client.messages.create(
                model=self.model,
                system=system or None,
                messages=chat,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or 1024,
                stop_sequences=stop,
            )
        except Exception as e:  # pragma: no cover
            raise LLMError(f"Anthropic call failed: {e}") from e
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
        )

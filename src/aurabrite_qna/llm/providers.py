"""Real LLM providers (all optional).

Each class implements the ``LLMClient`` protocol from ``base.py``. Nothing
in the rest of the codebase imports this module directly — the factory
does the wiring.
"""

from __future__ import annotations

import time

from .base import ChatMessage, LLMError, LLMResponse


def _force_gemini_studio_v1beta() -> None:
    """LiteLLM 1.102 routes any ``gemini-3*`` model to Google's ``v1alpha`` API.

    AI Studio ``generateContent`` for current Flash/Pro models lives on
    ``v1beta``. ``v1alpha`` returns 503 for those same model ids, which is
    exactly what we hit in production with ``gemini-3.6-flash``. Patch the
    URL builder once per process so Gemini 3.x calls go to ``v1beta``.
    """
    try:
        from litellm.llms.vertex_ai import vertex_llm_base as _base  # type: ignore
    except Exception:  # pragma: no cover
        return
    orig = getattr(_base, "_get_gemini_url", None)
    if orig is None or getattr(orig, "_aurabrite_v1beta", False):
        return

    def _wrapped(mode, model, stream=None):  # noqa: ANN001
        url, endpoint = orig(mode, model, stream)
        if isinstance(url, str) and "/v1alpha/" in url:
            url = url.replace("/v1alpha/", "/v1beta/")
        return url, endpoint

    _wrapped._aurabrite_v1beta = True  # type: ignore[attr-defined]
    _base._get_gemini_url = _wrapped


class LiteLLMClient:
    provider = "litellm"

    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        if "gemini" in (model or "").lower():
            _force_gemini_studio_v1beta()

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
                    "\n\nHint: Google returned 404 for this model id. "
                    "`ListModels` can still list a model that "
                    "`generateContent` has retired (e.g. gemini-2.5-flash "
                    "for newer AI Studio keys). Set LLM_MODEL to one of:\n"
                    "  gemini/gemini-3.5-flash\n"
                    "  gemini/gemini-flash-latest\n"
                    "  gemini/gemini-3.6-flash\n"
                    "then reboot the app."
                )
            elif "unauthenticated" in low or "api key not valid" in low:
                hint = (
                    "\n\nHint: GEMINI_API_KEY was rejected. Generate a "
                    "key at https://aistudio.google.com/apikey (newer "
                    "keys start with `AQ.`; older ones with `AIza...`)."
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

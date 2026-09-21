"""Core LLM interface + shared helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


class LLMError(RuntimeError):
    """Raised when an LLM call fails at the provider boundary."""


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Minimal LLM contract used by all agents."""

    provider: str
    model: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# JSON extraction helpers — every agent needs this
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Best-effort JSON parser.

    Strategy:
        1. Parse the whole thing.
        2. Parse the contents of any fenced ```json``` block.
        3. Balance-scan for the widest ``{...}`` or ``[...]`` and try each
           candidate from largest → smallest until one parses.

    Raises ``ValueError`` if nothing parses.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM response — nothing to parse.")

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for match in _JSON_BLOCK.finditer(text):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

    for candidate in _balanced_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not extract JSON from LLM response.")


def _balanced_candidates(text: str) -> list[str]:
    """Return balanced `{...}` and `[...]` substrings, widest first."""
    out: list[tuple[int, str]] = []  # (length, text)
    for opener, closer in (("{", "}"), ("[", "]")):
        stack: list[int] = []
        for i, ch in enumerate(text):
            if ch == opener:
                stack.append(i)
            elif ch == closer and stack:
                start = stack.pop()
                out.append((i - start, text[start : i + 1]))
    out.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in out]


def coerce_dict(text: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a dict from a possibly-messy LLM response, without raising."""
    defaults = defaults or {}
    try:
        obj = extract_json(text)
        if isinstance(obj, dict):
            return {**defaults, **obj}
    except Exception:
        pass
    return dict(defaults)


def messages_to_text(messages: Iterable[ChatMessage]) -> str:
    return "\n\n".join(f"[{m.role.upper()}]\n{m.content}" for m in messages)

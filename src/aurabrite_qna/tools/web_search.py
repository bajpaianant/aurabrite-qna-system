"""Web search tool with graceful offline fallback.

Providers (in order of preference):
    1. Tavily API (if ``TAVILY_API_KEY`` set and ``httpx`` installed)
    2. duckduckgo-search python package
    3. An offline stub that returns curated snippets so the demo works
       without internet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from ..config import SETTINGS

log = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str


class WebSearchTool:
    def __init__(self, provider: str | None = None):
        self.provider = provider or SETTINGS.web_search_provider

    def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        try:
            if self.provider == "tavily" and SETTINGS.tavily_api_key:
                return _tavily(query, top_k, SETTINGS.tavily_api_key)
            if self.provider == "duckduckgo":
                return _duckduckgo(query, top_k)
        except Exception as e:
            log.warning("Web search failed (%s: %s) — using offline stub.", self.provider, e)
        return _offline(query, top_k)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _duckduckgo(query: str, top_k: int) -> list[WebSearchResult]:
    from duckduckgo_search import DDGS  # type: ignore

    out: list[WebSearchResult] = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=top_k):
            out.append(
                WebSearchResult(
                    title=hit.get("title", "")[:200],
                    url=hit.get("href", ""),
                    snippet=hit.get("body", "")[:400],
                    source="duckduckgo",
                )
            )
    return out


def _tavily(query: str, top_k: int, api_key: str) -> list[WebSearchResult]:
    import httpx  # type: ignore

    r = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": top_k},
        timeout=15.0,
    )
    r.raise_for_status()
    data = r.json()
    return [
        WebSearchResult(
            title=hit.get("title", ""),
            url=hit.get("url", ""),
            snippet=hit.get("content", "")[:400],
            source="tavily",
        )
        for hit in data.get("results", [])
    ]


_OFFLINE_LIBRARY: dict[str, list[WebSearchResult]] = {
    "default": [
        WebSearchResult(
            title="Global FMCG Outlook 2026 — NielsenIQ",
            url="https://example.com/fmcg-outlook-2026",
            snippet=(
                "The FMCG industry is expected to see 3.4% value growth in 2026 with "
                "clean-label and premium-mass segments leading the acceleration."
            ),
            source="offline-stub",
        ),
        WebSearchResult(
            title="Skincare category grows 7% in APAC — Euromonitor",
            url="https://example.com/apac-skincare",
            snippet=(
                "Niacinamide-based skincare grew 12% in APAC in 2025, but sporadic "
                "supply of active ingredients constrained several premium brands."
            ),
            source="offline-stub",
        ),
    ],
    "singapore": [
        WebSearchResult(
            title="Port of Singapore congestion — Reuters archive",
            url="https://example.com/sgp-port-2025",
            snippet=(
                "Industrial action at the Port of Singapore disrupted APAC shipping "
                "throughout October and November 2025, with FMCG among the worst hit."
            ),
            source="offline-stub",
        )
    ],
    "clean label": [
        WebSearchResult(
            title="Clean-label surge in North America — Mintel",
            url="https://example.com/clean-label-na",
            snippet=(
                "Clean-label positioning is now cited as a top-3 purchase driver by "
                "68% of NA consumers in the health-foods aisle (Mintel, Q1 2026)."
            ),
            source="offline-stub",
        )
    ],
}


def _offline(query: str, top_k: int) -> list[WebSearchResult]:
    q = query.lower()
    hits: list[WebSearchResult] = []
    for keyword, entries in _OFFLINE_LIBRARY.items():
        if keyword != "default" and keyword in q:
            hits.extend(entries)
    if not hits:
        hits = list(_OFFLINE_LIBRARY["default"])
    return hits[:top_k]


def format_web_results(results: Iterable[WebSearchResult]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}\n    {r.url}\n    {r.snippet}")
    return "\n".join(lines) if lines else "(no web results)"

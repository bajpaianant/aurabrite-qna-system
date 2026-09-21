"""Deterministic, offline mock LLM.

The mock inspects the *system prompt tag* (which each agent embeds — e.g.
``[[AGENT: supervisor]]``) and returns purposefully-crafted JSON so the
end-to-end graph works without any API keys.

This is not a party trick: it's how we make the evaluation suite reproducible
and how we let contributors demo the system on a plane. Its outputs are
intentionally simple; the *data* is what makes the demo interesting.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .base import ChatMessage, LLMResponse, coerce_dict, messages_to_text

# ---------------------------------------------------------------------------
# Keyword → routing intelligence
# ---------------------------------------------------------------------------

_BRAND_HINTS = {
    "auraglow": "BR-AGL",
    "dentafresh": "BR-DEN",
    "cleanx": "BR-CLX",
    "nutrivita": "BR-NVT",
}

_REGION_HINTS = {
    "apac": "REG-APAC",
    "asia": "REG-APAC",
    "emea": "REG-EMEA",
    "europe": "REG-EMEA",
    "north america": "REG-NA",
    "north-america": "REG-NA",
    "na ": "REG-NA",
    "latam": "REG-LATAM",
}

_COUNTRY_HINTS = {
    "united kingdom": "MKT-UK", "uk": "MKT-UK",
    "germany": "MKT-DE", "india": "MKT-IN", "singapore": "MKT-SG",
    "united states": "MKT-US", "us ": "MKT-US", "usa": "MKT-US",
    "brazil": "MKT-BR",
}

_CHANNEL_HINTS = {
    "modern trade": "CH-MT", "walmart": "CH-MT", "tesco": "CH-MT",
    "traditional": "CH-GT", "general trade": "CH-GT",
    "e-commerce": "CH-EC", "ecommerce": "CH-EC", "amazon": "CH-EC",
    "quick-commerce": "CH-QC", "quick commerce": "CH-QC",
    "blinkit": "CH-QC", "getir": "CH-QC",
}


def _lc(t: str) -> str:
    return t.lower()


def _mentions_document_topics(t: str) -> bool:
    t = _lc(t)
    return any(
        k in t
        for k in [
            "strategy",
            "campaign",
            "sentiment",
            "insight",
            "clean-label",
            "clean label",
            "partnership",
            "rebate",
            "jbp",
            "supply",
            "disruption",
            "strike",
            "shortage",
            "niacinamide",
            "palm oil",
            "port of singapore",
            "why",
            "explain",
            "root cause",
            "consumer",
            "glossary",
            "definition",
            "stand for",
            "kpi",
            "spend",
            "spending",
            "working media",
            "media investment",
            "brand health",
        ]
    )


def _mentions_numeric_topics(t: str) -> bool:
    t = _lc(t)
    return any(
        k in t
        for k in [
            "revenue",
            "sales",
            "units",
            "volume",
            "gross margin",
            "margin",
            "discount",
            "dio",
            "inventory",
            "out of stock",
            "stockout",
            "top ",
            "growth",
            "share",
            "kpi",
            "how much",
            "how many",
            "trend",
            "yoy",
            "q1",
            "q2",
            "q3",
            "q4",
            "2024",
            "2025",
            "2026",
        ]
    )


def _mentions_web(t: str) -> bool:
    t = _lc(t)
    return any(
        k in t for k in ["news", "latest", "current", "market outlook", "competitor"]
    )


def _mentions_compute(t: str) -> bool:
    t = _lc(t)
    return any(
        k in t
        for k in [
            "cagr",
            "compound",
            "compare",
            "yoy",
            "year-on-year",
            "delta",
            "difference",
            "ratio",
            "correlation",
            "forecast",
        ]
    )


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

class MockLLMClient:
    provider = "mock"

    def __init__(self, model: str = "mock-supervisor"):
        self.model = model

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        text_prompt = messages_to_text(messages)

        agent_tag = _detect_agent_tag(text_prompt)
        user_question = _extract_user_question(messages)

        handler = _HANDLERS.get(agent_tag, _handle_default)
        payload = handler(user_question, text_prompt)
        text = json.dumps(payload, indent=2)

        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_in=len(text_prompt) // 4,
            tokens_out=len(text) // 4,
            raw={"agent": agent_tag},
        )


# ---------------------------------------------------------------------------
# Handlers per agent
# ---------------------------------------------------------------------------

_AGENT_TAG_RE = re.compile(r"\[\[AGENT:\s*([a-zA-Z_\-]+)\s*\]\]")


def _detect_agent_tag(prompt: str) -> str:
    m = _AGENT_TAG_RE.search(prompt)
    return m.group(1).lower() if m else "default"


def _extract_user_question(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _handle_supervisor(question: str, prompt: str) -> dict[str, Any]:
    q = _lc(question)
    plan: list[dict[str, Any]] = []

    wants_numbers = _mentions_numeric_topics(q)
    wants_docs = _mentions_document_topics(q)
    wants_web = _mentions_web(q)
    wants_compute = _mentions_compute(q)

    # Pure-definition questions ("what does DIO stand for") should not hit SQL
    pure_definition = any(k in q for k in ["stand for", "definition of", "glossary", "what does"])
    if pure_definition:
        wants_numbers = False
        wants_compute = False
        wants_docs = True

    if not (wants_numbers or wants_docs or wants_web):
        # fall back to doc research — safest default for a QnA over a brand
        wants_docs = True

    if wants_numbers:
        plan.append({"agent": "sql", "task": "Fetch the relevant sales / inventory metrics from the warehouse."})
    if wants_docs:
        plan.append({"agent": "docs", "task": "Retrieve narrative context (strategy, supply chain, insights) that explains the drivers."})
    if wants_compute:
        plan.append({"agent": "python", "task": "Compute derived metrics (YoY delta, ratios, CAGR) from the SQL result."})
    if wants_web:
        plan.append({"agent": "web", "task": "Look up external corroborating signals."})

    return {
        "plan": plan,
        "rationale": "Route based on whether the question asks for numbers, narrative, or external context.",
        "expected_output": "A concise, cited answer combining structured metrics and document evidence.",
    }


def _handle_sql(question: str, prompt: str) -> dict[str, Any]:
    """Produce a safe SQL query based on keywords in the question."""
    q = _lc(question)

    brand_filter = None
    for name, bid in _BRAND_HINTS.items():
        if name in q:
            brand_filter = bid
            break
    region_filter = None
    for name, rid in _REGION_HINTS.items():
        if name in q:
            region_filter = rid
            break
    country_filter = None
    for name, mid in _COUNTRY_HINTS.items():
        if name in q:
            country_filter = mid
            break
    channel_filter = None
    for name, cid in _CHANNEL_HINTS.items():
        if name in q:
            channel_filter = cid
            break

    date_from, date_to = _guess_date_window(q)

    metric = "net_revenue_usd"
    if "gross" in q:
        metric = "gross_revenue_usd"
    elif "units" in q or "volume" in q:
        metric = "volume_units"

    # Group-by uses human-friendly names by joining dim_product / dim_geography
    group_expressions: list[tuple[str, str]] = [("p.brand_name", "brand")]
    if "channel" in q:
        group_expressions.append(("c.channel_name", "channel"))
    if "market" in q or "country" in q:
        group_expressions.append(("g.country", "country"))
    if "region" in q:
        group_expressions.append(("g.region", "region"))
    if "month" in q or "trend" in q:
        group_expressions.append(("s.date", "date"))

    where = []
    if brand_filter:
        where.append(f"s.brand_id = '{brand_filter}'")
    if country_filter:
        where.append(f"s.market_id = '{country_filter}'")
    if region_filter:
        where.append(f"g.region_id = '{region_filter}'")
    if channel_filter:
        where.append(f"s.channel_id = '{channel_filter}'")
    if date_from:
        where.append(f"s.date >= DATE '{date_from}'")
    if date_to:
        where.append(f"s.date <= DATE '{date_to}'")

    select_cols = [f"{expr} AS {alias}" for expr, alias in group_expressions]
    select_cols.append(f"SUM({metric}) AS {metric.replace('_usd','')}")
    group_cols = [expr for expr, _ in group_expressions]
    order_metric = metric.replace("_usd", "")

    sql = (
        f"SELECT {', '.join(select_cols)} "
        "FROM fact_sales s "
        "JOIN dim_product p USING (sku_id) "
        "JOIN dim_geography g USING (market_id) "
        "JOIN dim_channel c USING (channel_id) "
        f"{'WHERE ' + ' AND '.join(where) if where else ''} "
        f"GROUP BY {', '.join(group_cols)} "
        f"ORDER BY {order_metric} DESC"
    )
    return {"sql": sql, "reason": "Keyword-derived scope."}


def _handle_docs(question: str, prompt: str) -> dict[str, Any]:
    q = question.strip().rstrip("?")
    lc = q.lower()
    queries: list[str] = [q]

    # Salient noun-phrase extraction: keep tokens that are >4 chars or
    # match a known FMCG term. This surfaces the "campaign spending" style
    # long-tail sections that a single full-question query might miss.
    salient: list[str] = []
    for token in re.split(r"[^a-zA-Z0-9\-]+", lc):
        if len(token) >= 4 and token not in {
            "what", "when", "where", "which", "does", "much", "many",
            "give", "root", "show", "this", "that", "have", "with",
            "from", "into", "them", "they", "were", "will",
        }:
            salient.append(token)
    if salient:
        queries.append(" ".join(salient[:6]))

    for pair in ("campaign spend", "brand sentiment", "clean-label", "supply chain",
                 "rebate", "kpi definition", "days of inventory"):
        if any(w in lc for w in pair.split()):
            queries.append(pair)

    # Deduplicate, preserving order
    seen: set[str] = set()
    dedup = [x for x in queries if not (x in seen or seen.add(x))]
    return {"queries": dedup[:4], "top_k": 6}


def _handle_web(question: str, prompt: str) -> dict[str, Any]:
    q = _lc(question).strip("?")
    return {
        "queries": [f"FMCG industry {q}", f"{q} 2026 market outlook"],
        "top_k": 3,
    }


def _handle_python(question: str, prompt: str) -> dict[str, Any]:
    # Pull SQL data from the prompt (agents inject it under a JSON block).
    data = coerce_dict(prompt, defaults={"rows": []})
    return {
        "code": "\n".join(
            [
                "# YoY / delta helper — safe on empty inputs",
                "result = {}",
                "if rows:",
                "    total = sum((r.get('net_revenue') or r.get('gross_revenue') or 0) for r in rows)",
                "    result['total'] = total",
                "    result['row_count'] = len(rows)",
                "    if len(rows) >= 2:",
                "        first = rows[0].get('net_revenue') or rows[0].get('gross_revenue') or 0",
                "        last  = rows[-1].get('net_revenue') or rows[-1].get('gross_revenue') or 0",
                "        result['delta_abs'] = last - first",
                "        result['delta_pct'] = ((last - first) / first) if first else None",
            ]
        ),
        "explanation": "Compute totals, first→last delta and percentage change.",
    }


def _handle_validator(question: str, prompt: str) -> dict[str, Any]:
    # Simple sanity checks — the real agent code enforces stricter rules.
    prompt_l = _lc(prompt)
    issues: list[str] = []
    if "sql_result" in prompt_l and "empty" in prompt_l:
        issues.append("SQL returned no rows — widen the date/entity filter.")
    if "documents" in prompt_l and "no_hits" in prompt_l:
        issues.append("No document evidence retrieved — try alternate keywords.")
    verdict = "pass" if not issues else "revise"
    return {"verdict": verdict, "issues": issues, "confidence": 0.72 if not issues else 0.35}


def _handle_synth(question: str, prompt: str) -> dict[str, Any]:
    # Never called for text — the synth agent uses a template renderer, not
    # the LLM output. But if it is invoked, produce a concise stub.
    return {"answer": "See the structured evidence.", "confidence": 0.6}


_HANDLERS = {
    "supervisor": _handle_supervisor,
    "sql": _handle_sql,
    "docs": _handle_docs,
    "web": _handle_web,
    "python": _handle_python,
    "validator": _handle_validator,
    "synth": _handle_synth,
    "synthesizer": _handle_synth,
}


def _handle_default(question: str, prompt: str) -> dict[str, Any]:
    return {"answer": "unsupported"}


# ---------------------------------------------------------------------------
# Date-window inference (deterministic!)
# ---------------------------------------------------------------------------

_QUARTER_RE = re.compile(r"(q[1-4])\s*(20\d{2})", re.IGNORECASE)


def _guess_date_window(q: str) -> tuple[str | None, str | None]:
    m = _QUARTER_RE.search(q)
    if m:
        quarter = int(m.group(1)[1])
        year = int(m.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3
        return (f"{year}-{start_month:02d}-01", f"{year}-{end_month:02d}-01")

    year_match = re.search(r"20(24|25|26)", q)
    if year_match:
        year = int(year_match.group(0))
        return (f"{year}-01-01", f"{year}-12-01")
    return (None, None)

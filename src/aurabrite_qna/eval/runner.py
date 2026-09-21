"""Evaluation runner.

Scores each gold-set item on:

    * ``keyword_recall``     — fraction of expected keywords present in the
                                final answer.
    * ``citation_recall``    — fraction of expected documents actually cited
                                in the assembled evidence.
    * ``routing_precision``  — fraction of the agents the supervisor planned
                                that matched the expected agent set.
    * ``routing_recall``     — fraction of expected agents that were actually
                                planned.
    * ``latency_ms``         — total wall-clock time for the graph.
    * ``pass``               — combined heuristic (all recalls ≥ 0.5).

The suite is deterministic when running with the mock LLM provider.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from ..agents import build_orchestrator

DEFAULT_DATASET = Path(__file__).parent / "gold_qna.json"


@dataclass
class ItemResult:
    id: str
    question: str
    answer: str
    plan_agents: list[str]
    cited_docs: list[str]
    keyword_hits: list[str]
    keyword_recall: float
    citation_recall: float
    routing_precision: float
    routing_recall: float
    latency_ms: float
    passed: bool


@dataclass
class EvalReport:
    items: list[ItemResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "items": [asdict(i) for i in self.items],
        }


def run_evaluation(
    dataset_path: Path | None = None, output_path: Path | None = None
) -> EvalReport:
    dataset_path = Path(dataset_path or DEFAULT_DATASET)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    orc = build_orchestrator()

    results: list[ItemResult] = []
    for item in dataset["items"]:
        result = _run_item(orc, item)
        results.append(result)

    summary = _summarise(results)
    report = EvalReport(items=results, summary=summary)

    if output_path is not None:
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return report


def _run_item(orc, item: dict[str, Any]) -> ItemResult:
    t0 = time.perf_counter()
    state = orc.answer(item["question"])
    latency = (time.perf_counter() - t0) * 1000.0

    ans = state.answer.lower()
    expected_keywords = [k.lower() for k in item.get("expected_keywords", [])]
    kw_hits = [k for k in expected_keywords if k in ans]
    kw_recall = (len(kw_hits) / len(expected_keywords)) if expected_keywords else 1.0

    expected_docs = set(item.get("expected_citations", []))
    cited_docs = _cited_docs(state)
    doc_recall = (
        (len(expected_docs & set(cited_docs)) / len(expected_docs))
        if expected_docs
        else 1.0
    )

    expected_agents = set(item.get("expected_agents", []))
    planned = {p.agent for p in state.plan}
    if expected_agents:
        routing_recall = len(expected_agents & planned) / len(expected_agents)
        routing_precision = (
            len(expected_agents & planned) / len(planned) if planned else 0.0
        )
    else:
        routing_recall = routing_precision = 1.0

    passed = kw_recall >= 0.5 and doc_recall >= 0.5 and routing_recall >= 0.5

    return ItemResult(
        id=item["id"],
        question=item["question"],
        answer=state.answer,
        plan_agents=sorted(planned),
        cited_docs=sorted(cited_docs),
        keyword_hits=kw_hits,
        keyword_recall=round(kw_recall, 3),
        citation_recall=round(doc_recall, 3),
        routing_precision=round(routing_precision, 3),
        routing_recall=round(routing_recall, 3),
        latency_ms=round(latency, 1),
        passed=passed,
    )


def _cited_docs(state) -> list[str]:
    docs: set[str] = set()
    for e in state.evidence:
        if e.kind == "docs":
            # Citation looks like "Foo.md § Section" — take everything up to §
            doc = e.citation.split("§", 1)[0].strip()
            if doc:
                docs.add(doc)
    return sorted(docs)


def _summarise(results: list[ItemResult]) -> dict[str, Any]:
    if not results:
        return {}
    return {
        "n_questions": len(results),
        "n_passed": sum(1 for r in results if r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 3),
        "avg_keyword_recall": round(mean(r.keyword_recall for r in results), 3),
        "avg_citation_recall": round(mean(r.citation_recall for r in results), 3),
        "avg_routing_precision": round(mean(r.routing_precision for r in results), 3),
        "avg_routing_recall": round(mean(r.routing_recall for r in results), 3),
        "avg_latency_ms": round(mean(r.latency_ms for r in results), 1),
    }

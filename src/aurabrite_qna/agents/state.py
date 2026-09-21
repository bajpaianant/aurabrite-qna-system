"""Shared state passed between graph nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EvidenceKind = Literal["sql", "docs", "web", "python"]


@dataclass
class PlanStep:
    agent: str
    task: str
    done: bool = False


@dataclass
class Evidence:
    kind: EvidenceKind
    summary: str
    citation: str
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "citation": self.citation,
            "payload": self.payload,
        }


@dataclass
class TraceEntry:
    node: str
    input_summary: str
    output_summary: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    question: str
    plan: list[PlanStep] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[TraceEntry] = field(default_factory=list)
    validation_round: int = 0
    validator_verdict: str = ""
    validator_issues: list[str] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # helpers -----------------------------------------------------------
    def evidence_by_kind(self, kind: EvidenceKind) -> list[Evidence]:
        return [e for e in self.evidence if e.kind == kind]

    def append_trace(self, node: str, input_summary: str, output_summary: str,
                     latency_ms: float, metadata: dict[str, Any] | None = None) -> None:
        self.trace.append(
            TraceEntry(
                node=node,
                input_summary=input_summary,
                output_summary=output_summary,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "plan": [p.__dict__ for p in self.plan],
            "evidence": [e.as_dict() for e in self.evidence],
            "trace": [t.__dict__ for t in self.trace],
            "answer": self.answer,
            "confidence": self.confidence,
            "validator": {
                "round": self.validation_round,
                "verdict": self.validator_verdict,
                "issues": self.validator_issues,
            },
            "done": self.done,
        }

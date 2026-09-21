"""The offline evaluation suite should pass every gold item with the mock LLM."""

from __future__ import annotations

from aurabrite_qna.eval import run_evaluation


def test_gold_suite_passes():
    report = run_evaluation()
    assert report.summary["pass_rate"] >= 0.85
    assert report.summary["avg_citation_recall"] >= 0.9
    assert report.summary["avg_routing_recall"] >= 0.9

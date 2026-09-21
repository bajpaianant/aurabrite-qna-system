"""End-to-end tests for the multi-agent orchestrator.

Uses the deterministic mock LLM provider so all tests are hermetic and fast.
"""

from __future__ import annotations

from aurabrite_qna.agents import build_orchestrator


def test_end_to_end_apac_question(sample_env):
    orc = build_orchestrator(
        warehouse_path=sample_env["warehouse_path"],
        docs_dir=sample_env["docs_dir"],
        vector_dir=sample_env["vector_dir"],
    )
    state = orc.answer("Why did AuraGlow sales drop in APAC in Q4 2025?")
    assert state.done
    assert state.plan  # supervisor produced a plan
    kinds = {e.kind for e in state.evidence}
    assert "sql" in kinds and "docs" in kinds
    answer_lc = state.answer.lower()
    assert "singapore" in answer_lc
    assert "auraglow" in answer_lc


def test_end_to_end_pure_definition(sample_env):
    orc = build_orchestrator(
        warehouse_path=sample_env["warehouse_path"],
        docs_dir=sample_env["docs_dir"],
        vector_dir=sample_env["vector_dir"],
    )
    state = orc.answer("What does DIO stand for?")
    # Definition question should route to docs, not SQL
    plan_agents = {p.agent for p in state.plan}
    assert "docs" in plan_agents
    assert "sql" not in plan_agents


def test_yoy_uses_python_agent(sample_env):
    orc = build_orchestrator(
        warehouse_path=sample_env["warehouse_path"],
        docs_dir=sample_env["docs_dir"],
        vector_dir=sample_env["vector_dir"],
    )
    state = orc.answer("What was the YoY growth of NutriVita net revenue between 2024 and 2025?")
    plan_agents = {p.agent for p in state.plan}
    assert "sql" in plan_agents and "python" in plan_agents
    py_ev = [e for e in state.evidence if e.kind == "python"]
    assert py_ev, "python agent should have produced evidence"


def test_validator_bounded_retry(sample_env):
    """Validator round count must never exceed the configured cap."""
    orc = build_orchestrator(
        warehouse_path=sample_env["warehouse_path"],
        docs_dir=sample_env["docs_dir"],
        vector_dir=sample_env["vector_dir"],
    )
    orc.max_validation_rounds = 1
    state = orc.answer("Explain the Dentafresh EMEA situation.")
    assert state.validation_round <= 1
    assert state.done

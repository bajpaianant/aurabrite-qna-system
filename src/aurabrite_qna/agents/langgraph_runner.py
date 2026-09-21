"""LangGraph-backed runner (optional).

This module wires the same worker functions used by ``Orchestrator`` into
a real LangGraph ``StateGraph``. It is disabled by default because the
in-process ``Orchestrator`` is more portable and has identical semantics,
but this file exists so we can demonstrate a genuine LangGraph topology
in the notebook / architecture doc.

Usage
-----
    >>> from aurabrite_qna.agents.langgraph_runner import build_langgraph_app
    >>> app = build_langgraph_app()
    >>> final = app.invoke({"question": "..."})
"""

from __future__ import annotations

import logging
from typing import Any

from .orchestrator import build_orchestrator
from .state import AgentState

log = logging.getLogger(__name__)


def build_langgraph_app():  # -> "langgraph.graph.CompiledGraph"
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "LangGraph is not installed. Install with `pip install langgraph`."
        ) from e

    orc = build_orchestrator()

    def _plan(state: dict[str, Any]) -> dict[str, Any]:
        s = _ensure_state(state)
        return orc.supervisor(s).__dict__

    def _sql(state):   s = _ensure_state(state); return orc.sql_agent(s).__dict__
    def _docs(state):  s = _ensure_state(state); return orc.docs_agent(s).__dict__
    def _web(state):   s = _ensure_state(state); return orc.web_agent(s).__dict__
    def _python(state):s = _ensure_state(state); return orc.python_agent(s).__dict__
    def _valid(state): s = _ensure_state(state); return orc.validator(s).__dict__
    def _synth(state): s = _ensure_state(state); return orc.synthesizer(s).__dict__

    def _route_after_plan(state: dict[str, Any]) -> str:
        plan_agents = {p["agent"] for p in state.get("plan", []) if not p.get("done")}
        if "sql" in plan_agents:      return "sql"
        if "docs" in plan_agents:     return "docs"
        if "web" in plan_agents:      return "web"
        if "python" in plan_agents:   return "python"
        return "validate"

    def _route_after_validate(state: dict[str, Any]) -> str:
        if state.get("validator_verdict") == "pass":
            return "synth"
        if state.get("validation_round", 0) >= orc.max_validation_rounds:
            return "synth"
        return "plan"

    graph = StateGraph(dict)
    graph.add_node("plan", _plan)
    graph.add_node("sql", _sql)
    graph.add_node("docs", _docs)
    graph.add_node("web", _web)
    graph.add_node("python", _python)
    graph.add_node("validate", _valid)
    graph.add_node("synth", _synth)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", _route_after_plan,
        {"sql": "sql", "docs": "docs", "web": "web", "python": "python", "validate": "validate"})
    for worker in ("sql", "docs", "web", "python"):
        graph.add_edge(worker, "validate")
    graph.add_conditional_edges("validate", _route_after_validate,
        {"plan": "plan", "synth": "synth"})
    graph.add_edge("synth", END)

    return graph.compile()


def _ensure_state(state: dict[str, Any] | AgentState) -> AgentState:
    if isinstance(state, AgentState):
        return state
    # Rehydrate a lightweight AgentState from the dict view LangGraph passes.
    from .state import Evidence, PlanStep, TraceEntry

    s = AgentState(
        question=state.get("question", ""),
        plan=[PlanStep(**p) if isinstance(p, dict) else p for p in state.get("plan", [])],
        evidence=[Evidence(**e) if isinstance(e, dict) else e for e in state.get("evidence", [])],
        trace=[TraceEntry(**t) if isinstance(t, dict) else t for t in state.get("trace", [])],
        validation_round=state.get("validation_round", 0),
        validator_verdict=state.get("validator_verdict", ""),
        validator_issues=state.get("validator_issues", []),
        answer=state.get("answer", ""),
        confidence=state.get("confidence", 0.0),
        done=state.get("done", False),
        metadata=state.get("metadata", {}),
    )
    return s

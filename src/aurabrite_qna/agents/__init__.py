"""Hierarchical supervisor-worker agent system.

The public entry-point is ``build_graph`` which constructs and returns an
executable graph (a LangGraph ``StateGraph`` when the dependency is
installed, otherwise a lightweight in-process fallback with identical
semantics — see ``fallback_graph.py``).

Nodes
-----
* **supervisor**  — plans which workers to invoke.
* **sql_analyst** — grounded structured analytics.
* **doc_researcher** — hybrid RAG over the enterprise knowledge base.
* **web_researcher** — external / web signals.
* **python_analyst** — deterministic calculations in the sandbox.
* **validator**   — cyclical critique. Loops back to supervisor if the
  evidence is insufficient (bounded by ``SETTINGS.validation_rounds``).
* **synthesizer** — writes the final grounded answer with citations.

The state passed between nodes is ``AgentState`` (see ``state.py``).
"""

from .orchestrator import Orchestrator, build_orchestrator
from .state import AgentState, Evidence, PlanStep

__all__ = [
    "AgentState",
    "Evidence",
    "Orchestrator",
    "PlanStep",
    "build_orchestrator",
]

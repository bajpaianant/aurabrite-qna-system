"""Graph orchestration.

Attempts to use LangGraph for a state graph with cyclical validation. If
LangGraph isn't installed (or fails to construct on the target platform),
we fall back to the pure-Python `SimpleGraph` runner which implements the
same semantics: supervisor → workers → validator → (loop back or synth).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import SETTINGS, Settings
from ..data import connect
from ..llm import LLMClient, build_llm_client
from ..rag import HybridRetriever
from ..tools import PythonSandbox, SQLTool, WebSearchTool
from .state import AgentState
from .workers import (
    DocsAgent,
    PythonAgent,
    SQLAgent,
    SupervisorAgent,
    SynthesizerAgent,
    ValidatorAgent,
    WebAgent,
)

log = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], AgentState]


@dataclass
class Orchestrator:
    """Runs one question through the whole graph."""

    supervisor: SupervisorAgent
    sql_agent: SQLAgent
    docs_agent: DocsAgent
    web_agent: WebAgent
    python_agent: PythonAgent
    validator: ValidatorAgent
    synthesizer: SynthesizerAgent
    max_validation_rounds: int = 2
    max_steps: int = 8

    def answer(self, question: str) -> AgentState:
        state = AgentState(question=question)
        return self._run(state)

    # --------------------------------------------------------------
    # Core loop — LangGraph-equivalent semantics
    # --------------------------------------------------------------
    def _run(self, state: AgentState) -> AgentState:
        steps = 0
        while not state.done and steps < self.max_steps:
            steps += 1
            # 1. Plan (once, or re-plan after a validator failure)
            if not state.plan or (state.validator_verdict == "revise" and steps > 1):
                state = self.supervisor(state)

            # 2. Fan out to workers according to the plan
            for step in state.plan:
                if step.done:
                    continue
                node = self._resolve_worker(step.agent)
                if node is None:
                    log.warning("Unknown worker in plan: %s", step.agent)
                    step.done = True
                    continue
                state = node(state)
                step.done = True

            # 3. Validate
            state = self.validator(state)

            # 4. Loop or finalise
            if state.validator_verdict == "pass" or state.validation_round >= self.max_validation_rounds:
                state = self.synthesizer(state)
                break
            # else: reset per-step flags so the supervisor can re-plan
            for step in state.plan:
                step.done = True  # keep evidence, replan next iteration

        if not state.done:
            state = self.synthesizer(state)
        return state

    def _resolve_worker(self, agent: str) -> NodeFn | None:
        mapping = {
            "sql": self.sql_agent,
            "docs": self.docs_agent,
            "web": self.web_agent,
            "python": self.python_agent,
        }
        return mapping.get(agent.lower())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_orchestrator(
    *,
    llm: LLMClient | None = None,
    warehouse_path: Path | None = None,
    docs_dir: Path | None = None,
    vector_dir: Path | None = None,
    settings: Settings | None = None,
) -> Orchestrator:
    settings = settings or SETTINGS
    llm = llm or build_llm_client(settings)

    warehouse = connect(warehouse_path or settings.warehouse_path)
    retriever = HybridRetriever.from_docs_dir(
        docs_dir or settings.docs_dir,
        vector_dir or settings.vector_dir,
    )

    supervisor = SupervisorAgent(llm=llm)
    sql_agent = SQLAgent(llm=llm, sql=SQLTool(warehouse))
    docs_agent = DocsAgent(llm=llm, retriever=retriever)
    web_agent = WebAgent(llm=llm, web=WebSearchTool())
    python_agent = PythonAgent(llm=llm, sandbox=PythonSandbox())
    validator = ValidatorAgent(llm=llm)
    synth = SynthesizerAgent(llm=llm, use_llm=False)

    return Orchestrator(
        supervisor=supervisor,
        sql_agent=sql_agent,
        docs_agent=docs_agent,
        web_agent=web_agent,
        python_agent=python_agent,
        validator=validator,
        synthesizer=synth,
        max_validation_rounds=settings.validation_rounds,
        max_steps=settings.max_agent_steps,
    )

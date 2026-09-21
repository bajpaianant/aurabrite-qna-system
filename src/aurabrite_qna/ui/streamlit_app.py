"""Streamlit front-end for the AuraBrite QnA multi-agent system.

Launch with:  ``aurabrite ui``   (or ``streamlit run src/aurabrite_qna/ui/streamlit_app.py``)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running the file directly with `streamlit run ...`
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from aurabrite_qna.agents import build_orchestrator  # noqa: E402
from aurabrite_qna.config import SETTINGS  # noqa: E402
from aurabrite_qna.data import connect  # noqa: E402
from aurabrite_qna.data.generator import generate_all  # noqa: E402
from aurabrite_qna.rag import HybridRetriever  # noqa: E402


st.set_page_config(page_title="AuraBrite QnA", page_icon="🌿", layout="wide")


@st.cache_resource(show_spinner=False)
def _get_orchestrator():
    return build_orchestrator()


@st.cache_resource(show_spinner=False)
def _get_retriever():
    return HybridRetriever.from_docs_dir(SETTINGS.docs_dir, SETTINGS.vector_dir)


def _ensure_data() -> None:
    if not SETTINGS.warehouse_path.exists():
        with st.spinner("Bootstrapping synthetic warehouse & RAG index (one-time) ..."):
            with connect(SETTINGS.warehouse_path) as wh:
                generate_all(wh, SETTINGS.docs_dir)
            HybridRetriever.rebuild(SETTINGS.docs_dir, SETTINGS.vector_dir)


def main() -> None:
    _ensure_data()

    st.title("🌿 AuraBrite Consumer Brands — Multi-Agent QnA")
    st.caption(
        "Hierarchical supervisor · SQL analyst · Document researcher · Web scout · "
        "Python sandbox · Validator · Synthesizer"
    )

    with st.sidebar:
        st.subheader("Configuration")
        st.text(f"LLM provider : {SETTINGS.llm_provider}")
        st.text(f"LLM model    : {SETTINGS.llm_model}")
        st.text(f"Web search   : {SETTINGS.web_search_provider}")
        st.text(f"Warehouse    : {SETTINGS.warehouse_path.name}")
        st.divider()
        st.subheader("Try one of these")
        examples = [
            "Why did AuraGlow sales drop in APAC in Q4 2025, and by how much?",
            "Show NutriVita e-commerce net revenue for Q1–Q2 2026.",
            "What is the top brand by units in Modern Trade in 2025?",
            "Explain the Dentafresh EMEA volume decline and its root cause.",
            "What is our clean-label opportunity for 2026?",
        ]
        chosen = None
        for ex in examples:
            if st.button(ex, use_container_width=True):
                chosen = ex

    question = st.text_input(
        "Enterprise question", value=chosen or "", placeholder="Ask about AuraBrite ..."
    )
    submitted = st.button("Ask", type="primary", disabled=not question)

    if submitted and question:
        with st.spinner("Running multi-agent graph ..."):
            state = _get_orchestrator().answer(question)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(state.answer)

        with col2:
            st.metric("Confidence", f"{state.confidence:.2f}")
            st.metric("Validation rounds", state.validation_round)
            st.metric("Evidence items", len(state.evidence))

        with st.expander("🧭 Plan & trace", expanded=False):
            if state.plan:
                st.write("**Plan:**")
                for step in state.plan:
                    st.write(f"- **{step.agent}** — {step.task}")
            trace_df = pd.DataFrame(
                [
                    {
                        "node": t.node,
                        "latency_ms": round(t.latency_ms, 2),
                        "summary": t.output_summary,
                    }
                    for t in state.trace
                ]
            )
            st.dataframe(trace_df, use_container_width=True, hide_index=True)

        with st.expander("📎 Evidence", expanded=False):
            for e in state.evidence:
                st.markdown(f"**[{e.kind}]** _{e.citation}_")
                if e.kind == "sql" and e.payload.get("markdown"):
                    st.markdown(e.payload["markdown"])
                    st.code(e.payload.get("sql", ""), language="sql")
                elif e.kind == "docs":
                    st.write(e.payload.get("text", e.summary))
                elif e.kind == "web":
                    st.write(e.payload.get("snippet", e.summary))
                    if e.payload.get("url"):
                        st.write(f"↗ {e.payload['url']}")
                elif e.kind == "python":
                    st.code(e.payload.get("code", ""), language="python")
                    st.write("**Result:**", e.payload.get("result"))
                st.divider()

        with st.expander("🧾 Full state (JSON)", expanded=False):
            st.code(json.dumps(state.to_dict(), indent=2, default=str), language="json")


if __name__ == "__main__":
    main()

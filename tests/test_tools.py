"""Unit tests for the tool suite: SQL, sandbox, RAG."""

from __future__ import annotations

import pytest

from aurabrite_qna.data import connect
from aurabrite_qna.rag import HybridRetriever
from aurabrite_qna.tools import (
    PythonSandbox,
    SandboxError,
    SQLTool,
    SQLValidationError,
)


# --------------------------------------------------------------------------
# SQL tool
# --------------------------------------------------------------------------

def test_sql_tool_runs_select(sample_env):
    wh = connect(sample_env["warehouse_path"])
    tool = SQLTool(wh)
    r = tool.run("SELECT COUNT(*) AS n FROM fact_sales")
    assert r.row_count == 1
    assert r.columns == ["n"]


def test_sql_tool_rejects_ddl(sample_env):
    wh = connect(sample_env["warehouse_path"])
    tool = SQLTool(wh)
    with pytest.raises(SQLValidationError):
        tool.run("DROP TABLE fact_sales")
    with pytest.raises(SQLValidationError):
        tool.run("SELECT 1; SELECT 2")
    with pytest.raises(SQLValidationError):
        tool.run("SELECT * FROM users")   # unknown table


def test_sql_tool_appends_limit(sample_env):
    wh = connect(sample_env["warehouse_path"])
    tool = SQLTool(wh, row_limit=17)
    r = tool.run("SELECT * FROM fact_sales")
    assert r.row_count <= 17
    assert "LIMIT" in tool.validate("SELECT * FROM fact_sales")


# --------------------------------------------------------------------------
# Python sandbox
# --------------------------------------------------------------------------

def test_sandbox_computes_result():
    sb = PythonSandbox()
    res = sb.run("result = sum([1,2,3,4])")
    assert res.result == 10
    assert not res.error


def test_sandbox_blocks_import():
    sb = PythonSandbox()
    with pytest.raises(SandboxError):
        sb.run("import os")
    with pytest.raises(SandboxError):
        sb.run("__import__('os')")


def test_sandbox_blocks_dunder_access():
    sb = PythonSandbox()
    with pytest.raises(SandboxError):
        sb.run("x = ().__class__.__mro__")


# --------------------------------------------------------------------------
# Hybrid RAG
# --------------------------------------------------------------------------

def test_retriever_finds_supply_chain_report(sample_env):
    r = HybridRetriever.from_docs_dir(sample_env["docs_dir"], sample_env["vector_dir"])
    hits = r.search("Singapore port strike Q4 2025", top_k=3)
    citations = [h.chunk.citation for h in hits]
    assert any("Supply_Chain_Disruption_Report" in c for c in citations)


def test_retriever_finds_campaign_spending(sample_env):
    r = HybridRetriever.from_docs_dir(sample_env["docs_dir"], sample_env["vector_dir"])
    hits = r.search("campaign spending 2026", top_k=3)
    assert any("Campaign Spending" in h.chunk.heading for h in hits)

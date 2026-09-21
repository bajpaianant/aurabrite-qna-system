"""Shared pytest fixtures: bootstrap a temp warehouse + doc set once per session."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurabrite_qna.data import connect
from aurabrite_qna.data.generator import generate_all
from aurabrite_qna.rag import HybridRetriever


@pytest.fixture(scope="session")
def sample_env(tmp_path_factory) -> dict:
    """Deterministic mini-warehouse + doc corpus for the whole test session."""
    root = tmp_path_factory.mktemp("aurabrite_env")
    warehouse_path = root / "wh.duckdb"
    docs_dir = root / "docs"
    vector_dir = root / "vec"

    with connect(warehouse_path) as wh:
        generate_all(wh, docs_dir, seed=42)
    HybridRetriever.rebuild(docs_dir, vector_dir)

    return {
        "root": root,
        "warehouse_path": warehouse_path,
        "docs_dir": docs_dir,
        "vector_dir": vector_dir,
    }

"""AuraBrite Consumer Brands — Enterprise Multi-Agent QnA System.

A hierarchical supervisor-worker agent architecture (LangGraph-based, with a
zero-dependency fallback) that answers cross-modal enterprise questions over
synthetic FMCG data: SQL warehouse, unstructured documents, web signals, and
executable Python analytics.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aurabrite-qna")
except PackageNotFoundError:  # pragma: no cover — running from source
    __version__ = "0.1.0"

__all__ = ["__version__"]

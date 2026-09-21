"""Retrieval-augmented-generation subsystem.

Public API:
    * ``build_index(docs_dir, out_dir)`` — one-shot indexing.
    * ``HybridRetriever`` — BM25 + TF-IDF cosine hybrid retriever.
    * ``DocumentChunk`` — a retrievable passage with citation metadata.
"""

from .chunker import DocumentChunk, chunk_documents
from .index import HybridIndex, build_index, load_index
from .retriever import HybridRetriever, RetrievalHit

__all__ = [
    "DocumentChunk",
    "HybridIndex",
    "HybridRetriever",
    "RetrievalHit",
    "build_index",
    "chunk_documents",
    "load_index",
]

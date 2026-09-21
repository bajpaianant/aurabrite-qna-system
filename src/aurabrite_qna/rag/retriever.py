"""Hybrid retriever with score fusion (Reciprocal Rank Fusion by default)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .chunker import DocumentChunk
from .index import HybridIndex, build_index, load_index

FusionMethod = Literal["rrf", "weighted"]


@dataclass
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    bm25: float
    cosine: float

    def as_dict(self) -> dict:
        return {
            "citation": self.chunk.citation,
            "chunk_id": self.chunk.chunk_id,
            "score": round(self.score, 4),
            "bm25": round(self.bm25, 4),
            "cosine": round(self.cosine, 4),
            "text": self.chunk.text,
        }


class HybridRetriever:
    def __init__(
        self,
        index: HybridIndex,
        fusion: FusionMethod = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        self.index = index
        self.fusion = fusion
        self.alpha = alpha
        self.rrf_k = rrf_k

    @classmethod
    def from_docs_dir(cls, docs_dir: Path, index_dir: Path) -> "HybridRetriever":
        try:
            idx = load_index(index_dir)
        except FileNotFoundError:
            idx = build_index(docs_dir, index_dir)
        return cls(idx)

    @classmethod
    def rebuild(cls, docs_dir: Path, index_dir: Path) -> "HybridRetriever":
        return cls(build_index(docs_dir, index_dir))

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        bm25 = self.index.bm25_scores(query)
        cos = self.index.cosine_scores(query)

        if self.fusion == "weighted":
            merged = [self.alpha * b + (1 - self.alpha) * c for b, c in zip(bm25, cos)]
        else:
            # Reciprocal Rank Fusion
            bm25_rank = _ranks(bm25)
            cos_rank = _ranks(cos)
            merged = [
                1.0 / (self.rrf_k + bm25_rank[i]) + 1.0 / (self.rrf_k + cos_rank[i])
                for i in range(len(bm25))
            ]

        # Heading-boost: reward sections whose heading tokens intersect the
        # query. Headings are highly topical and this is a cheap way to
        # break RRF ties that would otherwise favour the section with more
        # brand mentions in its body.
        q_terms = {t for t in query.lower().split() if len(t) > 3}
        boosted: list[float] = []
        for i, chunk in enumerate(self.index.chunks):
            heading_terms = {t for t in chunk.heading.lower().split() if len(t) > 3}
            overlap = len(q_terms & heading_terms)
            boost = 0.02 * overlap
            boosted.append(merged[i] + boost)

        ranked = sorted(enumerate(boosted), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalHit(
                chunk=self.index.chunks[i],
                score=score,
                bm25=bm25[i],
                cosine=cos[i],
            )
            for i, score in ranked
            if score > 0
        ]


def _ranks(scores: list[float]) -> list[int]:
    """Return 1-based rank per item (ties broken by original order)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for r, idx in enumerate(order, 1):
        ranks[idx] = r
    return ranks

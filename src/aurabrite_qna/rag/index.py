"""Build & load a hybrid BM25 + TF-IDF index.

Design notes
------------
We deliberately keep the vector side embedding-free by default: a bag-of-
words TF-IDF cosine gives us a semantic-ish channel that complements BM25
without pulling in ChromaDB/FAISS or a network-hosted embedding model.
When a user wires in a real embedding provider they can subclass
``HybridIndex.embed`` — the storage layout is compatible.

The index is a single JSON file (``index.json``) plus the pickled BM25
tokens list. Everything is stdlib except `rank_bm25` (already a dependency).
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rank_bm25 import BM25Okapi

from .chunker import DocumentChunk, chunk_documents

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")
_STOPWORDS = frozenset(
    """a an the and or of to in on at for is are was were be by with from as
    it its this that these those we our their they he she his her not no if
    but so than then into over under between about against during without
    within after before across per via up down out just also more most such
    can may might will would should could shall""".split()
)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


@dataclass
class HybridIndex:
    chunks: list[DocumentChunk]
    tokenised: list[list[str]]
    vocab: dict[str, int]          # term -> id
    idf: list[float]                # aligned with vocab id
    doc_vectors: list[dict[int, float]]  # sparse TF-IDF per chunk (L2-normed)

    # Not serialised — reconstructed from tokenised
    bm25: BM25Okapi | None = field(default=None, repr=False)

    # ----- persistence -------------------------------------------------
    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.json").write_text(
            json.dumps(
                {
                    "chunks": [asdict(c) for c in self.chunks],
                    "vocab": self.vocab,
                    "idf": self.idf,
                    "doc_vectors": [
                        {str(k): v for k, v in dv.items()} for dv in self.doc_vectors
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "tokens.pkl").write_bytes(pickle.dumps(self.tokenised))
        log.info("Wrote hybrid index → %s (%d chunks)", out_dir, len(self.chunks))

    @classmethod
    def load(cls, in_dir: Path) -> "HybridIndex":
        in_dir = Path(in_dir)
        raw = json.loads((in_dir / "index.json").read_text(encoding="utf-8"))
        tokens = pickle.loads((in_dir / "tokens.pkl").read_bytes())
        chunks = [DocumentChunk(**c) for c in raw["chunks"]]
        doc_vectors = [
            {int(k): v for k, v in dv.items()} for dv in raw["doc_vectors"]
        ]
        idx = cls(
            chunks=chunks,
            tokenised=tokens,
            vocab=raw["vocab"],
            idf=raw["idf"],
            doc_vectors=doc_vectors,
        )
        idx.bm25 = BM25Okapi(idx.tokenised)
        return idx

    # ----- retrieval helpers ------------------------------------------
    def bm25_scores(self, query: str) -> list[float]:
        assert self.bm25 is not None, "call HybridIndex.load or use build_index"
        return list(self.bm25.get_scores(_tokenize(query)))

    def cosine_scores(self, query: str) -> list[float]:
        q_vec = self._tfidf_vec(_tokenize(query))
        norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {k: v / norm for k, v in q_vec.items()}
        scores: list[float] = []
        for dv in self.doc_vectors:
            # dv already L2-normed
            common = set(q_vec) & set(dv)
            scores.append(sum(q_vec[t] * dv[t] for t in common))
        return scores

    def _tfidf_vec(self, tokens: Sequence[str]) -> dict[int, float]:
        counts = Counter(tokens)
        vec: dict[int, float] = {}
        for term, count in counts.items():
            term_id = self.vocab.get(term)
            if term_id is None:
                continue
            tf = 1.0 + math.log(count)
            vec[term_id] = tf * self.idf[term_id]
        return vec


# ---------------------------------------------------------------------------
# Public build helpers
# ---------------------------------------------------------------------------

def build_index(docs_dir: Path, out_dir: Path) -> HybridIndex:
    chunks = chunk_documents(docs_dir)
    if not chunks:
        raise ValueError(f"No .md documents found under {docs_dir}")

    tokenised = [_tokenize(c.text) for c in chunks]
    vocab, idf = _build_vocab_idf(tokenised)

    doc_vectors: list[dict[int, float]] = []
    for toks in tokenised:
        counts = Counter(toks)
        vec: dict[int, float] = {}
        for term, count in counts.items():
            tid = vocab.get(term)
            if tid is None:
                continue
            tf = 1.0 + math.log(count)
            vec[tid] = tf * idf[tid]
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vec = {k: v / norm for k, v in vec.items()}
        doc_vectors.append(vec)

    idx = HybridIndex(
        chunks=chunks,
        tokenised=tokenised,
        vocab=vocab,
        idf=idf,
        doc_vectors=doc_vectors,
    )
    idx.bm25 = BM25Okapi(tokenised)
    idx.save(out_dir)
    return idx


def load_index(in_dir: Path) -> HybridIndex:
    return HybridIndex.load(in_dir)


def _build_vocab_idf(tokenised: list[list[str]]) -> tuple[dict[str, int], list[float]]:
    df: Counter[str] = Counter()
    for toks in tokenised:
        for term in set(toks):
            df[term] += 1
    n_docs = len(tokenised)
    vocab: dict[str, int] = {}
    idf: list[float] = []
    for term in sorted(df):
        vocab[term] = len(vocab)
        # standard smooth IDF
        idf.append(math.log((n_docs + 1) / (df[term] + 1)) + 1.0)
    return vocab, idf

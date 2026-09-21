"""Markdown-friendly chunker.

Splits documents on heading boundaries first, then falls back to paragraph
packing to keep chunks around 800 characters. Preserves the heading path so
citations look like ``AuraGlow_Brand_Strategy_2025_2026.md § 4. Brand Sentiment``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class DocumentChunk:
    doc_name: str
    heading: str
    text: str
    chunk_id: str

    @property
    def citation(self) -> str:
        if self.heading:
            return f"{self.doc_name} § {self.heading}"
        return self.doc_name


def chunk_documents(
    docs_dir: Path, target_chars: int = 800
) -> list[DocumentChunk]:
    docs_dir = Path(docs_dir)
    chunks: list[DocumentChunk] = []
    for md_path in sorted(docs_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        chunks.extend(_chunk_one(md_path.name, text, target_chars))
    return chunks


def _chunk_one(doc_name: str, text: str, target_chars: int) -> Iterable[DocumentChunk]:
    # Split into (heading, body) sections
    positions = [(m.start(), m.group(1), m.group(2)) for m in _HEADING_RE.finditer(text)]
    if not positions:
        # No headings — pack the whole file
        yield from _pack("", text, doc_name, target_chars, 0)
        return

    if positions[0][0] > 0:
        yield from _pack("", text[: positions[0][0]], doc_name, target_chars, 0)

    for i, (pos, _, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[pos:end]
        # Drop the heading line from the body for cleaner chunk text
        body_lines = body.splitlines()
        body_text = "\n".join(body_lines[1:]) if body_lines else body
        yield from _pack(heading.strip(), body_text.strip(), doc_name, target_chars, i)


def _pack(
    heading: str, body: str, doc_name: str, target_chars: int, section_idx: int
) -> Iterable[DocumentChunk]:
    body = body.strip()
    if not body:
        return
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    buf: list[str] = []
    running = 0
    part = 0
    for p in paras:
        if running + len(p) > target_chars and buf:
            yield DocumentChunk(
                doc_name=doc_name,
                heading=heading,
                text="\n\n".join(buf),
                chunk_id=f"{doc_name}#{section_idx:02d}.{part:02d}",
            )
            part += 1
            buf = [p]
            running = len(p)
        else:
            buf.append(p)
            running += len(p) + 2
    if buf:
        yield DocumentChunk(
            doc_name=doc_name,
            heading=heading,
            text="\n\n".join(buf),
            chunk_id=f"{doc_name}#{section_idx:02d}.{part:02d}",
        )

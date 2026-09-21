"""Uniform SQL access layer.

Prefers DuckDB (columnar, fast) and transparently falls back to the stdlib
`sqlite3` backend when DuckDB isn't installed. Both engines speak enough of
the ANSI dialect used by our schema and queries.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

try:
    import duckdb  # type: ignore

    _HAS_DUCKDB = True
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore
    _HAS_DUCKDB = False


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]

    def to_markdown(self, max_rows: int = 25) -> str:
        if not self.rows:
            return "_(empty result set)_"
        head = "| " + " | ".join(self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        body_rows = self.rows[:max_rows]
        body = "\n".join(
            "| " + " | ".join(_fmt(v) for v in r) + " |" for r in body_rows
        )
        truncated = (
            f"\n_(showing {max_rows} of {len(self.rows)} rows)_"
            if len(self.rows) > max_rows
            else ""
        )
        return f"{head}\n{sep}\n{body}{truncated}"


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


class Warehouse:
    """Thin, dialect-agnostic SQL connection wrapper."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_DUCKDB:
            self.backend = "duckdb"
            self._conn = duckdb.connect(str(self.path))
        else:  # pragma: no cover — CI covers duckdb path
            self.backend = "sqlite"
            sqlite_path = self.path.with_suffix(".sqlite")
            self._conn = sqlite3.connect(str(sqlite_path))
            self._conn.execute("PRAGMA foreign_keys = ON;")
        log.debug("Opened %s warehouse at %s", self.backend, self.path)

    # ---- low-level ----
    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        if self.backend == "duckdb":
            self._conn.execute(sql, params or [])
        else:
            self._conn.execute(sql, params or [])
            self._conn.commit()

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        if self.backend == "duckdb":
            self._conn.executemany(sql, rows)
        else:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def query(self, sql: str, params: Sequence[Any] | None = None) -> QueryResult:
        if self.backend == "duckdb":
            cur = self._conn.execute(sql, params or [])
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        else:
            cur = self._conn.execute(sql, params or [])
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return QueryResult(columns=cols, rows=rows)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Warehouse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def connect(path: Path | str) -> Warehouse:
    return Warehouse(Path(path))

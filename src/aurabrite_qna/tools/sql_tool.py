"""Safe SQL execution tool.

Uses `sqlparse` if available for pretty parsing, but the security perimeter
is enforced with pure-Python token scanning — we refuse anything that
mutates the warehouse or reaches beyond the allow-listed tables.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from ..data import TABLE_NAMES
from ..data.warehouse import QueryResult, Warehouse

log = logging.getLogger(__name__)

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|GRANT|REVOKE|ATTACH|DETACH|COPY|PRAGMA|VACUUM|EXPORT)\b",
    re.IGNORECASE,
)
_MULTI_STATEMENT = re.compile(r";\s*\S")
_ALLOWED_TABLES = set(TABLE_NAMES)


class SQLValidationError(ValueError):
    """Raised when a query fails our safety perimeter."""


@dataclass
class SQLExecution:
    sql: str
    result: QueryResult
    row_count: int
    columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "columns": self.columns,
            "row_count": self.row_count,
            "rows": self.result.to_records(),
        }


class SQLTool:
    """Read-only SQL runner scoped to the AuraBrite warehouse."""

    def __init__(self, warehouse: Warehouse, row_limit: int = 500):
        self.warehouse = warehouse
        self.row_limit = row_limit

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def run(self, sql: str) -> SQLExecution:
        clean = self.validate(sql)
        result = self.warehouse.query(clean)
        # Enforce row cap defensively
        if len(result.rows) > self.row_limit:
            result.rows = result.rows[: self.row_limit]
        return SQLExecution(
            sql=clean,
            result=result,
            row_count=len(result.rows),
            columns=result.columns,
        )

    def validate(self, sql: str) -> str:
        s = (sql or "").strip().rstrip(";").strip()
        if not s:
            raise SQLValidationError("Empty SQL.")
        if _MULTI_STATEMENT.search(s + ";"):
            raise SQLValidationError("Multi-statement queries are not allowed.")
        if not re.match(r"^\s*(WITH|SELECT)\b", s, re.IGNORECASE):
            raise SQLValidationError("Only SELECT / WITH queries are permitted.")
        if _FORBIDDEN.search(s):
            raise SQLValidationError("Query contains a forbidden keyword.")
        self._check_tables(s)
        # Enforce a limit if the query has none — cheap safety net.
        if re.search(r"\blimit\b\s+\d+", s, re.IGNORECASE) is None:
            s = f"{s} LIMIT {self.row_limit}"
        return s

    # --------------------------------------------------------------
    # Introspection helpers (for the SQL agent prompt)
    # --------------------------------------------------------------
    def describe_schema(self) -> str:
        """Return a compact, human-readable schema summary."""
        parts: list[str] = ["Tables:"]
        for t in TABLE_NAMES:
            try:
                cols = self.warehouse.query(f"SELECT * FROM {t} LIMIT 0").columns
                parts.append(f"  - {t}({', '.join(cols)})")
            except Exception as e:
                parts.append(f"  - {t}  [introspection failed: {e}]")
        return "\n".join(parts)

    # --------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------
    def _check_tables(self, sql: str) -> None:
        # Very permissive: extract candidate identifiers after FROM/JOIN and
        # ensure each is in our allow-list. Aliases and subqueries pass
        # because the referenced base tables still show up.
        tokens = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)", sql, re.IGNORECASE)
        for tok in tokens:
            base = tok.split(".")[-1].strip('"').lower()
            if base not in _ALLOWED_TABLES and base not in {"(", ""}:
                raise SQLValidationError(f"Reference to unknown table: {tok}")

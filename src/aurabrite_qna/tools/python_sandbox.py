"""AST-validated Python execution sandbox.

We deliberately avoid `exec(...)` on raw code. Instead, we parse the code,
walk the AST, reject anything unsafe (imports, attribute access on dunders,
network / filesystem calls) and only then compile and run in a locked-down
namespace.

This is *not* a security boundary against an adversarial attacker — it is a
defence-in-depth measure that eliminates the common footguns of LLM-generated
Python (accidental `os.system`, imports of `requests`, etc.).
"""

from __future__ import annotations

import ast
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "print",
}

_FORBIDDEN_NAMES = {"__import__", "eval", "exec", "compile", "open", "input", "globals",
                    "locals", "vars", "dir", "help", "getattr", "setattr", "delattr",
                    "breakpoint", "memoryview", "id"}


class SandboxError(RuntimeError):
    """Raised when validation or execution fails."""


@dataclass
class SandboxResult:
    result: Any
    stdout: str
    variables: dict[str, Any]
    error: str | None = None
    ast_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "result": _safe(self.result),
            "stdout": self.stdout,
            "variables": {k: _safe(v) for k, v in self.variables.items() if not k.startswith("_")},
            "error": self.error,
            "ast_notes": self.ast_notes,
        }


def _safe(v: Any) -> Any:
    """Best-effort JSON-friendly coercion."""
    try:
        import json

        json.dumps(v)
        return v
    except TypeError:
        try:
            return list(v)  # generators, sets
        except TypeError:
            return repr(v)


class PythonSandbox:
    """Validate + run a Python snippet in a restricted namespace."""

    def __init__(self) -> None:
        self._safe_globals = self._build_safe_globals()

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    def validate(self, code: str) -> list[str]:
        notes: list[str] = []
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise SandboxError(f"Syntax error: {e}") from e

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SandboxError("`import` is not permitted inside the sandbox.")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise SandboxError(f"Attribute access on `{node.attr}` is not permitted.")
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                raise SandboxError(f"Reference to forbidden name `{node.id}`.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_NAMES:
                    raise SandboxError(f"Call to forbidden function `{node.func.id}`.")
        notes.append(f"AST OK: {sum(1 for _ in ast.walk(tree))} nodes, no violations.")
        return notes

    # --------------------------------------------------------------
    # Execution
    # --------------------------------------------------------------
    def run(self, code: str, variables: dict[str, Any] | None = None) -> SandboxResult:
        notes = self.validate(code)

        import io
        import contextlib

        local_ns: dict[str, Any] = dict(variables or {})
        stdout = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, "<sandbox>", "exec"), self._safe_globals, local_ns)  # noqa: S102
        except SandboxError:
            raise
        except Exception as e:
            return SandboxResult(
                result=None,
                stdout=stdout.getvalue(),
                variables=local_ns,
                error=f"{type(e).__name__}: {e}",
                ast_notes=notes,
            )

        return SandboxResult(
            result=local_ns.get("result"),
            stdout=stdout.getvalue(),
            variables=local_ns,
            error=None,
            ast_notes=notes,
        )

    # --------------------------------------------------------------
    # Namespace
    # --------------------------------------------------------------
    def _build_safe_globals(self) -> dict[str, Any]:
        builtins = {name: __builtins__[name] if isinstance(__builtins__, dict)
                    else getattr(__builtins__, name)
                    for name in _ALLOWED_BUILTINS}
        return {
            "__builtins__": builtins,
            "math": math,
            "statistics": statistics,
        }

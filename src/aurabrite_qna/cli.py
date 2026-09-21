"""Command-line interface for the AuraBrite QnA system.

    aurabrite init       # regenerate the warehouse + documents + RAG index
    aurabrite ask "..."  # run a single question
    aurabrite chat       # interactive loop
    aurabrite eval       # run the offline evaluation suite
    aurabrite ui         # launch the Streamlit UI
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agents import build_orchestrator
from .config import SETTINGS
from .data import connect
from .data.generator import generate_all
from .rag import HybridRetriever

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    seed: int = typer.Option(20260921, help="RNG seed for reproducible data."),
    force: bool = typer.Option(False, help="Regenerate even if data exists."),
) -> None:
    """Generate synthetic warehouse + documents and build the RAG index."""
    if SETTINGS.warehouse_path.exists() and not force:
        console.print(
            f"[yellow]Warehouse already exists at {SETTINGS.warehouse_path}. Use --force to overwrite.[/yellow]"
        )

    console.print(Panel.fit("Generating synthetic AuraBrite warehouse + documents ...", style="bold blue"))
    with connect(SETTINGS.warehouse_path) as wh:
        report = generate_all(wh, SETTINGS.docs_dir, seed=seed)

    tbl = Table(title="Warehouse tables")
    tbl.add_column("Table"); tbl.add_column("Rows", justify="right")
    for t in report.tables:
        tbl.add_row(t.name, f"{t.rows:,}")
    console.print(tbl)
    console.print(f"Wrote {len(report.documents)} documents to {SETTINGS.docs_dir}")

    console.print("Building hybrid RAG index ...")
    HybridRetriever.rebuild(SETTINGS.docs_dir, SETTINGS.vector_dir)
    console.print("[green]✔ Ready.[/green]")


# ---------------------------------------------------------------------------
# ask / chat
# ---------------------------------------------------------------------------

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    show_trace: bool = typer.Option(True, help="Print the agent trace."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON only."),
) -> None:
    """Answer a single question and print the result."""
    orc = build_orchestrator()
    state = orc.answer(question)
    if json_out:
        typer.echo(json.dumps(state.to_dict(), indent=2, default=str))
        return
    console.print(Markdown(state.answer))
    if show_trace:
        _render_trace(state)


@app.command()
def chat() -> None:
    """Interactive REPL — Ctrl-C or 'exit' to quit."""
    orc = build_orchestrator()
    console.print(Panel("AuraBrite QnA · type 'exit' to quit", style="bold green"))
    while True:
        try:
            q = console.input("[bold cyan]?[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        state = orc.answer(q)
        console.print(Markdown(state.answer))
        _render_trace(state)


def _render_trace(state) -> None:
    tbl = Table(title="Agent trace", show_lines=False)
    tbl.add_column("Node")
    tbl.add_column("Latency (ms)", justify="right")
    tbl.add_column("Summary")
    for t in state.trace:
        tbl.add_row(t.node, f"{t.latency_ms:.1f}", t.output_summary[:120])
    console.print(tbl)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

@app.command()
def eval(
    dataset: Optional[Path] = typer.Option(
        None, help="Path to gold Q&A JSON. Defaults to bundled gold set."
    ),
    output: Optional[Path] = typer.Option(None, help="Write per-question results to JSON."),
) -> None:
    """Run the offline evaluation suite and print summary metrics."""
    from .eval.runner import run_evaluation

    report = run_evaluation(dataset_path=dataset, output_path=output)
    tbl = Table(title="Evaluation summary")
    tbl.add_column("Metric"); tbl.add_column("Value", justify="right")
    for k, v in report.summary.items():
        tbl.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
    console.print(tbl)
    if output:
        console.print(f"[dim]Full report → {output}[/dim]")


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------

@app.command()
def ui() -> None:
    """Launch the Streamlit UI (blocking)."""
    from . import ui as ui_pkg  # noqa: F401
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


if __name__ == "__main__":
    app()

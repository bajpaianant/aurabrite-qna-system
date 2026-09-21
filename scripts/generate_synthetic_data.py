#!/usr/bin/env python3
"""Regenerate the AuraBrite synthetic warehouse and narrative documents.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --seed 42 --warehouse ./my.duckdb
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script from a fresh clone without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aurabrite_qna.config import SETTINGS  # noqa: E402
from aurabrite_qna.data import connect  # noqa: E402
from aurabrite_qna.data.generator import generate_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--warehouse", type=Path, default=SETTINGS.warehouse_path)
    parser.add_argument("--docs-dir", type=Path, default=SETTINGS.docs_dir)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("generator")

    log.info("Warehouse : %s", args.warehouse)
    log.info("Docs dir  : %s", args.docs_dir)
    log.info("Seed      : %d", args.seed)

    with connect(args.warehouse) as wh:
        report = generate_all(warehouse=wh, docs_dir=args.docs_dir, seed=args.seed)

    print()
    print(report.as_text())
    print()
    print("Done. Load the warehouse with:")
    print(f"  duckdb {args.warehouse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

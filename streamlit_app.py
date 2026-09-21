"""Streamlit Community Cloud entrypoint.

Cloud auto-detects `streamlit_app.py` at the repo root. We keep this file
tiny — it just puts `src/` on the import path and delegates to the real UI
module. Locally you can still launch the app either way:

    streamlit run streamlit_app.py
    streamlit run src/aurabrite_qna/ui/streamlit_app.py
    aurabrite ui
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Delegate to the real Streamlit app.
from aurabrite_qna.ui.streamlit_app import main  # noqa: E402

main()

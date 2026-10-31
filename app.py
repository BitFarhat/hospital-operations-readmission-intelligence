"""Optional Streamlit entry point. Prefer: streamlit run src/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import app as dashboard  # noqa: E402  (src/app.py)

if __name__ == "__main__":
    dashboard.main()

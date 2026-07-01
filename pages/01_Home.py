"""Home page."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path
from typing import Callable


def _load_home_main() -> Callable[[], None]:
    """Load the root Streamlit app entry point without importing the app package.

    Returns:
        Callable that renders the home page.
    """
    root_app_path = Path(__file__).resolve().parents[1] / "app.py"
    root_app_globals = run_path(str(root_app_path))
    return root_app_globals["main"]


_load_home_main()()

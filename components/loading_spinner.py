"""Loading spinner component."""

from __future__ import annotations

import streamlit as st


def render_loading(message: str = "Processing", progress: int = 0) -> None:
    """Render loading status and progress bar.

    Args:
        message: Status message.
        progress: Progress value from 0 to 100.

    Returns:
        None.
    """
    st.info(message)
    st.progress(max(0, min(progress, 100)))


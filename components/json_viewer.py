"""JSON viewer component."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_json_viewer(data: Any, title: str = "JSON", expanded: bool = False) -> None:
    """Render expandable JSON data.

    Args:
        data: JSON-serializable data.
        title: Viewer title.
        expanded: Whether the viewer should be expanded.

    Returns:
        None.
    """
    with st.expander(title, expanded=expanded):
        st.json(data, expanded=expanded)
        st.button("Copy", key=f"copy_{title.replace(' ', '_').lower()}", disabled=True)


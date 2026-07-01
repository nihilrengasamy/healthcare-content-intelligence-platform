"""Status badge components."""

from __future__ import annotations

import streamlit as st

from utils.constants import PLACEHOLDER_STATUS


def render_status_badge(status: str) -> None:
    """Render a colored status badge.

    Args:
        status: Status label.

    Returns:
        None.
    """
    color = PLACEHOLDER_STATUS.get(status, "#6b7280")
    st.markdown(
        f'<span class="hcip-badge" style="background:{color};">{status}</span>',
        unsafe_allow_html=True,
    )


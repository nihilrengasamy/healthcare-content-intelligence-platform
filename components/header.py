"""Header component."""

from __future__ import annotations

from textwrap import dedent

import streamlit as st

from config.settings import APP_STAGE


def render_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page header.

    Args:
        title: Page title.
        subtitle: Page subtitle.

    Returns:
        None.
    """
    st.markdown(
        dedent(
            f"""
            <div class="hcip-header">
                <div class="hcip-kicker">{APP_STAGE}</div>
                <h1 class="hcip-title">{title}</h1>
                <div class="hcip-subtitle">{subtitle}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

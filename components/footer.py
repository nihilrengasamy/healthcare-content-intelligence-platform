"""Footer component."""

from __future__ import annotations

from textwrap import dedent

import streamlit as st

from config.settings import APP_NAME, APP_VERSION


def render_footer() -> None:
    """Render application footer.

    Returns:
        None.
    """
    st.markdown(
        dedent(
            f"""
            <div class="hcip-footer">
                {APP_NAME} · Version {APP_VERSION} · Frontend framework
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

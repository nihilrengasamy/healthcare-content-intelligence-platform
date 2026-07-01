"""Sidebar component."""

from __future__ import annotations

import platform

import streamlit as st

from config.settings import APP_NAME, APP_STAGE, BACKEND_STATUS
from components.status_badges import render_status_badge
from utils.session_manager import SessionManager


def render_sidebar(current_page: str) -> None:
    """Render the professional application sidebar.

    Args:
        current_page: Current page label.

    Returns:
        None.
    """
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(APP_STAGE)
        st.divider()
        st.markdown("#### Current Page")
        st.write(current_page)
        st.markdown("#### Application Status")
        render_status_badge("Ready")
        st.markdown("#### Backend Status")
        st.write(BACKEND_STATUS)
        st.divider()
        st.markdown("#### System Information")
        st.write(f"Python: {platform.python_version()}")
        st.write(f"Runtime: Streamlit")
        st.divider()
        if st.button("Reset Session State"):
            SessionManager.reset()
            st.success("Session state reset.")
        with st.expander("Session Keys", expanded=False):
            for key, status in SessionManager.get_status_summary().items():
                st.write(f"{key}: {status}")


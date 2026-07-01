"""About page."""

from __future__ import annotations

import streamlit as st

from components.footer import render_footer
from components.header import render_header
from components.sidebar import render_sidebar
from config.settings import APP_NAME, APP_STAGE, APP_VERSION
from utils.session_manager import SessionManager
from utils.theme import apply_theme


st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("About")
render_header("About", "Platform scope, architecture, and project status.")
st.markdown(
    f"""
    <div class="hcip-card">
        <h3>{APP_NAME}</h3>
        <p><strong>Version:</strong> {APP_VERSION}</p>
        <p><strong>Stage:</strong> {APP_STAGE}</p>
        <p>
        This frontend framework provides the navigable enterprise UI shell for
        healthcare content intelligence workflows. Backend wiring is reserved
        for Phase 4B.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
render_footer()


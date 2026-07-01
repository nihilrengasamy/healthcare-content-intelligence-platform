"""Reusable page helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.status_badges import render_status_badge


def render_placeholder_page(
    title: str,
    description: str,
    purpose: str,
    session_key: str,
) -> None:
    """Render a consistent workflow page shell.

    Args:
        title: Page title.
        description: Page description.
        purpose: Page purpose.
        session_key: Related session state key.

    Returns:
        None.
    """
    render_header(title, description)
    left, right = st.columns([0.68, 0.32], gap="large")

    with left:
        st.markdown("#### Purpose")
        st.info(purpose)
        if st.button("Prepare Workspace", key=f"{session_key}_prepare"):
            st.session_state[session_key] = {
                "status": "prepared",
                "message": f"{title} workspace initialized.",
            }
            st.success("Workspace prepared.")

        st.markdown("#### Results")
        state_value: Any = st.session_state.get(session_key)
        if state_value:
            render_json_viewer(state_value, title="Session Preview", expanded=True)
        else:
            st.empty()
            st.caption("No results are available yet.")

    with right:
        st.markdown("#### Status")
        render_status_badge("Ready")
        st.write("Workflow page is available.")
        st.markdown("#### Information")
        st.write("Use the page action buttons to run the configured workflow.")
        render_json_viewer(
            {
                "session_key": session_key,
                "backend_integration": "available",
                "page_status": "ready",
            },
            title="Page Metadata",
            expanded=False,
        )

    render_footer()


def render_home_metrics() -> None:
    """Render homepage metrics.

    Returns:
        None.
    """
    render_metric_row(
        [
            {"title": "Documents Uploaded", "value": "0", "delta": "Ready", "icon": "DOC", "color": "#1455a0"},
            {"title": "Policies Processed", "value": "0", "delta": "Waiting", "icon": "POL", "color": "#047857"},
            {"title": "Rules Extracted", "value": "0", "delta": "Waiting", "icon": "RUL", "color": "#7c3aed"},
            {"title": "Claims Evaluated", "value": "0", "delta": "Waiting", "icon": "CLM", "color": "#b45309"},
        ]
    )


def sample_activity_frame() -> list[dict[str, str]]:
    """Return representative workflow activity data.

    Returns:
        Representative workflow rows.
    """
    return [
        {"Stage": "Upload", "Status": "Ready", "Owner": "Frontend"},
        {"Stage": "Classification", "Status": "Ready", "Owner": "AI Modules"},
        {"Stage": "RAG", "Status": "Ready", "Owner": "AI Modules"},
        {"Stage": "Decision Support", "Status": "Ready", "Owner": "AI Modules"},
    ]

"""Streamlit entry point for the Healthcare Content Intelligence Platform."""

from __future__ import annotations

from textwrap import dedent

import streamlit as st

from components.chart_helpers import render_bar_chart
from components.footer import render_footer
from components.header import render_header
from components.sidebar import render_sidebar
from components.table_view import render_table
from config.settings import APP_NAME, LAYOUT, PAGE_ICON, SIDEBAR_STATE
from utils.constants import FEATURE_CARDS
from utils.helpers import render_home_metrics, sample_activity_frame
from utils.session_manager import SessionManager
from utils.theme import apply_theme


def configure_app() -> None:
    """Configure Streamlit page settings.

    Returns:
        None.
    """
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE,
    )
    apply_theme()
    SessionManager.initialize()


def render_workflow_diagram() -> None:
    """Render high-level workflow diagram.

    Returns:
        None.
    """
    st.markdown(
        """
        ```mermaid
        flowchart LR
            A["Upload"] --> B["Classify"]
            B --> C["Summarize"]
            C --> D["Retrieve"]
            D --> E["Extract Rules"]
            E --> F["Evaluate Claims"]
            F --> G["Explain"]
            G --> H["Evaluate Quality"]
        ```
        """
    )


def render_feature_cards() -> None:
    """Render homepage feature cards.

    Returns:
        None.
    """
    columns = st.columns(4)
    for column, (title, description) in zip(columns, FEATURE_CARDS):
        with column:
            st.markdown(
                dedent(
                    f"""
                    <div class="hcip-card">
                        <h4>{title}</h4>
                        <p class="hcip-muted">{description}</p>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )


def main() -> None:
    """Render the Streamlit application homepage.

    Returns:
        None.
    """
    configure_app()
    render_sidebar("Home")
    render_header(
        "Healthcare Content Intelligence Platform",
        "Enterprise AI framework for healthcare content management, policy intelligence, and decision support.",
    )
    render_home_metrics()
    st.markdown("### Architecture Overview")
    render_workflow_diagram()
    st.markdown("### System Capabilities")
    render_feature_cards()
    st.markdown("### Workflow Status")
    status_df = sample_activity_frame()
    render_table(status_df, title="Workflow Readiness", search=False)
    chart_data = [
        {"Capability": "Documents", "Count": 0},
        {"Capability": "Policies", "Count": 0},
        {"Capability": "Rules", "Count": 0},
        {"Capability": "Claims", "Count": 0},
    ]
    render_bar_chart(chart_data, x="Capability", y="Count", title="Placeholder Activity")
    render_footer()


if __name__ == "__main__":
    main()

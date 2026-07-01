"""Metric card components."""

from __future__ import annotations

from textwrap import dedent

import streamlit as st


def render_metric_card(
    title: str,
    value: str,
    delta: str = "",
    icon: str = "",
    color: str = "#1455a0",
) -> None:
    """Render one metric card.

    Args:
        title: Metric title.
        value: Metric value.
        delta: Metric delta or status.
        icon: Optional icon.
        color: Accent color.

    Returns:
        None.
    """
    st.markdown(
        dedent(
            f"""
            <div class="hcip-metric" style="border-top: 4px solid {color};">
                <div class="hcip-metric-label">{icon} {title}</div>
                <div class="hcip-metric-value">{value}</div>
                <div class="hcip-muted">{delta}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list[dict[str, str]]) -> None:
    """Render a row of metric cards.

    Args:
        metrics: Metric configuration dictionaries.

    Returns:
        None.
    """
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        with column:
            render_metric_card(
                title=metric.get("title", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta", ""),
                icon=metric.get("icon", ""),
                color=metric.get("color", "#1455a0"),
            )

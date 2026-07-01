"""Reusable chart helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import plotly.graph_objects as go
import streamlit as st


def _records(dataframe: Any) -> list[dict[str, Any]]:
    """Convert simple table-like inputs into row dictionaries.

    Args:
        dataframe: DataFrame-like object or iterable of mappings.

    Returns:
        List of row dictionaries.
    """
    if dataframe is None:
        return []
    if hasattr(dataframe, "to_dict"):
        try:
            return list(dataframe.to_dict(orient="records"))
        except TypeError:
            return list(dataframe.to_dict("records"))
    if isinstance(dataframe, Mapping):
        return [dict(dataframe)]
    if isinstance(dataframe, Iterable) and not isinstance(dataframe, (str, bytes)):
        return [dict(row) for row in dataframe if isinstance(row, Mapping)]
    return []


def _column(rows: list[dict[str, Any]], key: str) -> list[Any]:
    """Extract a chart column from row dictionaries."""
    return [row.get(key, 0) for row in rows]


def _show_or_empty(rows: list[dict[str, Any]], title: str) -> bool:
    """Render an empty chart message when no rows are available."""
    if rows:
        return False
    st.info(f"No data available for {title}.")
    return True


def render_bar_chart(dataframe: Any, x: str, y: str, title: str) -> None:
    """Render a bar chart.

    Args:
        dataframe: Source table-like data.
        x: X-axis column.
        y: Y-axis column.
        title: Chart title.

    Returns:
        None.
    """
    rows = _records(dataframe)
    if _show_or_empty(rows, title):
        return
    fig = go.Figure(data=[go.Bar(x=_column(rows, x), y=_column(rows, y))])
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    st.plotly_chart(fig, use_container_width=True)


def render_pie_chart(dataframe: Any, names: str, values: str, title: str) -> None:
    """Render a pie chart.

    Args:
        dataframe: Source table-like data.
        names: Label column.
        values: Value column.
        title: Chart title.

    Returns:
        None.
    """
    rows = _records(dataframe)
    if _show_or_empty(rows, title):
        return
    fig = go.Figure(data=[go.Pie(labels=_column(rows, names), values=_column(rows, values))])
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)


def render_line_chart(dataframe: Any, x: str, y: str, title: str) -> None:
    """Render a line chart.

    Args:
        dataframe: Source table-like data.
        x: X-axis column.
        y: Y-axis column.
        title: Chart title.

    Returns:
        None.
    """
    rows = _records(dataframe)
    if _show_or_empty(rows, title):
        return
    fig = go.Figure(data=[go.Scatter(x=_column(rows, x), y=_column(rows, y), mode="lines+markers")])
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    st.plotly_chart(fig, use_container_width=True)


def render_histogram(dataframe: Any, x: str, title: str) -> None:
    """Render a histogram.

    Args:
        dataframe: Source table-like data.
        x: Histogram value column.
        title: Chart title.

    Returns:
        None.
    """
    rows = _records(dataframe)
    if _show_or_empty(rows, title):
        return
    fig = go.Figure(data=[go.Histogram(x=_column(rows, x))])
    fig.update_layout(title=title, xaxis_title=x, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

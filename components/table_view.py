"""Reusable table component."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping
from typing import Any

import streamlit as st


def _records_from_table(dataframe: Any) -> list[dict[str, Any]]:
    """Convert common table-like inputs into row dictionaries.

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


def _to_csv(records: list[dict[str, Any]]) -> str:
    """Serialize table records to CSV text.

    Args:
        records: Row dictionaries.

    Returns:
        CSV text.
    """
    if not records:
        return ""

    buffer = io.StringIO()
    fieldnames = list(dict.fromkeys(key for row in records for key in row))
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def _to_markdown(records: list[dict[str, Any]]) -> str:
    """Serialize records to a Markdown table.

    Args:
        records: Row dictionaries.

    Returns:
        Markdown table text.
    """
    if not records:
        return ""

    headers = list(dict.fromkeys(key for row in records for key in row))
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    rows = [
        "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"
        for row in records
    ]
    return "\n".join([header_line, separator, *rows])


def render_table(
    dataframe: Any,
    title: str = "Table",
    search: bool = True,
    download: bool = True,
) -> None:
    """Render a searchable table with CSV download.

    Args:
        dataframe: Table-like object to render.
        title: Table title.
        search: Whether to show search box.
        download: Whether to show CSV download.

    Returns:
        None.
    """
    st.markdown(f"#### {title}")
    records = _records_from_table(dataframe)
    if not records:
        st.info("No table data available.")
        return

    view = records
    if search:
        query = st.text_input("Search", key=f"search_{title}").strip().lower()
        if query:
            view = [
                row
                for row in records
                if any(query in str(value).lower() for value in row.values())
            ]

    st.markdown(_to_markdown(view))
    if download:
        st.download_button(
            "Download CSV",
            _to_csv(view),
            file_name=f"{title.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

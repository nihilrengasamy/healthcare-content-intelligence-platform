"""Version comparison page."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from utils import pandas_compat as pd
import streamlit as st

from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.sidebar import render_sidebar
from components.table_view import render_table
from utils.runtime_mode import is_low_memory_demo_mode, trim_documents_for_demo
from utils.session_manager import SessionManager
from utils.theme import apply_theme


def _change_label(value: dict[str, Any]) -> str:
    """Return a readable label for a structured change row."""
    old_text = str(value.get("old", "")).strip()
    new_text = str(value.get("new", "")).strip()

    if old_text and new_text:
        return f"{old_text} -> {new_text}"
    if new_text:
        return new_text
    if old_text:
        return old_text
    return str(value.get("meaning", "Policy wording changed.")).strip()


def _save_upload(uploaded_file: Any) -> Path:
    """Save an uploaded comparison file temporarily."""
    directory = Path(tempfile.mkdtemp(prefix="hcip_compare_"))
    path = directory / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return path


def _load_pdf(uploaded_file: Any) -> list[Any]:
    """Load PDF pages with PDFLoader."""
    from modules.pdf_loader import PDFLoader

    documents = PDFLoader().load_pdf(_save_upload(uploaded_file))
    if is_low_memory_demo_mode():
        return trim_documents_for_demo(documents)
    return documents


def _changes_frame(report: dict[str, Any], field: str) -> pd.DataFrame:
    """Build DataFrame for report change sections."""
    values = report.get(field, [])
    rows = []
    for value in values:
        if isinstance(value, dict):
            row = {
                "change": _change_label(value),
                "meaning": value.get("meaning", ""),
                "new": value.get("new", ""),
                "old": value.get("old", ""),
                "similarity": value.get("similarity", ""),
            }
            rows.append(row)
        else:
            rows.append({"change": value})
    return pd.DataFrame(rows)


st.set_page_config(page_title="Version Comparison", page_icon="COMPARE", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Version Comparison")
render_header(
    "Version Comparison",
    "Compare old and new healthcare policy versions and identify meaningful content changes.",
)

if is_low_memory_demo_mode():
    st.info(
        "Hosted demo mode is active. Version comparison uses a smaller page window from each PDF "
        "to stay responsive in the cloud."
    )

col1, col2 = st.columns(2)
with col1:
    old_file = st.file_uploader("Old policy PDF", type=["pdf"], key="old_policy_pdf")
with col2:
    new_file = st.file_uploader("New policy PDF", type=["pdf"], key="new_policy_pdf")

try:
    from modules.prompt_manager import PromptManager

    with st.expander("Comparison Prompt Template", expanded=False):
        st.code(PromptManager().get_prompt("version_comparison"))
except Exception as error:
    st.info(f"Prompt manager unavailable: {error}")

if old_file and new_file and st.button("Compare Versions", type="primary"):
    try:
        from modules.compare_versions import PolicyVersionComparator

        with st.spinner("Loading and comparing policies"):
            old_docs = _load_pdf(old_file)
            new_docs = _load_pdf(new_file)
            report = PolicyVersionComparator().compare_documents(old_docs, new_docs)
        st.session_state["comparison"] = report
        st.success("Version comparison completed.")
    except Exception as error:
        st.error(f"Version comparison failed: {error}")

comparison = st.session_state.get("comparison")
if comparison:
    render_metric_row(
        [
            {"title": "Added", "value": str(len(comparison.get("added_sections", []))), "delta": "Sections", "icon": "ADD", "color": "#1455a0"},
            {"title": "Removed", "value": str(len(comparison.get("removed_sections", []))), "delta": "Sections", "icon": "DEL", "color": "#b91c1c"},
            {"title": "Modified", "value": str(len(comparison.get("modified_sections", []))), "delta": "Sections", "icon": "MOD", "color": "#b45309"},
            {"title": "Critical", "value": str(len(comparison.get("critical_changes", []))), "delta": "Changes", "icon": "HIGH", "color": "#7c3aed"},
        ]
    )
    st.markdown("#### Summary of Changes")
    st.info(comparison.get("overall_summary", ""))
    tabs = st.tabs(["Added", "Removed", "Modified", "Clinical", "Billing", "Coding", "JSON"])
    for tab, field in zip(
        tabs[:-1],
        ["added_sections", "removed_sections", "modified_sections", "clinical_changes", "billing_changes", "coding_changes"],
    ):
        with tab:
            frame = _changes_frame(comparison, field)
            if frame.empty:
                st.info("No changes in this category.")
            else:
                render_table(frame, title=field.replace("_", " ").title(), search=True)
    with tabs[-1]:
        render_json_viewer(comparison, "Comparison JSON", expanded=True)
else:
    st.info("Upload an old and new policy PDF to compare versions.")

render_footer()



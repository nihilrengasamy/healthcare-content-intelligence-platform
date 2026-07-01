"""Document classification page."""

from __future__ import annotations

from typing import Any

from utils import pandas_compat as pd
import streamlit as st

from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.sidebar import render_sidebar
from components.table_view import render_table
from utils.session_manager import SessionManager
from utils.theme import apply_theme


def _all_page_documents(payloads: list[dict[str, Any]]) -> list[Any]:
    """Return all page-level documents from uploaded document payloads."""
    documents: list[Any] = []
    for payload in payloads:
        documents.extend(payload.get("documents", []))
    return documents


def _score_frame(classification: dict[str, Any]) -> pd.DataFrame:
    """Build keyword score DataFrame."""
    scores = classification.get("scores", {})
    return pd.DataFrame(
        [{"Document Type": key, "Score": value} for key, value in scores.items()]
    )


st.set_page_config(page_title="Document Classification", page_icon="CLASS", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Document Classification")
render_header(
    "Document Classification",
    "Classify uploaded healthcare content into billing, clinical, contract, coverage, regulatory, or unknown categories.",
)

payloads = st.session_state.get("uploaded_documents", [])
if not payloads:
    st.warning("Upload and extract PDFs before running classification.")
else:
    if st.button("Classify Uploaded Documents", type="primary"):
        try:
            from modules.document_classifier import HealthcareDocumentClassifier

            classifier = HealthcareDocumentClassifier()
            with st.spinner("Classifying document pages"):
                classification = classifier.classify_documents(_all_page_documents(payloads))
            st.session_state["document_classification"] = classification
            st.success("Classification completed.")
        except Exception as error:
            st.error(f"Classification failed: {error}")

classification = st.session_state.get("document_classification")
if classification:
    render_metric_row(
        [
            {"title": "Document Type", "value": classification.get("document_type", "unknown"), "delta": "Aggregate", "icon": "TYPE", "color": "#1455a0"},
            {"title": "Confidence", "value": f"{classification.get('confidence', 0):.2f}", "delta": "Weighted", "icon": "CONF", "color": "#047857"},
            {"title": "Pages", "value": str(len(classification.get("page_classifications", []))), "delta": "Classified", "icon": "PAGE", "color": "#7c3aed"},
            {"title": "Signals", "value": str(len(classification.get("signals", []))), "delta": "Detected", "icon": "SIG", "color": "#b45309"},
        ]
    )
    st.markdown("#### Reason")
    st.info(classification.get("reason", ""))
    st.markdown("#### Signals")
    st.write(classification.get("signals", []))
    if classification.get("page_classifications"):
        first_page = classification["page_classifications"][0]
        render_table(_score_frame(first_page), title="Keyword Scores", search=False)
    render_json_viewer(classification, "Classification JSON", expanded=False)
else:
    st.info("Classification results will appear here.")

render_footer()



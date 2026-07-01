"""Document upload and PDF extraction page."""

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
from utils.runtime_mode import is_low_memory_demo_mode, runtime_mode_label, trim_documents_for_demo
from utils.session_manager import SessionManager
from utils.theme import apply_theme


def _store_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded file to a temporary PDF path.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Path to the temporary file.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="hcip_upload_"))
    file_path = temp_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def _extract_pdf(uploaded_file: Any) -> dict[str, Any]:
    """Extract one uploaded PDF using the backend PDFLoader.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Document payload for session state.
    """
    from modules.pdf_loader import PDFLoader

    loader = PDFLoader()
    file_path = _store_uploaded_file(uploaded_file)
    documents = loader.load_pdf(file_path)
    hosted_demo = is_low_memory_demo_mode()
    if hosted_demo:
        documents = trim_documents_for_demo(documents)
    chunks = loader.split_documents(documents)
    metadata = loader.extract_metadata(file_path)
    statistics = loader.get_statistics(chunks)
    if hosted_demo:
        statistics["mode"] = runtime_mode_label()
        statistics["full_page_count"] = metadata.get("pages", 0)
        statistics["processed_pages"] = len(documents)
    return {
        "filename": uploaded_file.name,
        "source_path": str(file_path),
        "documents": documents,
        "chunks": chunks,
        "metadata": metadata,
        "statistics": statistics,
        "preview": documents[0].page_content[:1500] if documents else "",
    }


def _metadata_frame(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Build metadata table for uploaded documents.

    Args:
        payloads: Uploaded document payloads.

    Returns:
        Metadata DataFrame.
    """
    rows = []
    for payload in payloads:
        metadata = payload.get("metadata", {})
        rows.append(
            {
                "Filename": payload.get("filename", ""),
                "Pages": metadata.get("pages", 0),
                "Chunks": len(payload.get("chunks", [])),
                "File Size": metadata.get("file_size", ""),
                "Title": metadata.get("title", ""),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Document Upload", page_icon="PDF", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Document Upload")
render_header(
    "Document Upload",
    "Upload healthcare PDFs, extract pages, split chunks, and store document objects for downstream AI workflows.",
)

if is_low_memory_demo_mode():
    st.info(
        "Hosted demo mode is active. The app processes a smaller page window per PDF "
        "to stay responsive on low-memory cloud infrastructure."
    )

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Phase 4B processes PDFs only. DOCX/TXT support can be added later.",
)

if uploaded_files and st.button("Extract Uploaded PDFs", type="primary"):
    extracted_payloads: list[dict[str, Any]] = []
    progress = st.progress(0)
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            with st.spinner(f"Extracting {uploaded_file.name}"):
                extracted_payloads.append(_extract_pdf(uploaded_file))
        except Exception as error:
            st.error(f"Could not process {uploaded_file.name}: {error}")
        progress.progress(index / len(uploaded_files))

    if extracted_payloads:
        st.session_state["uploaded_documents"] = extracted_payloads
        st.success(f"Processed {len(extracted_payloads)} document(s).")

payloads = st.session_state.get("uploaded_documents", [])
if not payloads:
    st.info("Upload one or more PDF files to begin.")
else:
    total_pages = sum(payload.get("metadata", {}).get("pages", 0) for payload in payloads)
    total_chunks = sum(len(payload.get("chunks", [])) for payload in payloads)
    processed_pages = sum(payload.get("statistics", {}).get("processed_pages", payload.get("metadata", {}).get("pages", 0)) for payload in payloads)
    render_metric_row(
        [
            {"title": "Documents", "value": str(len(payloads)), "delta": "Uploaded", "icon": "DOC", "color": "#1455a0"},
            {"title": "Pages", "value": str(processed_pages if is_low_memory_demo_mode() else total_pages), "delta": "Processed" if is_low_memory_demo_mode() else "Extracted", "icon": "PAGE", "color": "#047857"},
            {"title": "Chunks", "value": str(total_chunks), "delta": "Ready", "icon": "CHUNK", "color": "#7c3aed"},
            {"title": "Status", "value": "Ready", "delta": runtime_mode_label(), "icon": "OK", "color": "#b45309"},
        ]
    )
    if is_low_memory_demo_mode() and processed_pages != total_pages:
        st.caption(
            f"Hosted demo processed {processed_pages} page(s) from {total_pages} total page(s) "
            "to keep cloud memory usage low."
        )
    render_table(_metadata_frame(payloads), title="Uploaded Document Metadata", search=False)
    selected = st.selectbox("Preview document", [payload["filename"] for payload in payloads])
    selected_payload = next(payload for payload in payloads if payload["filename"] == selected)
    st.text_area("Text Preview", selected_payload.get("preview", ""), height=260)
    render_json_viewer(selected_payload.get("metadata", {}), "Document Metadata", expanded=False)

render_footer()



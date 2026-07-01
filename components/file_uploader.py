"""Reusable file upload component."""

from __future__ import annotations

import streamlit as st


def render_file_uploader(key: str = "file_uploader") -> list[object]:
    """Render a file uploader for document inputs.

    Args:
        key: Streamlit widget key.

    Returns:
        Uploaded files.
    """
    files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=key,
    )
    for uploaded_file in files:
        st.write(f"Uploaded: {uploaded_file.name}")
    return list(files)


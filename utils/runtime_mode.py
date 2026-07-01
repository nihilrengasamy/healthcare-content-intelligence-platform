"""Runtime mode helpers for local and hosted Streamlit deployments."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document


TRUE_VALUES = {"1", "true", "yes", "on"}


def is_low_memory_demo_mode() -> bool:
    """Return whether the app should run in hosted low-memory demo mode."""
    explicit_value = os.getenv("HCIP_LOW_MEMORY_DEMO", "").strip().lower()
    if explicit_value:
        return explicit_value in TRUE_VALUES

    return bool(os.getenv("RENDER")) or bool(os.getenv("RENDER_SERVICE_ID"))


def runtime_mode_label() -> str:
    """Return a human-readable runtime mode label."""
    return "Low-memory hosted demo" if is_low_memory_demo_mode() else "Full local workflow"


def trim_documents_for_demo(
    documents: list[Document],
    *,
    max_pages: int = 6,
    max_chars_per_page: int = 2200,
) -> list[Document]:
    """Trim page documents for lower-memory hosted usage.

    Args:
        documents: Page-level LangChain documents.
        max_pages: Maximum number of pages to keep.
        max_chars_per_page: Maximum number of characters to keep per page.

    Returns:
        Trimmed LangChain documents suitable for hosted demo mode.
    """
    trimmed_documents: list[Document] = []
    for document in documents[:max_pages]:
        text = str(getattr(document, "page_content", "") or "").strip()
        text = text[:max_chars_per_page].strip()
        if not text:
            continue

        trimmed_documents.append(
            Document(
                page_content=text,
                metadata=dict(getattr(document, "metadata", {}) or {}),
            )
        )
    return trimmed_documents


def trim_text_for_demo(
    text: str,
    *,
    max_chars: int = 12000,
) -> str:
    """Trim combined source text for low-memory display and processing."""
    return str(text or "").strip()[:max_chars].strip()


def summarize_uploaded_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight payload summary for diagnostics and UI messaging."""
    statistics = payload.get("statistics", {}) if isinstance(payload, dict) else {}
    return {
        "filename": payload.get("filename", ""),
        "pages": statistics.get("pages", 0),
        "chunks": statistics.get("chunks", 0),
    }

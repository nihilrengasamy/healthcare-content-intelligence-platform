"""Unit tests for the PDFLoader module."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from langchain_core.documents import Document

from modules.pdf_loader import PDFLoader


def _create_pdf(path: Path, pages: list[str], metadata: dict[str, str] | None = None) -> Path:
    """Create a small PDF for tests.

    Args:
        path: Output PDF path.
        pages: Text content for each page.
        metadata: Optional PDF metadata.

    Returns:
        Path to the created PDF.
    """
    document = fitz.open()
    if metadata:
        document.set_metadata(metadata)

    for page_text in pages:
        page = document.new_page()
        page.insert_text((72, 72), page_text)

    document.save(path)
    document.close()
    return path


def test_load_pdf_valid_pdf(tmp_path: Path) -> None:
    """Verify a valid PDF loads one LangChain document per page."""
    pdf_path = _create_pdf(
        tmp_path / "policy.pdf",
        ["Billing policy page one", "Coverage policy page two"],
    )
    loader = PDFLoader()

    documents = loader.load_pdf(pdf_path)

    assert len(documents) == 2
    assert all(isinstance(document, Document) for document in documents)
    assert documents[0].metadata["filename"] == "policy.pdf"
    assert documents[0].metadata["page_number"] == 1
    assert documents[0].metadata["source"] == str(pdf_path)


def test_load_pdf_invalid_pdf_returns_empty_list(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify corrupted PDFs are handled without crashing."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_text("not a real PDF", encoding="utf-8")
    loader = PDFLoader()

    documents = loader.load_pdf(pdf_path)

    assert documents == []
    assert "Failed to load PDF" in caplog.text


def test_load_multiple_pdfs_ignores_non_pdf_files(tmp_path: Path) -> None:
    """Verify a directory load combines PDF pages and ignores non-PDF files."""
    _create_pdf(tmp_path / "policy_a.pdf", ["A page one"])
    _create_pdf(tmp_path / "policy_b.pdf", ["B page one", "B page two"])
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    loader = PDFLoader()

    documents = loader.load_multiple_pdfs(tmp_path)

    assert len(documents) == 3
    assert {document.metadata["filename"] for document in documents} == {
        "policy_a.pdf",
        "policy_b.pdf",
    }


def test_extract_metadata_uses_pymupdf_metadata(tmp_path: Path) -> None:
    """Verify metadata extraction returns file and PDF metadata fields."""
    pdf_path = _create_pdf(
        tmp_path / "contract.pdf",
        ["Contract text"],
        metadata={"title": "Contract Title", "author": "Cotiviti Demo"},
    )
    loader = PDFLoader()

    metadata = loader.extract_metadata(pdf_path)

    assert metadata["filename"] == "contract.pdf"
    assert metadata["pages"] == 1
    assert metadata["file_size"]
    assert metadata["title"] == "Contract Title"
    assert metadata["author"] == "Cotiviti Demo"
    assert "creation_date" in metadata


def test_split_documents_chunks_long_content() -> None:
    """Verify long documents are split into overlapping chunks."""
    loader = PDFLoader(chunk_size=1000, chunk_overlap=200)
    documents = [
        Document(
            page_content="word " * 500,
            metadata={"filename": "policy.pdf", "page_number": 1, "source": "policy.pdf"},
        )
    ]

    chunks = loader.split_documents(documents)

    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)
    assert chunks[0].metadata["filename"] == "policy.pdf"


def test_get_statistics_counts_documents_pages_chunks_and_words() -> None:
    """Verify statistics are calculated from LangChain documents."""
    loader = PDFLoader()
    documents = [
        Document(
            page_content="one two three",
            metadata={"source": "a.pdf", "page_number": 1},
        ),
        Document(
            page_content="four five",
            metadata={"source": "a.pdf", "page_number": 1, "chunk_index": 2},
        ),
        Document(
            page_content="six",
            metadata={"source": "a.pdf", "page_number": 2},
        ),
    ]

    statistics = loader.get_statistics(documents)

    assert statistics == {
        "documents": 3,
        "pages": 2,
        "chunks": 3,
        "average_words": 2,
        "total_words": 6,
    }


def test_missing_pdf_returns_empty_list(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify missing files are logged and skipped."""
    loader = PDFLoader()

    documents = loader.load_pdf(tmp_path / "missing.pdf")

    assert documents == []
    assert "does not exist" in caplog.text


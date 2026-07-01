"""PDF loading utilities for healthcare content intelligence workflows.

This module loads PDF documents from a healthcare content repository and
converts each readable page into a LangChain ``Document`` object for downstream
summarization, version comparison, embedding, and retrieval workflows.

Example:
    ```python
    loader = PDFLoader()

    docs = loader.load_multiple_pdfs("data/")
    chunks = loader.split_documents(docs)
    stats = loader.get_statistics(chunks)

    print(stats)
    ```
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFLoader:
    """Loads healthcare PDFs and converts them into LangChain documents."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the PDF loader.

        Args:
            chunk_size: Maximum number of characters per chunk when splitting
                documents.
            chunk_overlap: Number of overlapping characters between adjacent
                chunks.
            logger: Optional logger instance. If not provided, a module-level
                logger is used.

        Returns:
            None.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger or logging.getLogger(__name__)

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """Load a PDF and convert each readable page into a LangChain document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list of LangChain ``Document`` objects, one per readable page.
            Returns an empty list when the file is missing, encrypted,
            corrupted, empty, or otherwise unreadable.
        """
        path = Path(file_path)
        if not self._is_readable_pdf_path(path):
            return []

        try:
            with fitz.open(path) as pdf_document:
                if self._is_encrypted(pdf_document, path):
                    return []

                if pdf_document.page_count == 0:
                    self.logger.warning("Empty PDF skipped: %s", path)
                    return []

                documents = self._extract_page_documents(pdf_document, path)
                self.logger.info(
                    "PDF loaded: %s; pages extracted: %s",
                    path,
                    len(documents),
                )
                return documents
        except (fitz.FileDataError, fitz.FileNotFoundError, RuntimeError, ValueError) as error:
            self.logger.error("Failed to load PDF %s: %s", path, error)
            return []

    def load_multiple_pdfs(self, directory: str | Path) -> list[Document]:
        """Load every PDF file from a directory.

        Non-PDF files are ignored.

        Args:
            directory: Directory containing PDF files.

        Returns:
            A combined list of LangChain ``Document`` objects extracted from all
            readable PDFs in the directory. Returns an empty list if the
            directory is missing or no readable PDFs are found.
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            self.logger.error("PDF directory does not exist: %s", directory_path)
            return []

        if not directory_path.is_dir():
            self.logger.error("PDF directory path is not a directory: %s", directory_path)
            return []

        documents: list[Document] = []
        for pdf_path in sorted(directory_path.iterdir()):
            if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
                documents.extend(self.load_pdf(pdf_path))

        self.logger.info(
            "Loaded %s LangChain documents from directory: %s",
            len(documents),
            directory_path,
        )
        return documents

    def extract_metadata(self, file_path: str | Path) -> dict[str, Any]:
        """Extract file and PyMuPDF metadata from a PDF.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A metadata dictionary with filename, page count, file size, title,
            author, and creation date. Missing metadata fields are returned as
            empty strings, and unreadable PDFs return safe default values.
        """
        path = Path(file_path)
        metadata: dict[str, Any] = {
            "filename": path.name,
            "pages": 0,
            "file_size": self._format_file_size(path.stat().st_size) if path.exists() else "",
            "title": "",
            "author": "",
            "creation_date": "",
        }

        if not self._is_readable_pdf_path(path):
            return metadata

        try:
            with fitz.open(path) as pdf_document:
                if self._is_encrypted(pdf_document, path):
                    return metadata

                pdf_metadata = pdf_document.metadata or {}
                metadata.update(
                    {
                        "pages": pdf_document.page_count,
                        "title": pdf_metadata.get("title") or "",
                        "author": pdf_metadata.get("author") or "",
                        "creation_date": pdf_metadata.get("creationDate") or "",
                    }
                )
                self.logger.info("Metadata extracted: %s", path)
                return metadata
        except (fitz.FileDataError, fitz.FileNotFoundError, RuntimeError, ValueError) as error:
            self.logger.error("Failed to extract metadata from %s: %s", path, error)
            return metadata

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split long LangChain documents into smaller chunks.

        Args:
            documents: LangChain documents to split.

        Returns:
            Chunked LangChain documents using a 1000-character chunk size and
            200-character overlap by default.
        """
        if not documents:
            self.logger.warning("No documents provided for splitting.")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_index": chunk_index,
            }

        self.logger.info(
            "Split %s documents into %s chunks.",
            len(documents),
            len(chunks),
        )
        return chunks

    def get_statistics(self, documents: list[Document]) -> dict[str, int]:
        """Calculate basic statistics for LangChain documents.

        Args:
            documents: LangChain documents or chunks.

        Returns:
            A dictionary containing document count, unique page count, chunk
            count, average word count, and total word count.
        """
        total_words = sum(self._count_words(document.page_content) for document in documents)
        average_words = round(total_words / len(documents)) if documents else 0
        pages = self._count_unique_pages(documents)

        statistics = {
            "documents": len(documents),
            "pages": pages,
            "chunks": len(documents),
            "average_words": average_words,
            "total_words": total_words,
        }
        self.logger.info("Document statistics calculated: %s", statistics)
        return statistics

    def _is_readable_pdf_path(self, path: Path) -> bool:
        """Validate that a path points to an existing PDF file.

        Args:
            path: Candidate PDF path.

        Returns:
            ``True`` when the path exists, is a file, and has a PDF extension;
            otherwise ``False``.
        """
        if not path.exists():
            self.logger.error("PDF file does not exist: %s", path)
            return False

        if not path.is_file():
            self.logger.error("PDF path is not a file: %s", path)
            return False

        if path.suffix.lower() != ".pdf":
            self.logger.warning("Non-PDF file skipped: %s", path)
            return False

        return True

    def _is_encrypted(self, pdf_document: fitz.Document, path: Path) -> bool:
        """Determine whether a PDF is encrypted and inaccessible.

        Args:
            pdf_document: Open PyMuPDF document.
            path: Source PDF path used for logging.

        Returns:
            ``True`` when the PDF requires a password; otherwise ``False``.
        """
        if pdf_document.needs_pass:
            self.logger.error("Encrypted PDF skipped: %s", path)
            return True
        return False

    def _extract_page_documents(
        self,
        pdf_document: fitz.Document,
        path: Path,
    ) -> list[Document]:
        """Extract readable pages from an open PDF document.

        Args:
            pdf_document: Open PyMuPDF document.
            path: Source PDF path.

        Returns:
            A list of LangChain documents, one per readable non-empty page.
        """
        documents: list[Document] = []
        metadata = self.extract_metadata(path)

        for page_index in range(pdf_document.page_count):
            page_number = page_index + 1
            try:
                page_text = pdf_document.load_page(page_index).get_text("text")
            except RuntimeError as error:
                self.logger.error(
                    "Unreadable page skipped: %s page %s: %s",
                    path,
                    page_number,
                    error,
                )
                continue

            if not page_text.strip():
                self.logger.warning("Empty page skipped: %s page %s", path, page_number)
                continue

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "filename": path.name,
                        "page_number": page_number,
                        "page": page_number,
                        "source": str(path),
                        "document_metadata": metadata,
                    },
                )
            )

        return documents

    def _format_file_size(self, size_bytes: int) -> str:
        """Format a byte count as a human-readable file size.

        Args:
            size_bytes: File size in bytes.

        Returns:
            Human-readable file size string.
        """
        units = ("B", "KB", "MB", "GB")
        size = float(size_bytes)

        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f"{size:.2f} {unit}"
            size /= 1024

        return f"{size_bytes} B"

    def _count_unique_pages(self, documents: list[Document]) -> int:
        """Count unique page references across documents.

        Args:
            documents: LangChain documents or chunks.

        Returns:
            Number of unique source/page combinations. If page metadata is not
            available, falls back to the number of documents.
        """
        page_keys = {
            (
                document.metadata.get("source"),
                document.metadata.get("page_number") or document.metadata.get("page"),
            )
            for document in documents
            if document.metadata.get("page_number") or document.metadata.get("page")
        }
        return len(page_keys) if page_keys else len(documents)

    def _count_words(self, text: str) -> int:
        """Count words in a text string.

        Args:
            text: Text content to count.

        Returns:
            Number of whitespace-separated words.
        """
        return len(text.split())

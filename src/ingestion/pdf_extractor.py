"""PDF extraction service using PyMuPDF."""


class PDFExtractor:
    """Extracts text, page content, and layout metadata from PDFs."""

    def extract_text(self, file_path: str) -> str:
        """Extract full text from a PDF."""
        # TODO: Use PyMuPDF to extract document text.
        pass

    def extract_pages(self, file_path: str) -> list[dict]:
        """Extract page-level content from a PDF."""
        # TODO: Return page text and page metadata.
        pass

    def extract_tables(self, file_path: str) -> list[dict]:
        """Extract table-like content from a PDF."""
        # TODO: Identify and extract table structures when available.
        pass


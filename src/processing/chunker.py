"""Document chunking service for LLM and embedding workflows."""


class DocumentChunker:
    """Splits long healthcare documents into semantically useful chunks."""

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[dict]:
        """Split text into chunks with metadata."""
        # TODO: Use LangChain text splitters and preserve source metadata.
        pass


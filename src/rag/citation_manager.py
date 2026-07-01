"""Citation management for RAG outputs."""


class CitationManager:
    """Attaches source document, page, section, and chunk citations."""

    def build_citations(self, chunks: list[dict]) -> list[dict]:
        """Build citations from retrieved chunks."""
        # TODO: Normalize citation metadata.
        pass

    def attach_citations(self, response: dict, citations: list[dict]) -> dict:
        """Attach citations to an LLM response."""
        # TODO: Add citations to response payload.
        pass


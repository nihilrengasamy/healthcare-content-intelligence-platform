"""Document loading service for healthcare content sources."""


class DocumentLoader:
    """Loads supported healthcare documents from local or uploaded sources."""

    def load_document(self, file_path: str) -> object:
        """Load a document from a file path."""
        # TODO: Validate and load supported document files.
        pass

    def load_batch(self, directory_path: str) -> list[object]:
        """Load all supported documents from a directory."""
        # TODO: Load multiple documents for batch ingestion.
        pass


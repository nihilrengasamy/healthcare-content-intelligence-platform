"""Embedding service using sentence-transformers."""


class EmbeddingService:
    """Generates embeddings for healthcare content and user queries."""

    def generate_embeddings(self, chunks: list[dict]) -> object:
        """Generate embeddings for document chunks."""
        # TODO: Use sentence-transformers to embed chunks.
        pass

    def embed_query(self, query: str) -> object:
        """Generate an embedding for a query."""
        # TODO: Embed a user query for semantic retrieval.
        pass


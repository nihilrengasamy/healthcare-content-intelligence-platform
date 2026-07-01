"""FAISS vector storage for healthcare content intelligence workflows.

This module creates, manages, saves, loads, and queries a FAISS vector index
for healthcare document chunks prepared by ``modules.embeddings``. It does not
call GPT and is designed to provide reusable semantic retrieval for later RAG,
rule extraction, and feature extraction modules.

Example:
    ```python
    from modules.pdf_loader import PDFLoader
    from modules.embeddings import HealthcareEmbeddingGenerator
    from modules.vector_store import HealthcareVectorStore

    loader = PDFLoader()

    docs = loader.load_multiple_pdfs("data/")
    docs = loader.split_documents(docs)

    embedder = HealthcareEmbeddingGenerator()

    prepared = embedder.prepare_for_vector_store(
        docs,
        embedder.generate_embeddings(docs)
    )

    vector_store = HealthcareVectorStore()
    vector_store.create_index(prepared)

    results = vector_store.similarity_search("When is lumbar MRI covered?")

    print(results)
    ```
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """Protocol for sentence-transformers compatible embedding models."""

    def encode(
        self,
        sentences: str | list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode one string or a list of strings.

        Args:
            sentences: Query text or text collection to embed.
            convert_to_numpy: Whether to return NumPy arrays.
            show_progress_bar: Whether to display model progress.
            batch_size: Batch size for embedding generation.

        Returns:
            Embedding vector or matrix.
        """
        ...


class HealthcareVectorStore:
    """Creates and queries a FAISS index for healthcare document chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_vectors: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the healthcare vector store.

        Args:
            embedding_model: Optional sentence-transformers compatible model
                used for query embeddings.
            model_name: Sentence-transformers model name for lazy model loading.
            normalize_vectors: Whether to L2-normalize vectors for cosine-like
                similarity search with FAISS inner product indexes.
            logger: Optional logger instance.

        Returns:
            None.
        """
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.normalize_vectors = normalize_vectors
        self.logger = logger or logging.getLogger(__name__)
        self.index: Any | None = None
        self.texts: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.embedding_dimension = 0
        self._last_search_time_seconds = 0.0

    def create_index(self, prepared_documents: dict[str, list[Any]]) -> Any | None:
        """Create a FAISS vector index from prepared document data.

        Args:
            prepared_documents: Dictionary from ``embeddings.py`` containing
                ``texts``, ``embeddings``, and ``metadata`` lists.

        Returns:
            FAISS index when creation succeeds; otherwise ``None``.

        Raises:
            This method logs invalid input or FAISS failures and returns
            ``None`` instead of raising.
        """
        try:
            texts, embeddings, metadata = self._validate_prepared_documents(
                prepared_documents
            )
            if embeddings.size == 0:
                self.logger.warning("Cannot create FAISS index from empty embeddings.")
                return None

            import faiss

            vectors = self._prepare_vectors(embeddings)
            self.embedding_dimension = int(vectors.shape[1])
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.index.add(vectors)
            self.texts = texts
            self.metadata = metadata
            self.logger.info("Index created with %s documents.", len(self.texts))
            return self.index
        except Exception as error:
            self.logger.error("Failed to create FAISS index: %s", error)
            self.delete_index()
            return None

    def add_documents(self, prepared_documents: dict[str, list[Any]]) -> bool:
        """Add new healthcare document chunks to an existing FAISS index.

        Args:
            prepared_documents: Dictionary containing ``texts``, ``embeddings``,
                and ``metadata`` lists.

        Returns:
            ``True`` when documents are added successfully; otherwise ``False``.
        """
        try:
            texts, embeddings, metadata = self._validate_prepared_documents(
                prepared_documents
            )
            if embeddings.size == 0:
                self.logger.warning("No embeddings provided for incremental indexing.")
                return False

            if self.index is None:
                return self.create_index(prepared_documents) is not None

            vectors = self._prepare_vectors(embeddings)
            if vectors.shape[1] != self.embedding_dimension:
                self.logger.error(
                    "Embedding dimension mismatch: expected %s, received %s.",
                    self.embedding_dimension,
                    vectors.shape[1],
                )
                return False

            self.index.add(vectors)
            self.texts.extend(texts)
            self.metadata.extend(metadata)
            self.logger.info("Documents indexed incrementally: %s", len(texts))
            return True
        except Exception as error:
            self.logger.error("Failed to add documents to FAISS index: %s", error)
            return False

    def similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search the FAISS index for relevant healthcare document chunks.

        Args:
            query: Natural language search query.
            k: Number of top results to return.

        Returns:
            List of result dictionaries containing text, similarity score, and
            metadata. Returns an empty list if the index or query is invalid.
        """
        return self.similarity_search_with_scores(query, k)

    def similarity_search_with_scores(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search the FAISS index and include similarity scores.

        Args:
            query: Natural language search query.
            k: Number of top results to return.

        Returns:
            List of dictionaries with ``text``, ``score``, and ``metadata``.
        """
        start_time = time.perf_counter()
        if self.index is None or not self.texts:
            self.logger.warning("Similarity search requested with missing or empty index.")
            return []

        if not isinstance(query, str) or not query.strip():
            self.logger.warning("Invalid query provided for similarity search.")
            return []

        if k < 1:
            self.logger.warning("Search k must be greater than or equal to 1.")
            return []

        query_embedding = self._embed_query(query)
        if query_embedding.size == 0:
            return []

        try:
            query_vector = self._prepare_query_vector(query_embedding)
            top_k = min(k, len(self.texts))
            scores, indices = self.index.search(query_vector, top_k)
            results = self._format_search_results(scores[0], indices[0])
            self._last_search_time_seconds = time.perf_counter() - start_time
            self.logger.info(
                "Search performed in %.4f seconds. Results: %s",
                self._last_search_time_seconds,
                len(results),
            )
            return results
        except Exception as error:
            self.logger.error("Similarity search failed: %s", error)
            return []

    def save_index(self, index_path: str | Path) -> bool:
        """Persist the FAISS index and document payload locally.

        Args:
            index_path: Destination file path for the serialized vector store.

        Returns:
            ``True`` when the index is saved successfully; otherwise ``False``.
        """
        if self.index is None:
            self.logger.warning("Cannot save missing FAISS index.")
            return False

        path = Path(index_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "index": self._serialize_index(self.index),
                "texts": self.texts,
                "metadata": self.metadata,
                "embedding_dimension": self.embedding_dimension,
                "model": self.model_name,
                "normalize_vectors": self.normalize_vectors,
            }
            with path.open("wb") as output_file:
                pickle.dump(payload, output_file)
            self.logger.info("Index saved: %s", path)
            return True
        except Exception as error:
            self.logger.error("Failed to save FAISS index to %s: %s", path, error)
            return False

    def load_index(self, index_path: str | Path) -> bool:
        """Load a previously saved FAISS index and document payload.

        Args:
            index_path: Source file path for the serialized vector store.

        Returns:
            ``True`` when the index is loaded successfully; otherwise ``False``.
        """
        path = Path(index_path)
        if not path.exists():
            self.logger.error("Index file does not exist: %s", path)
            return False

        try:
            with path.open("rb") as input_file:
                payload = pickle.load(input_file)

            if not self._is_valid_index_payload(payload):
                self.logger.error("Corrupted index payload: %s", path)
                return False

            self.index = self._deserialize_index(payload["index"])
            self.texts = payload["texts"]
            self.metadata = payload["metadata"]
            self.embedding_dimension = int(payload["embedding_dimension"])
            self.model_name = str(payload.get("model", self.model_name))
            self.normalize_vectors = bool(
                payload.get("normalize_vectors", self.normalize_vectors)
            )
            self.logger.info("Index loaded: %s", path)
            return True
        except Exception as error:
            self.logger.error("Failed to load FAISS index from %s: %s", path, error)
            self.delete_index()
            return False

    def delete_index(self) -> None:
        """Delete the current in-memory vector index.

        Args:
            None.

        Returns:
            None.
        """
        self.index = None
        self.texts = []
        self.metadata = []
        self.embedding_dimension = 0
        self.logger.info("Current vector index deleted.")

    def get_index_statistics(self) -> dict[str, int | str]:
        """Return vector index statistics.

        Args:
            None.

        Returns:
            Dictionary containing document count, chunk count, embedding
            dimension, index type, and embedding model name.
        """
        return {
            "documents": len(self.texts),
            "chunks": len(self.texts),
            "embedding_dimension": self.embedding_dimension,
            "index_type": "FAISS",
            "model": self.model_name,
        }

    def _validate_prepared_documents(
        self,
        prepared_documents: dict[str, list[Any]],
    ) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
        """Validate prepared document payloads from the embeddings module.

        Args:
            prepared_documents: Prepared embedding payload.

        Returns:
            Tuple containing texts, embedding matrix, and metadata.

        Raises:
            ValueError: If the payload is invalid.
        """
        if not isinstance(prepared_documents, dict):
            raise ValueError("prepared_documents must be a dictionary.")

        texts = prepared_documents.get("texts", [])
        embeddings = prepared_documents.get("embeddings", [])
        metadata = prepared_documents.get("metadata", [])

        if not isinstance(texts, list) or not isinstance(metadata, list):
            raise ValueError("texts and metadata must be lists.")

        if len(texts) != len(embeddings) or len(texts) != len(metadata):
            raise ValueError("texts, embeddings, and metadata must have matching lengths.")

        if not texts:
            return [], np.empty((0, 0), dtype="float32"), []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        normalized_metadata = [
            self._normalize_metadata(item if isinstance(item, dict) else {})
            for item in metadata
        ]
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix.")

        if vectors.shape[0] != len(texts):
            raise ValueError("Embedding row count must match text count.")

        if vectors.shape[1] < 1:
            raise ValueError("Embeddings must have at least one dimension.")

        return texts, vectors, normalized_metadata

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Normalize metadata for retrieval output.

        Args:
            metadata: Source metadata.

        Returns:
            Metadata with source and page aliases preserved.
        """
        normalized = dict(metadata or {})
        normalized.setdefault("source", "")
        if "page" not in normalized and "page_number" in normalized:
            normalized["page"] = normalized["page_number"]
        if "page_number" not in normalized and "page" in normalized:
            normalized["page_number"] = normalized["page"]
        return normalized

    def _prepare_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        """Prepare embedding vectors for FAISS indexing.

        Args:
            embeddings: Raw embedding matrix.

        Returns:
            Float32 embedding matrix, optionally normalized.
        """
        vectors = np.asarray(embeddings, dtype="float32")
        if self.normalize_vectors:
            vectors = self._normalize_matrix(vectors)
        return np.ascontiguousarray(vectors)

    def _prepare_query_vector(self, query_embedding: np.ndarray) -> np.ndarray:
        """Prepare a query embedding for FAISS search.

        Args:
            query_embedding: Raw query embedding vector.

        Returns:
            Two-dimensional float32 query vector.

        Raises:
            ValueError: If the query dimension does not match the index.
        """
        query_vector = np.asarray(query_embedding, dtype="float32")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != self.embedding_dimension:
            raise ValueError(
                "Query embedding dimension does not match the FAISS index dimension."
            )
        if self.normalize_vectors:
            query_vector = self._normalize_matrix(query_vector)
        return np.ascontiguousarray(query_vector)

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding matrix.

        Args:
            matrix: Embedding matrix.

        Returns:
            Normalized embedding matrix.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a natural language query.

        Args:
            query: Search query.

        Returns:
            Query embedding vector, or an empty array when embedding fails.
        """
        model = self._load_embedding_model()
        if model is None:
            return np.array([])

        try:
            vector = model.encode(
                query.strip(),
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1,
            )
            return np.asarray(vector, dtype="float32")
        except Exception as error:
            self.logger.error("Failed to embed query: %s", error)
            return np.array([])

    def _load_embedding_model(self) -> EmbeddingModel | None:
        """Load the sentence-transformers query embedding model.

        Args:
            None.

        Returns:
            Embedding model when available; otherwise ``None``.
        """
        if self.embedding_model is not None:
            return self.embedding_model

        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded for vector search: %s", self.model_name)
            return self.embedding_model
        except Exception as error:
            self.logger.error("Failed to load embedding model %s: %s", self.model_name, error)
            return None

    def _format_search_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Format raw FAISS results.

        Args:
            scores: FAISS similarity scores.
            indices: FAISS result indices.

        Returns:
            List of retrieval result dictionaries.
        """
        results: list[dict[str, Any]] = []
        for score, index in zip(scores, indices):
            item_index = int(index)
            if item_index < 0 or item_index >= len(self.texts):
                continue
            results.append(
                {
                    "text": self.texts[item_index],
                    "score": float(score),
                    "metadata": self.metadata[item_index],
                }
            )
        return results

    def _is_valid_index_payload(self, payload: Any) -> bool:
        """Validate a serialized vector store payload.

        Args:
            payload: Pickle payload.

        Returns:
            ``True`` when the payload has the required fields and shapes.
        """
        if not isinstance(payload, dict):
            return False

        required_fields = {"index", "texts", "metadata", "embedding_dimension"}
        if not required_fields.issubset(payload):
            return False

        if not isinstance(payload["texts"], list):
            return False

        if not isinstance(payload["metadata"], list):
            return False

        if len(payload["texts"]) != len(payload["metadata"]):
            return False

        return isinstance(payload["embedding_dimension"], int)

    def _serialize_index(self, index: Any) -> bytes:
        """Serialize a FAISS index to bytes.

        Args:
            index: FAISS index object.

        Returns:
            Serialized FAISS index bytes.
        """
        import faiss

        return bytes(faiss.serialize_index(index))

    def _deserialize_index(self, index_bytes: bytes) -> Any:
        """Deserialize a FAISS index from bytes.

        Args:
            index_bytes: Serialized FAISS index bytes.

        Returns:
            FAISS index object.
        """
        import faiss

        return faiss.deserialize_index(np.frombuffer(index_bytes, dtype="uint8"))

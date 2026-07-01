"""Embedding generation for healthcare content intelligence workflows.

This module accepts LangChain ``Document`` objects produced by
``modules.pdf_loader.PDFLoader`` and generates sentence-transformer embeddings
for downstream FAISS indexing, RAG, semantic search, rule extraction, and
feature extraction.

Example:
    ```python
    from modules.pdf_loader import PDFLoader
    from modules.embeddings import HealthcareEmbeddingGenerator

    loader = PDFLoader()

    docs = loader.load_multiple_pdfs("data/")
    docs = loader.split_documents(docs)

    embedder = HealthcareEmbeddingGenerator()
    embeddings = embedder.generate_embeddings(docs)

    prepared = embedder.prepare_for_vector_store(docs, embeddings)

    print(prepared)
    ```
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from langchain_core.documents import Document


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
            sentences: Text or texts to encode.
            convert_to_numpy: Whether the output should be a NumPy array.
            show_progress_bar: Whether to show embedding progress.
            batch_size: Batch size used by the embedding model.

        Returns:
            Embedding vector or matrix.
        """
        ...


class HealthcareEmbeddingGenerator:
    """Generates embeddings for healthcare LangChain documents."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model: EmbeddingModel | None = None,
        batch_size: int = 32,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the healthcare embedding generator.

        Args:
            model_name: Name of the HuggingFace sentence-transformers model.
            model: Optional preloaded sentence-transformer compatible model.
            batch_size: Number of texts to embed per batch.
            logger: Optional logger instance.

        Returns:
            None.

        Raises:
            ValueError: If ``batch_size`` is less than 1.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be greater than or equal to 1.")

        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        self.embeddings: list[dict[str, Any]] = []
        self._last_processing_time_seconds = 0.0
        self._embedding_dimension = 0
        self._document_count = 0

    def load_embedding_model(self) -> EmbeddingModel | None:
        """Load the configured sentence-transformers model.

        Args:
            None.

        Returns:
            A ``SentenceTransformer`` model instance when loading succeeds;
            otherwise ``None``.

        Raises:
            This method logs model loading failures and returns ``None`` rather
            than raising.
        """
        if self.model is not None:
            return self.model

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded: %s", self.model_name)
            return self.model
        except Exception as error:
            self.logger.error("Failed to load embedding model %s: %s", self.model_name, error)
            return None

    def generate_embeddings(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Generate embeddings for LangChain documents.

        Args:
            documents: List of LangChain ``Document`` objects from
                ``PDFLoader``.

        Returns:
            A list of dictionaries containing source text, embedding vectors,
            and preserved metadata. Returns an empty list when input is empty,
            invalid, or the embedding model cannot be loaded.

        Raises:
            This method logs invalid inputs and model failures instead of
            raising.
        """
        start_time = time.perf_counter()
        valid_documents = self._validate_documents(documents)
        self._document_count = len(valid_documents)

        if not valid_documents:
            self.logger.warning("No valid documents provided for embedding generation.")
            self.embeddings = []
            self._last_processing_time_seconds = 0.0
            self._embedding_dimension = 0
            return []

        model = self.load_embedding_model()
        if model is None:
            self.embeddings = []
            self._last_processing_time_seconds = time.perf_counter() - start_time
            return []

        texts = [document.page_content.strip() for document in valid_documents]
        try:
            vectors = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            vectors = np.asarray(vectors, dtype=float)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            self._embedding_dimension = int(vectors.shape[1]) if vectors.size else 0
            self.embeddings = [
                {
                    "text": document.page_content,
                    "embedding": vectors[index].tolist(),
                    "metadata": self._normalize_metadata(document.metadata),
                }
                for index, document in enumerate(valid_documents)
            ]
            self._last_processing_time_seconds = time.perf_counter() - start_time
            self.logger.info(
                "Embeddings generated for %s documents in %.2f seconds.",
                len(self.embeddings),
                self._last_processing_time_seconds,
            )
            return self.embeddings
        except Exception as error:
            self.logger.error("Failed to generate embeddings: %s", error)
            self.embeddings = []
            self._last_processing_time_seconds = time.perf_counter() - start_time
            self._embedding_dimension = 0
            return []

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding vector for one string.

        Args:
            text: Text to embed.

        Returns:
            NumPy embedding vector. Returns an empty array when text is invalid
            or the embedding model fails.

        Raises:
            This method logs invalid text and model failures instead of raising.
        """
        if not isinstance(text, str) or not text.strip():
            self.logger.warning("Invalid text provided for embedding generation.")
            return np.array([])

        model = self.load_embedding_model()
        if model is None:
            return np.array([])

        try:
            vector = model.encode(
                text.strip(),
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1,
            )
            vector = np.asarray(vector, dtype=float)
            self._embedding_dimension = int(vector.shape[-1]) if vector.size else 0
            return vector
        except Exception as error:
            self.logger.error("Failed to generate embedding for text: %s", error)
            return np.array([])

    def prepare_for_vector_store(
        self,
        documents: list[Document],
        embeddings: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        """Prepare texts, embeddings, and metadata for FAISS vector storage.

        Args:
            documents: Original LangChain documents.
            embeddings: Embedding records returned by ``generate_embeddings``.

        Returns:
            Dictionary containing aligned ``texts``, ``embeddings``, and
            ``metadata`` lists. Does not create or persist a FAISS index.
        """
        if not embeddings:
            self.logger.warning("No embeddings provided for vector store preparation.")
            return {"texts": [], "embeddings": [], "metadata": []}

        if documents and len(documents) != len(embeddings):
            self.logger.warning(
                "Document count and embedding count differ: %s documents, %s embeddings.",
                len(documents),
                len(embeddings),
            )

        prepared = {
            "texts": [record.get("text", "") for record in embeddings],
            "embeddings": [record.get("embedding", []) for record in embeddings],
            "metadata": [record.get("metadata", {}) for record in embeddings],
        }
        self.logger.info("Prepared %s embeddings for vector store.", len(embeddings))
        return prepared

    def save_embeddings(self, output_path: str | Path) -> bool:
        """Save generated embeddings to disk using pickle.

        Args:
            output_path: Destination pickle file path.

        Returns:
            ``True`` when embeddings are saved successfully; otherwise
            ``False``.

        Raises:
            This method logs write failures and returns ``False`` instead of
            raising.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as output_file:
                pickle.dump(self.embeddings, output_file)
            self.logger.info("Embeddings saved: %s", path)
            return True
        except Exception as error:
            self.logger.error("Failed to save embeddings to %s: %s", path, error)
            return False

    def load_embeddings(self, input_path: str | Path) -> list[dict[str, Any]]:
        """Load saved embeddings from a pickle file.

        Args:
            input_path: Source pickle file path.

        Returns:
            Loaded embedding records. Returns an empty list when loading fails
            or the file contains invalid data.

        Raises:
            This method logs loading failures and returns an empty list instead
            of raising.
        """
        path = Path(input_path)
        if not path.exists():
            self.logger.error("Embedding file does not exist: %s", path)
            return []

        try:
            with path.open("rb") as input_file:
                loaded = pickle.load(input_file)

            if not self._is_valid_embedding_payload(loaded):
                self.logger.error("Corrupted embeddings payload: %s", path)
                return []

            self.embeddings = loaded
            self._document_count = len(loaded)
            self._embedding_dimension = self._infer_embedding_dimension(loaded)
            self.logger.info("Embeddings loaded: %s", path)
            return self.embeddings
        except Exception as error:
            self.logger.error("Failed to load embeddings from %s: %s", path, error)
            return []

    def get_embedding_statistics(self) -> dict[str, int | str]:
        """Return embedding generation statistics.

        Args:
            None.

        Returns:
            Dictionary containing document count, chunk count, embedding
            dimension, model name, and processing time.
        """
        return {
            "documents": self._document_count,
            "chunks": len(self.embeddings),
            "embedding_dimension": self._embedding_dimension,
            "model": self.model_name,
            "processing_time": f"{self._last_processing_time_seconds:.2f}s",
        }

    def _validate_documents(self, documents: list[Document]) -> list[Document]:
        """Validate and filter LangChain documents for embedding.

        Args:
            documents: Candidate LangChain document list.

        Returns:
            Valid documents with non-empty text.
        """
        if not isinstance(documents, list):
            self.logger.error("Documents input must be a list.")
            return []

        valid_documents: list[Document] = []
        for document in documents:
            if not isinstance(document, Document):
                self.logger.warning("Invalid document skipped: %s", type(document).__name__)
                continue
            if not isinstance(document.page_content, str) or not document.page_content.strip():
                self.logger.warning("Empty document skipped during embedding generation.")
                continue
            valid_documents.append(document)

        return valid_documents

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Normalize metadata for vector store compatibility.

        Args:
            metadata: LangChain document metadata.

        Returns:
            Metadata dictionary with source and page aliases preserved.
        """
        normalized = dict(metadata or {})
        if "page" not in normalized and "page_number" in normalized:
            normalized["page"] = normalized["page_number"]
        if "page_number" not in normalized and "page" in normalized:
            normalized["page_number"] = normalized["page"]
        normalized.setdefault("source", "")
        return normalized

    def _is_valid_embedding_payload(self, payload: Any) -> bool:
        """Validate loaded embedding records.

        Args:
            payload: Loaded pickle payload.

        Returns:
            ``True`` when the payload is a list of embedding records; otherwise
            ``False``.
        """
        if not isinstance(payload, list):
            return False
        return all(
            isinstance(record, dict)
            and isinstance(record.get("text", ""), str)
            and isinstance(record.get("embedding", []), list)
            and isinstance(record.get("metadata", {}), dict)
            for record in payload
        )

    def _infer_embedding_dimension(self, embeddings: list[dict[str, Any]]) -> int:
        """Infer embedding dimension from embedding records.

        Args:
            embeddings: Embedding records.

        Returns:
            Embedding vector dimension, or zero if unavailable.
        """
        if not embeddings:
            return 0
        first_embedding = embeddings[0].get("embedding", [])
        return len(first_embedding) if isinstance(first_embedding, list) else 0


"""Unit tests for the healthcare FAISS vector store."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modules.vector_store import HealthcareVectorStore


class FakeEmbeddingModel:
    """Deterministic query embedding model for vector store tests."""

    def encode(
        self,
        sentences: str | list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode test queries into deterministic vectors.

        Args:
            sentences: Text or list of texts.
            convert_to_numpy: Whether to return NumPy arrays.
            show_progress_bar: Whether to display progress.
            batch_size: Batch size.

        Returns:
            Deterministic vector or matrix.
        """
        if isinstance(sentences, list):
            return np.array([self._encode_one(sentence) for sentence in sentences])
        return np.array(self._encode_one(sentences))

    def _encode_one(self, text: str) -> list[float]:
        """Encode one text into a simple semantic test vector.

        Args:
            text: Text to encode.

        Returns:
            Three-dimensional fake embedding vector.
        """
        lowered = text.lower()
        return [
            1.0 if "mri" in lowered or "lumbar" in lowered else 0.0,
            1.0 if "authorization" in lowered else 0.0,
            1.0 if "cpt" in lowered or "billing" in lowered else 0.0,
        ]


def _prepared_documents() -> dict[str, list[object]]:
    """Build prepared document payload for tests.

    Args:
        None.

    Returns:
        Prepared vector store payload.
    """
    return {
        "texts": [
            "MRI is covered after six weeks of conservative therapy.",
            "Prior authorization is required for advanced imaging.",
            "Billing rules apply to CPT 72148.",
        ],
        "embeddings": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "metadata": [
            {"source": "coverage.pdf", "page": 1},
            {"source": "auth.pdf", "page": 2},
            {"source": "billing.pdf", "page": 3},
        ],
    }


def test_create_index() -> None:
    """Verify a FAISS index can be created from prepared documents."""
    vector_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())

    index = vector_store.create_index(_prepared_documents())

    assert index is not None
    assert vector_store.get_index_statistics()["chunks"] == 3
    assert vector_store.get_index_statistics()["embedding_dimension"] == 3


def test_similarity_search_returns_relevant_documents() -> None:
    """Verify semantic search returns matching healthcare chunks."""
    vector_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())
    vector_store.create_index(_prepared_documents())

    results = vector_store.similarity_search("When is lumbar MRI covered?", k=2)

    assert results
    assert "MRI is covered" in results[0]["text"]
    assert results[0]["metadata"]["source"] == "coverage.pdf"


def test_similarity_search_with_scores() -> None:
    """Verify scored search returns score fields."""
    vector_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())
    vector_store.create_index(_prepared_documents())

    results = vector_store.similarity_search_with_scores(
        "Show billing rules for CPT 72148.",
        k=1,
    )

    assert len(results) == 1
    assert "score" in results[0]
    assert results[0]["score"] > 0
    assert results[0]["metadata"]["source"] == "billing.pdf"


def test_incremental_updates() -> None:
    """Verify new documents can be added without rebuilding the index."""
    vector_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())
    vector_store.create_index(_prepared_documents())
    new_documents = {
        "texts": ["Clinical guideline requires documented diagnosis."],
        "embeddings": [[0.5, 0.0, 0.0]],
        "metadata": [{"source": "guideline.pdf", "page": 4}],
    }

    added = vector_store.add_documents(new_documents)

    assert added is True
    assert vector_store.get_index_statistics()["chunks"] == 4


def test_save_and_load_index(tmp_path: Path) -> None:
    """Verify a FAISS index can be saved and loaded."""
    output_path = tmp_path / "healthcare_index.pkl"
    vector_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())
    vector_store.create_index(_prepared_documents())

    assert vector_store.save_index(output_path) is True

    loaded_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())
    assert loaded_store.load_index(output_path) is True

    results = loaded_store.similarity_search("prior authorization requirements", k=1)
    assert results[0]["metadata"]["source"] == "auth.pdf"


def test_load_missing_index_returns_false(tmp_path: Path) -> None:
    """Verify missing index files are handled safely."""
    vector_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())

    loaded = vector_store.load_index(tmp_path / "missing.pkl")

    assert loaded is False


def test_delete_index() -> None:
    """Verify deleting the current index clears stored state."""
    vector_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())
    vector_store.create_index(_prepared_documents())

    vector_store.delete_index()

    assert vector_store.index is None
    assert vector_store.get_index_statistics()["chunks"] == 0
    assert vector_store.similarity_search("MRI", k=1) == []


def test_dimension_mismatch_on_incremental_update_returns_false() -> None:
    """Verify dimension mismatches are handled without crashing."""
    vector_store = HealthcareVectorStore(embedding_model=FakeEmbeddingModel())
    vector_store.create_index(_prepared_documents())
    mismatched_documents = {
        "texts": ["Invalid embedding dimension."],
        "embeddings": [[1.0, 0.0]],
        "metadata": [{"source": "bad.pdf", "page": 1}],
    }

    added = vector_store.add_documents(mismatched_documents)

    assert added is False
    assert vector_store.get_index_statistics()["chunks"] == 3


def test_statistics() -> None:
    """Verify index statistics expose required fields."""
    vector_store = HealthcareVectorStore(
        embedding_model=FakeEmbeddingModel(),
        model_name="test-model",
    )
    vector_store.create_index(_prepared_documents())

    statistics = vector_store.get_index_statistics()

    assert statistics == {
        "documents": 3,
        "chunks": 3,
        "embedding_dimension": 3,
        "index_type": "FAISS",
        "model": "test-model",
    }


"""Unit tests for healthcare embedding generation."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from modules.embeddings import HealthcareEmbeddingGenerator


class FakeEmbeddingModel:
    """Deterministic fake sentence-transformer model for tests."""

    def __init__(self, dimension: int = 4) -> None:
        """Initialize the fake embedding model.

        Args:
            dimension: Number of values in each fake embedding vector.

        Returns:
            None.
        """
        self.dimension = dimension
        self.calls = 0

    def encode(
        self,
        sentences: str | list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode text into deterministic numeric vectors.

        Args:
            sentences: Text or texts to encode.
            convert_to_numpy: Whether to return a NumPy array.
            show_progress_bar: Whether to show progress.
            batch_size: Batch size.

        Returns:
            Fake embedding vector or matrix.
        """
        self.calls += 1
        if isinstance(sentences, str):
            return np.array([float(len(sentences))] * self.dimension)
        return np.array(
            [[float(len(sentence))] * self.dimension for sentence in sentences]
        )


def test_generate_embedding_single_document() -> None:
    """Verify one LangChain document produces one embedding record."""
    model = FakeEmbeddingModel()
    generator = HealthcareEmbeddingGenerator(model=model)
    document = Document(
        page_content="Billing policy content",
        metadata={"source": "policy.pdf", "page_number": 1},
    )

    embeddings = generator.generate_embeddings([document])

    assert len(embeddings) == 1
    assert embeddings[0]["text"] == "Billing policy content"
    assert len(embeddings[0]["embedding"]) == 4
    assert embeddings[0]["metadata"]["source"] == "policy.pdf"
    assert embeddings[0]["metadata"]["page"] == 1


def test_generate_embeddings_multiple_documents_uses_batch_call() -> None:
    """Verify multiple documents are embedded in a single batch call."""
    model = FakeEmbeddingModel()
    generator = HealthcareEmbeddingGenerator(model=model)
    documents = [
        Document(page_content="Coverage policy", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Clinical guideline", metadata={"source": "b.pdf", "page": 2}),
    ]

    embeddings = generator.generate_embeddings(documents)

    assert len(embeddings) == 2
    assert model.calls == 1


def test_empty_document_is_skipped() -> None:
    """Verify empty documents do not produce embeddings."""
    model = FakeEmbeddingModel()
    generator = HealthcareEmbeddingGenerator(model=model)
    documents = [Document(page_content="   ", metadata={"source": "empty.pdf"})]

    embeddings = generator.generate_embeddings(documents)

    assert embeddings == []
    assert model.calls == 0


def test_generate_embedding_invalid_text_returns_empty_array() -> None:
    """Verify invalid text input returns an empty NumPy array."""
    model = FakeEmbeddingModel()
    generator = HealthcareEmbeddingGenerator(model=model)

    vector = generator.generate_embedding("")

    assert isinstance(vector, np.ndarray)
    assert vector.size == 0
    assert model.calls == 0


def test_generate_embedding_dimension() -> None:
    """Verify single-text embedding returns expected vector dimension."""
    model = FakeEmbeddingModel(dimension=6)
    generator = HealthcareEmbeddingGenerator(model=model)

    vector = generator.generate_embedding("medical necessity criteria")

    assert vector.shape == (6,)
    assert generator.get_embedding_statistics()["embedding_dimension"] == 6


def test_prepare_for_vector_store() -> None:
    """Verify prepared output is aligned for downstream FAISS indexing."""
    model = FakeEmbeddingModel()
    generator = HealthcareEmbeddingGenerator(model=model)
    documents = [
        Document(page_content="Policy one", metadata={"source": "one.pdf", "page": 1}),
        Document(page_content="Policy two", metadata={"source": "two.pdf", "page": 2}),
    ]
    embeddings = generator.generate_embeddings(documents)

    prepared = generator.prepare_for_vector_store(documents, embeddings)

    assert prepared["texts"] == ["Policy one", "Policy two"]
    assert len(prepared["embeddings"]) == 2
    assert prepared["metadata"][0]["source"] == "one.pdf"


def test_save_and_load_embeddings(tmp_path: Path) -> None:
    """Verify embeddings can be saved and loaded with pickle."""
    model = FakeEmbeddingModel()
    generator = HealthcareEmbeddingGenerator(model=model)
    documents = [Document(page_content="Contract content", metadata={"source": "contract.pdf"})]
    generator.generate_embeddings(documents)
    output_path = tmp_path / "embeddings.pkl"

    assert generator.save_embeddings(output_path) is True

    loaded_generator = HealthcareEmbeddingGenerator(model=model)
    loaded = loaded_generator.load_embeddings(output_path)

    assert loaded == generator.embeddings
    assert loaded_generator.get_embedding_statistics()["embedding_dimension"] == 4


def test_load_corrupted_embeddings_returns_empty_list(tmp_path: Path) -> None:
    """Verify corrupted embedding payloads are handled safely."""
    output_path = tmp_path / "corrupted.pkl"
    with output_path.open("wb") as output_file:
        pickle.dump({"not": "a valid embedding payload"}, output_file)

    generator = HealthcareEmbeddingGenerator(model=FakeEmbeddingModel())

    loaded = generator.load_embeddings(output_path)

    assert loaded == []


def test_get_embedding_statistics() -> None:
    """Verify embedding statistics are calculated."""
    model = FakeEmbeddingModel(dimension=5)
    generator = HealthcareEmbeddingGenerator(model=model, model_name="test-model")
    documents = [
        Document(page_content="A", metadata={}),
        Document(page_content="B", metadata={}),
    ]

    generator.generate_embeddings(documents)
    statistics = generator.get_embedding_statistics()

    assert statistics["documents"] == 2
    assert statistics["chunks"] == 2
    assert statistics["embedding_dimension"] == 5
    assert statistics["model"] == "test-model"
    assert statistics["processing_time"].endswith("s")


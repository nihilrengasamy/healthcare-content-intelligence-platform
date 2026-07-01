"""Unit tests for the healthcare RAG assistant."""

from __future__ import annotations

from modules.rag import HealthcareRAGAssistant


class FakeResponse:
    """Simple fake LLM response."""

    def __init__(self, content: str) -> None:
        """Initialize a fake response.

        Args:
            content: Response content.

        Returns:
            None.
        """
        self.content = content


class FakeLLM:
    """Fake LangChain-compatible LLM."""

    def __init__(self, content: str) -> None:
        """Initialize the fake LLM.

        Args:
            content: Response content returned for every invocation.

        Returns:
            None.
        """
        self.content = content
        self.calls = 0

    def invoke(self, messages: list[object]) -> FakeResponse:
        """Return a fake LLM response.

        Args:
            messages: Prompt messages.

        Returns:
            Fake response.
        """
        self.calls += 1
        return FakeResponse(self.content)


class FakeVectorStore:
    """Fake healthcare vector store."""

    def __init__(self, results: list[dict[str, object]]) -> None:
        """Initialize the fake vector store.

        Args:
            results: Retrieval results to return.

        Returns:
            None.
        """
        self.results = results
        self.calls = 0

    def similarity_search_with_scores(self, query: str, k: int = 5) -> list[dict[str, object]]:
        """Return fake similarity search results.

        Args:
            query: Search query.
            k: Number of results requested.

        Returns:
            Fake retrieval results.
        """
        self.calls += 1
        return self.results[:k]


def _retrieval_results() -> list[dict[str, object]]:
    """Build fake retrieval results.

    Args:
        None.

    Returns:
        List of fake vector store results.
    """
    return [
        {
            "text": "Lumbar MRI is covered after six weeks of conservative therapy.",
            "score": 0.94,
            "metadata": {"source": "Billing_Policy.pdf", "page": 12},
        },
        {
            "text": "Prior authorization is required for advanced imaging.",
            "score": 0.88,
            "metadata": {"source": "Coverage_Policy.pdf", "page_number": 8},
        },
    ]


def test_single_question_answer_with_sources() -> None:
    """Verify one question returns an answer with citations."""
    vector_store = FakeVectorStore(_retrieval_results())
    llm = FakeLLM("Lumbar MRI is covered after six weeks of conservative therapy.")
    assistant = HealthcareRAGAssistant(vector_store=vector_store, llm=llm)

    response = assistant.answer_with_sources("When is lumbar MRI covered?")

    assert response["question"] == "When is lumbar MRI covered?"
    assert "six weeks" in response["answer"]
    assert response["citations"][0] == {"source": "Billing_Policy.pdf", "page": 12}
    assert response["confidence"] > 0
    assert llm.calls == 1


def test_multiple_questions() -> None:
    """Verify batch answering supports multiple questions."""
    vector_store = FakeVectorStore(_retrieval_results())
    llm = FakeLLM("This answer is grounded in retrieved policy content.")
    assistant = HealthcareRAGAssistant(vector_store=vector_store, llm=llm)

    responses = assistant.answer_batch(
        [
            "When is lumbar MRI covered?",
            "What are the prior authorization requirements?",
        ]
    )

    assert len(responses) == 2
    assert assistant.get_usage_statistics()["queries_answered"] == 2
    assert llm.calls == 2


def test_no_results_returns_unavailable_response() -> None:
    """Verify no retrieval results returns a safe unavailable answer."""
    vector_store = FakeVectorStore([])
    assistant = HealthcareRAGAssistant(vector_store=vector_store, llm=FakeLLM("unused"))

    response = assistant.answer_with_sources("What ICD codes are mentioned?")

    assert "unavailable" in response["answer"].lower()
    assert response["citations"] == []
    assert response["confidence"] == 0.0


def test_missing_vector_store() -> None:
    """Verify missing vector store does not crash."""
    assistant = HealthcareRAGAssistant(llm=FakeLLM("unused"))

    response = assistant.answer_with_sources("Show the billing policy for CPT 72148.")

    assert "unavailable" in response["answer"].lower()
    assert response["retrieved_documents"] == []


def test_citation_generation_deduplicates_sources() -> None:
    """Verify citations are generated and deduplicated."""
    duplicate_results = _retrieval_results() + [_retrieval_results()[0]]
    assistant = HealthcareRAGAssistant(
        vector_store=FakeVectorStore(duplicate_results),
        llm=FakeLLM("Answer with duplicate retrieved source."),
    )

    response = assistant.answer_with_sources("When is lumbar MRI covered?")

    assert len(response["citations"]) == 2
    assert {"source": "Coverage_Policy.pdf", "page": 8} in response["citations"]


def test_build_context_preserves_metadata() -> None:
    """Verify context includes source and page metadata."""
    assistant = HealthcareRAGAssistant()

    context = assistant.build_context(_retrieval_results())

    assert "Source: Billing_Policy.pdf" in context
    assert "Page: 12" in context
    assert "Lumbar MRI is covered" in context


def test_evaluate_response_grounded() -> None:
    """Verify grounded responses receive low hallucination risk."""
    assistant = HealthcareRAGAssistant()
    response = {
        "answer": "Lumbar MRI is covered after six weeks.",
        "citations": [{"source": "Billing_Policy.pdf", "page": 12}],
        "retrieved_documents": _retrieval_results(),
    }

    evaluation = assistant.evaluate_response(response)

    assert evaluation["grounded"] is True
    assert evaluation["confidence"] > 0
    assert evaluation["hallucination_risk"] == "Low"


def test_evaluate_response_ungrounded() -> None:
    """Verify ungrounded responses receive high hallucination risk."""
    assistant = HealthcareRAGAssistant()

    evaluation = assistant.evaluate_response({"answer": "Unsupported answer."})

    assert evaluation["grounded"] is False
    assert evaluation["hallucination_risk"] == "High"


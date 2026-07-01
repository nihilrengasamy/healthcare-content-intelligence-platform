"""Unit tests for the HealthcareSummarizer module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from modules.summarizer import HealthcareSummarizer


class FakeResponse:
    """Simple response object that mimics a LangChain chat response."""

    def __init__(self, content: str) -> None:
        """Initialize a fake response.

        Args:
            content: Response content.

        Returns:
            None.
        """
        self.content = content


class FakeLLM:
    """Test double for a successful LLM."""

    def __init__(self, content: str) -> None:
        """Initialize a fake LLM.

        Args:
            content: JSON response content to return.

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
            Fake response with configured content.
        """
        self.calls += 1
        return FakeResponse(self.content)


class FailingLLM:
    """Test double for an LLM that always fails."""

    def __init__(self) -> None:
        """Initialize a failing fake LLM.

        Args:
            None.

        Returns:
            None.
        """
        self.calls = 0

    def invoke(self, messages: list[object]) -> FakeResponse:
        """Raise a runtime error for every request.

        Args:
            messages: Prompt messages.

        Returns:
            This method does not return.

        Raises:
            RuntimeError: Always raised to simulate API failure.
        """
        self.calls += 1
        raise RuntimeError("API unavailable")


def _summary_json() -> str:
    """Build a valid summary JSON string.

    Args:
        None.

    Returns:
        Valid JSON summary string.
    """
    return json.dumps(
        {
            "document_type": "Coverage Policy",
            "purpose": "Define coverage requirements.",
            "covered_services": ["Service A"],
            "excluded_services": ["Service B"],
            "eligibility_criteria": ["Member must meet criteria."],
            "medical_necessity": ["Documented diagnosis required."],
            "prior_authorization": "Required before service.",
            "coding_requirements": ["Use CPT 12345."],
            "key_dates": ["2026-01-01"],
            "important_changes": ["Added prior authorization."],
            "executive_summary": "This policy defines coverage for Service A.",
        }
    )


def test_summarize_document_returns_structured_summary() -> None:
    """Verify a single document summary returns required fields."""
    summarizer = HealthcareSummarizer(llm=FakeLLM(_summary_json()))
    document = Document(
        page_content="Coverage policy text",
        metadata={"filename": "Coverage_Policy.pdf"},
    )

    summary = summarizer.summarize_document(document)

    assert summary["document_type"] == "Coverage Policy"
    assert summary["covered_services"] == ["Service A"]
    assert summary["executive_summary"]


def test_summarize_documents_returns_document_names() -> None:
    """Verify multiple documents are summarized into result records."""
    fake_llm = FakeLLM(_summary_json())
    summarizer = HealthcareSummarizer(llm=fake_llm, max_workers=2)
    documents = [
        Document(page_content="Policy one", metadata={"filename": "one.pdf"}),
        Document(page_content="Policy two", metadata={"filename": "two.pdf"}),
    ]

    results = summarizer.summarize_documents(documents)

    assert len(results) == 2
    assert {result["document"] for result in results} == {"one.pdf", "two.pdf"}
    assert fake_llm.calls == 2


def test_empty_document_returns_empty_summary() -> None:
    """Verify empty document content does not call the LLM."""
    fake_llm = FakeLLM(_summary_json())
    summarizer = HealthcareSummarizer(llm=fake_llm)
    document = Document(page_content="   ", metadata={"filename": "empty.pdf"})

    summary = summarizer.summarize_document(document)

    assert summary["document_type"] == ""
    assert summary["covered_services"] == []
    assert fake_llm.calls == 0


def test_api_failure_returns_empty_summary_after_retries() -> None:
    """Verify API failures are retried and return a safe empty summary."""
    fake_llm = FailingLLM()
    summarizer = HealthcareSummarizer(llm=fake_llm, max_retries=1)

    summary = summarizer.generate_structured_summary("Policy text")

    assert summary["executive_summary"] == ""
    assert fake_llm.calls == 2


def test_invalid_json_response_returns_empty_summary() -> None:
    """Verify invalid JSON responses are handled safely."""
    fake_llm = FakeLLM("not json")
    summarizer = HealthcareSummarizer(llm=fake_llm, max_retries=0)

    summary = summarizer.generate_structured_summary("Policy text")

    assert summary["document_type"] == ""
    assert summary["coding_requirements"] == []
    assert fake_llm.calls == 1


def test_json_validation_adds_missing_required_fields() -> None:
    """Verify partial JSON is normalized to the required schema."""
    fake_llm = FakeLLM('{"document_type": "Billing Policy"}')
    summarizer = HealthcareSummarizer(llm=fake_llm)

    summary = summarizer.generate_structured_summary("Billing policy text")

    assert summary["document_type"] == "Billing Policy"
    assert "covered_services" in summary
    assert "executive_summary" in summary
    assert summary["covered_services"] == []


def test_export_summary_json(tmp_path: Path) -> None:
    """Verify summaries export as JSON."""
    summarizer = HealthcareSummarizer(llm=FakeLLM(_summary_json()))
    output_path = tmp_path / "summary.json"
    summary = {"document": "policy.pdf", "summary": json.loads(_summary_json())}

    summarizer.export_summary(summary, output_path)

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["document"] == "policy.pdf"
    assert exported["summary"]["document_type"] == "Coverage Policy"


def test_export_summary_markdown(tmp_path: Path) -> None:
    """Verify summaries optionally export as Markdown."""
    summarizer = HealthcareSummarizer(llm=FakeLLM(_summary_json()))
    output_path = tmp_path / "summary.md"
    summary = {"document": "policy.pdf", "summary": json.loads(_summary_json())}

    summarizer.export_summary(summary, output_path)

    exported = output_path.read_text(encoding="utf-8")
    assert "# Healthcare Document Summaries" in exported
    assert "policy.pdf" in exported


def test_get_summary_statistics() -> None:
    """Verify summary statistics are calculated."""
    summarizer = HealthcareSummarizer(llm=FakeLLM(_summary_json()))
    summaries = [
        {"document": "one.pdf", "summary": json.loads(_summary_json())},
        {"document": "two.pdf", "summary": json.loads(_summary_json())},
    ]

    statistics = summarizer.get_summary_statistics(summaries)

    assert statistics["documents_processed"] == 2
    assert statistics["average_summary_length"] > 0
    assert statistics["processing_time"].endswith("s")


def test_summarize_document_rejects_invalid_input() -> None:
    """Verify invalid document input raises a type error."""
    summarizer = HealthcareSummarizer(llm=FakeLLM(_summary_json()))

    with pytest.raises(TypeError):
        summarizer.summarize_document("not a document")  # type: ignore[arg-type]


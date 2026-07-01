"""Unit tests for healthcare document classification."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from modules.document_classifier import HealthcareDocumentClassifier


class FakeResponse:
    """Fake LLM response object."""

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
        """Initialize fake LLM.

        Args:
            content: Response content.

        Returns:
            None.
        """
        self.content = content
        self.calls = 0

    def invoke(self, messages: list[object]) -> FakeResponse:
        """Return fake response.

        Args:
            messages: Prompt messages.

        Returns:
            Fake response.
        """
        self.calls += 1
        return FakeResponse(self.content)


def test_classify_billing_coding_policy_text() -> None:
    """Verify billing/coding policy classification."""
    classifier = HealthcareDocumentClassifier()

    result = classifier.classify_text(
        "Billing policy for CPT 72148 and ICD M54.16 claim coding reimbursement modifier denial."
    )

    assert result["document_type"] == "billing_coding_policy"


def test_classify_clinical_guideline_text() -> None:
    """Verify clinical guideline classification."""
    classifier = HealthcareDocumentClassifier()

    result = classifier.classify_text(
        "Clinical practice guideline with evidence-based treatment recommendations and standard of care."
    )

    assert result["document_type"] == "clinical_practice_guideline"


def test_classify_payer_provider_contract_text() -> None:
    """Verify payer-provider contract classification."""
    classifier = HealthcareDocumentClassifier()

    result = classifier.classify_text(
        "Provider agreement contract with payer fee schedule, allowed amount $750, copay and coinsurance payment terms."
    )

    assert result["document_type"] == "payer_provider_contract"


def test_classify_coverage_policy_text() -> None:
    """Verify coverage policy classification."""
    classifier = HealthcareDocumentClassifier()

    result = classifier.classify_text(
        "Coverage policy: MRI is covered when medical necessity criteria are met. Prior authorization required. Excluded services are listed."
    )

    assert result["document_type"] == "coverage_policy"


def test_classify_regulatory_document_text() -> None:
    """Verify regulatory document classification."""
    classifier = HealthcareDocumentClassifier()

    result = classifier.classify_text(
        "CMS compliance regulation under 42 CFR and HIPAA federal audit requirement."
    )

    assert result["document_type"] == "regulatory_document"


def test_classify_unknown_document() -> None:
    """Verify uncertain text is classified as unknown."""
    classifier = HealthcareDocumentClassifier()

    result = classifier.classify_text("This document contains a generic office memo.")

    assert result["document_type"] == "unknown"


def test_detect_icd_cpt_hcpcs_codes() -> None:
    """Verify code signals are detected."""
    classifier = HealthcareDocumentClassifier()

    signals = classifier.detect_document_signals("Diagnosis M54.16 CPT 72148 HCPCS J1100.")

    assert "ICD code present" in signals
    assert "CPT code present" in signals
    assert "HCPCS code present" in signals


def test_detect_contract_terms() -> None:
    """Verify contract term signals are detected."""
    classifier = HealthcareDocumentClassifier()

    signals = classifier.detect_document_signals("Contract fee schedule allowed amount $750 coinsurance 20%.")

    assert "Dollar amount present" in signals
    assert "Percentage present" in signals
    assert "Contract language present" in signals


def test_detect_prior_authorization_language() -> None:
    """Verify prior authorization signals are detected."""
    classifier = HealthcareDocumentClassifier()

    signals = classifier.detect_document_signals("Prior authorization is required.")

    assert "Prior authorization mentioned" in signals


def test_classify_langchain_document_object() -> None:
    """Verify LangChain Document classification preserves metadata."""
    classifier = HealthcareDocumentClassifier()
    document = Document(
        page_content="Coverage policy with covered benefits and prior authorization.",
        metadata={"filename": "policy.pdf", "page_number": 3},
    )

    result = classifier.classify_document(document)

    assert result["document_type"] == "coverage_policy"
    assert result["source_document"] == "policy.pdf"
    assert result["page_number"] == 3
    assert result["original_metadata"]["filename"] == "policy.pdf"


def test_classify_documents_aggregate() -> None:
    """Verify page classifications aggregate to document classification."""
    classifier = HealthcareDocumentClassifier()
    documents = [
        Document(page_content="Coverage policy covered benefit prior authorization."),
        Document(page_content="Excluded services medical necessity coverage criteria."),
    ]

    result = classifier.classify_documents(documents)

    assert result["document_type"] == "coverage_policy"
    assert len(result["page_classifications"]) == 2


def test_classify_from_structured_summary() -> None:
    """Verify classification from summarizer output."""
    classifier = HealthcareDocumentClassifier()
    summary = {
        "covered_services": ["Lumbar MRI"],
        "excluded_services": ["Experimental imaging"],
        "prior_authorization": "Prior authorization required.",
        "medical_necessity": ["Six weeks therapy"],
    }

    result = classifier.classify_from_summary(summary)

    assert result["document_type"] == "coverage_policy"


def test_save_and_load_classifications(tmp_path: Path) -> None:
    """Verify classifications save and load as JSON."""
    classifier = HealthcareDocumentClassifier()
    result = classifier.classify_text("CMS compliance regulation 42 CFR.")
    output_path = tmp_path / "classifications.json"

    assert classifier.save_classifications([result], output_path) is True
    loaded = classifier.load_classifications(output_path)

    assert loaded[0]["document_type"] == "regulatory_document"


def test_classification_statistics() -> None:
    """Verify classification statistics."""
    classifier = HealthcareDocumentClassifier()
    results = [
        classifier.classify_text("CPT 72148 billing coding claim."),
        classifier.classify_text("CMS HIPAA compliance regulation."),
    ]

    statistics = classifier.get_classification_statistics(results)

    assert statistics["total_documents"] == 2
    assert statistics["billing_coding_policy"] == 1
    assert statistics["regulatory_document"] == 1
    assert statistics["average_confidence"] > 0


def test_llm_fallback_behavior() -> None:
    """Verify invalid LLM output falls back to keyword classification."""
    classifier = HealthcareDocumentClassifier(llm=FakeLLM("not-json"))

    result = classifier.classify_text("CPT 72148 billing coding claim reimbursement.")

    assert result["document_type"] == "billing_coding_policy"
    assert result["method"] == "keyword"


def test_llm_classification_can_improve_result() -> None:
    """Verify mocked LLM classification can be used."""
    classifier = HealthcareDocumentClassifier(
        llm=FakeLLM(
            json.dumps(
                {
                    "document_type": "coverage_policy",
                    "confidence": 0.9,
                    "signals": ["Prior authorization mentioned"],
                    "reason": "LLM classified as coverage policy.",
                }
            )
        )
    )

    result = classifier.classify_text("Prior authorization and covered benefit language.")

    assert result["document_type"] == "coverage_policy"
    assert result["method"] == "keyword_and_llm"


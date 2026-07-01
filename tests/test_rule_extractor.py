"""Unit tests for healthcare rule extraction."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from modules.rule_extractor import HealthcareRuleExtractor


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
        """Initialize the fake LLM.

        Args:
            content: Response content.

        Returns:
            None.
        """
        self.content = content
        self.calls = 0

    def invoke(self, messages: list[object]) -> FakeResponse:
        """Return a fake response.

        Args:
            messages: Prompt messages.

        Returns:
            Fake LLM response.
        """
        self.calls += 1
        return FakeResponse(self.content)


def _valid_rule() -> dict[str, object]:
    """Build a valid rule dictionary.

    Args:
        None.

    Returns:
        Valid rule dictionary.
    """
    return {
        "rule_id": "RULE_001",
        "rule_type": "coverage",
        "document_type": "Coverage Policy",
        "service": "Lumbar spine MRI",
        "condition_logic": "AND",
        "conditions": [
            {
                "field": "therapy_weeks",
                "operator": ">=",
                "value": 6,
                "unit": "week",
                "logic": "AND",
                "description": "Six weeks of conservative therapy required.",
            }
        ],
        "decision": "approve",
        "action": "Approve when criteria are met.",
        "exceptions": ["Neurological deficits are present."],
        "required_documentation": ["Conservative therapy notes."],
        "source_text": "Lumbar spine MRI is covered after six weeks.",
        "source_document": "policy.pdf",
        "page_number": 3,
        "confidence": 0.92,
    }


def test_extract_rules_from_raw_text() -> None:
    """Verify rules can be extracted from raw text with a mocked LLM."""
    llm = FakeLLM(json.dumps([_valid_rule()]))
    extractor = HealthcareRuleExtractor(llm=llm)

    rules = extractor.extract_rules_from_text(
        "Lumbar spine MRI is covered after six weeks of conservative therapy."
    )

    assert len(rules) == 1
    assert rules[0]["rule_type"] == "coverage"
    assert rules[0]["service"] == "Lumbar spine MRI"
    assert llm.calls == 1


def test_extract_rules_from_documents_preserves_metadata() -> None:
    """Verify document metadata is preserved on extracted rules."""
    llm = FakeLLM(json.dumps([_valid_rule()]))
    extractor = HealthcareRuleExtractor(llm=llm)
    documents = [
        Document(
            page_content="Prior authorization is required.",
            metadata={
                "filename": "coverage.pdf",
                "page_number": 12,
                "document_type": "Coverage Policy",
            },
        )
    ]

    rules = extractor.extract_rules_from_documents(documents)

    assert len(rules) == 1
    assert rules[0]["source_document"] == "coverage.pdf"
    assert rules[0]["page_number"] == 12
    assert rules[0]["document_type"] == "Coverage Policy"


def test_extract_rules_from_summary() -> None:
    """Verify rules can be created from structured summaries."""
    extractor = HealthcareRuleExtractor(llm=FakeLLM("[]"))
    summary = {
        "document_type": "Billing Policy",
        "covered_services": ["Lumbar MRI"],
        "excluded_services": ["Experimental imaging"],
        "medical_necessity": ["Six weeks conservative therapy"],
        "prior_authorization": "Prior authorization is required.",
        "coding_requirements": ["Use CPT 72148"],
    }

    rules = extractor.extract_rules_from_summary(summary)

    assert len(rules) == 5
    assert {rule["rule_type"] for rule in rules} >= {
        "coverage",
        "exclusion",
        "medical_necessity",
        "prior_authorization",
        "coding",
    }


def test_validate_valid_rule() -> None:
    """Verify a valid rule passes schema validation."""
    extractor = HealthcareRuleExtractor(llm=FakeLLM("[]"))

    validation = extractor.validate_rule_schema(_valid_rule())

    assert validation == {"valid": True, "errors": []}


def test_reject_invalid_rule() -> None:
    """Verify invalid rule types are rejected by schema validation."""
    extractor = HealthcareRuleExtractor(llm=FakeLLM("[]"))
    invalid_rule = _valid_rule()
    invalid_rule["rule_type"] = "unsupported"

    validation = extractor.validate_rule_schema(invalid_rule)

    assert validation["valid"] is False
    assert validation["errors"]


def test_validate_rules_separates_valid_and_invalid() -> None:
    """Verify batch validation separates valid and invalid rules."""
    extractor = HealthcareRuleExtractor(llm=FakeLLM("[]"))
    invalid_rule = _valid_rule()
    invalid_rule["confidence"] = 1.5

    validation = extractor.validate_rules([_valid_rule(), invalid_rule])

    assert len(validation["valid_rules"]) == 1
    assert len(validation["invalid_rules"]) == 1


def test_save_and_load_rules(tmp_path: Path) -> None:
    """Verify rules can be saved and loaded as JSON."""
    extractor = HealthcareRuleExtractor(llm=FakeLLM("[]"))
    output_path = tmp_path / "rules.json"
    rules = [_valid_rule()]

    assert extractor.save_rules(rules, output_path) is True
    loaded_rules = extractor.load_rules(output_path)

    assert loaded_rules == rules


def test_handle_empty_input() -> None:
    """Verify empty text returns no rules."""
    llm = FakeLLM(json.dumps([_valid_rule()]))
    extractor = HealthcareRuleExtractor(llm=llm)

    rules = extractor.extract_rules_from_text("   ")

    assert rules == []
    assert llm.calls == 0


def test_handle_invalid_json() -> None:
    """Verify invalid LLM JSON returns no rules without crashing."""
    llm = FakeLLM("not-json")
    extractor = HealthcareRuleExtractor(llm=llm, max_retries=0)

    rules = extractor.extract_rules_from_text("Prior authorization is required.")

    assert rules == []
    assert llm.calls == 1


def test_rule_statistics() -> None:
    """Verify rule statistics are calculated by type and confidence."""
    extractor = HealthcareRuleExtractor(llm=FakeLLM("[]"))
    coverage_rule = _valid_rule()
    coding_rule = _valid_rule()
    coding_rule["rule_id"] = "RULE_002"
    coding_rule["rule_type"] = "coding"
    coding_rule["confidence"] = 0.8

    statistics = extractor.get_rule_statistics([coverage_rule, coding_rule])

    assert statistics["total_rules"] == 2
    assert statistics["coverage_rules"] == 1
    assert statistics["coding_rules"] == 1
    assert statistics["average_confidence"] == 0.86


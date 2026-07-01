"""Unit tests for healthcare feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from modules.feature_extractor import HealthcareFeatureExtractor


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


def _feature_payload() -> dict[str, object]:
    """Build a valid feature payload.

    Args:
        None.

    Returns:
        Feature payload dictionary.
    """
    return {
        "patient_age": None,
        "age_requirement": None,
        "gender_requirement": None,
        "icd_codes": ["M54.16"],
        "cpt_codes": ["72148"],
        "hcpcs_codes": ["J1100"],
        "diagnosis": "Lumbar radiculopathy",
        "procedure": "Lumbar spine MRI",
        "service": "Lumbar spine MRI",
        "therapy_weeks": 6,
        "prior_authorization_required": True,
        "medical_necessity_criteria": ["Six weeks conservative therapy"],
        "excluded_services": [],
        "covered_services": ["Lumbar spine MRI"],
        "frequency_limit": "",
        "contract_terms": {
            "allowed_amount": 750,
            "copay": None,
            "coinsurance": None,
            "currency": "USD",
        },
        "provider_specialty": "",
        "documentation_required": ["Therapy notes"],
        "coverage_type": "covered",
        "effective_date": "01/01/2026",
        "termination_date": "",
        "source_text": "Lumbar spine MRI CPT 72148 diagnosis M54.16.",
        "source_document": "policy.pdf",
        "page_number": 1,
        "document_type": "Coverage Policy",
        "confidence": 0.91,
    }


def test_extract_features_from_raw_text() -> None:
    """Verify features are extracted from raw text."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM(json.dumps(_feature_payload())))

    features = extractor.extract_features_from_text(
        "Lumbar spine MRI CPT 72148 is covered for diagnosis M54.16 "
        "after six weeks of conservative therapy. Prior authorization is required. "
        "Allowed amount is $750."
    )

    assert features["icd_codes"] == ["M54.16"]
    assert features["cpt_codes"] == ["72148"]
    assert features["therapy_weeks"] == 6
    assert features["prior_authorization_required"] is True
    assert features["contract_terms"]["allowed_amount"] == 750.0


def test_extract_features_from_documents_preserves_metadata() -> None:
    """Verify document metadata is preserved."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM(json.dumps(_feature_payload())))
    documents = [
        Document(
            page_content="CPT 72148 is covered for diagnosis M54.16.",
            metadata={
                "filename": "coverage.pdf",
                "page_number": 12,
                "document_type": "Coverage Policy",
            },
        )
    ]

    features = extractor.extract_features_from_documents(documents)

    assert len(features) == 1
    assert features[0]["source_document"] == "coverage.pdf"
    assert features[0]["page_number"] == 12
    assert features[0]["document_type"] == "Coverage Policy"


def test_extract_features_from_summary() -> None:
    """Verify summary fields are converted into feature fields."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("{}"))
    summary = {
        "document_type": "Coverage Policy",
        "covered_services": ["Lumbar spine MRI"],
        "excluded_services": ["Experimental imaging"],
        "medical_necessity": ["Six weeks conservative therapy"],
        "prior_authorization": "Prior authorization is required.",
        "coding_requirements": ["Use CPT 72148"],
        "key_dates": ["01/01/2026"],
    }

    features = extractor.extract_features_from_summary(summary)

    assert features["covered_services"] == ["Lumbar spine MRI"]
    assert features["excluded_services"] == ["Experimental imaging"]
    assert features["prior_authorization_required"] is True
    assert features["effective_date"] == "01/01/2026"


def test_extract_features_from_rules() -> None:
    """Verify rule conditions are transformed into model-ready features."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("{}"))
    rules = [
        {
            "rule_type": "coverage",
            "service": "Lumbar spine MRI",
            "conditions": [
                {"field": "therapy_weeks", "operator": ">=", "value": 6},
                {"field": "prior_authorization", "operator": "==", "value": True},
            ],
            "source_text": "CPT 72148 diagnosis M54.16.",
            "confidence": 0.9,
        }
    ]

    features = extractor.extract_features_from_rules(rules)

    assert features["therapy_weeks"] == 6
    assert features["prior_authorization_required"] is True
    assert features["covered_services"] == ["Lumbar spine MRI"]
    assert features["confidence"] == 0.9


def test_regex_extraction_of_icd_cpt_hcpcs_and_dollar_amount() -> None:
    """Verify regex extraction finds ICD, CPT, HCPCS, and dollar amounts."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("not-json"), max_retries=0)

    features = extractor.extract_features_from_text(
        "Diagnosis M54.16 and E11.9. CPT 72148, HCPCS J1100, allowed amount $750."
    )

    assert features["icd_codes"] == ["M54.16", "E11.9"]
    assert features["cpt_codes"] == ["72148"]
    assert features["hcpcs_codes"] == ["J1100"]
    assert features["contract_terms"]["allowed_amount"] == 750.0


def test_validate_valid_features() -> None:
    """Verify a valid feature record passes schema validation."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("{}"))

    validation = extractor.validate_feature_schema(_feature_payload())

    assert validation == {"valid": True, "errors": []}


def test_reject_invalid_features() -> None:
    """Verify invalid feature records are rejected."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("{}"))
    invalid = _feature_payload()
    invalid["confidence"] = 1.5

    validation = extractor.validate_feature_schema(invalid)

    assert validation["valid"] is False
    assert validation["errors"]


def test_to_dataframe() -> None:
    """Verify feature records convert to a Pandas DataFrame."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("{}"))

    dataframe = extractor.to_dataframe([_feature_payload()])

    assert len(dataframe) == 1
    assert "contract_allowed_amount" in dataframe.columns


def test_save_and_load_features_json(tmp_path: Path) -> None:
    """Verify JSON feature persistence."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("{}"))
    output_path = tmp_path / "features.json"

    assert extractor.save_features([_feature_payload()], output_path) is True
    loaded = extractor.load_features(output_path)

    assert loaded == [_feature_payload()]


def test_save_and_load_features_csv(tmp_path: Path) -> None:
    """Verify CSV feature persistence."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("{}"))
    output_path = tmp_path / "features.csv"

    assert extractor.save_features([_feature_payload()], output_path) is True
    loaded = extractor.load_features(output_path)

    assert len(loaded) == 1
    assert "contract_allowed_amount" in loaded[0]


def test_handle_empty_input() -> None:
    """Verify empty input returns an empty feature schema."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM(json.dumps(_feature_payload())))

    features = extractor.extract_features_from_text("   ")

    assert features["confidence"] == 0.0
    assert features["icd_codes"] == []


def test_handle_invalid_json() -> None:
    """Verify invalid LLM JSON falls back to regex extraction."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("not-json"), max_retries=0)

    features = extractor.extract_features_from_text("CPT 99213 is covered.")

    assert features["cpt_codes"] == ["99213"]
    assert features["coverage_type"] == "covered"


def test_feature_statistics() -> None:
    """Verify feature statistics are calculated."""
    extractor = HealthcareFeatureExtractor(llm=FakeLLM("{}"))
    features = _feature_payload()

    statistics = extractor.get_feature_statistics([features])

    assert statistics["total_feature_records"] == 1
    assert statistics["records_with_icd_codes"] == 1
    assert statistics["records_with_cpt_codes"] == 1
    assert statistics["records_with_prior_auth"] == 1
    assert statistics["records_with_contract_terms"] == 1
    assert statistics["average_confidence"] == 0.91


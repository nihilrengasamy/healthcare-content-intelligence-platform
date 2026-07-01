"""Unit tests for claim explainability."""

from __future__ import annotations

import json
from pathlib import Path

from modules.explainability import ClaimExplainabilityEngine


def _decision(decision: str = "approve") -> dict[str, object]:
    """Build a sample decision result.

    Args:
        decision: Final decision value.

    Returns:
        Decision dictionary.
    """
    return {
        "claim_decision": decision,
        "decision_score": 0.91 if decision == "approve" else 0.40,
        "confidence": 0.90,
        "recommendation": "Approve claim",
        "reasons": [
            "Required coverage rule matched.",
            "Prior authorization requirement satisfied.",
        ],
        "risk_flags": [] if decision == "approve" else ["Required documentation is incomplete."],
        "conflicts": [] if decision != "manual_review" else ["Rule engine approves but ML predicts denial."],
    }


def _rule_results() -> list[dict[str, object]]:
    """Build sample rule results.

    Args:
        None.

    Returns:
        Rule result list.
    """
    return [
        {
            "rule_id": "RULE_001",
            "matched": True,
            "decision": "approve",
            "reason": "Therapy requirement satisfied.",
            "conditions_evaluated": [
                {
                    "field": "therapy_weeks",
                    "expected": ">= 6",
                    "actual": 6,
                    "matched": True,
                }
            ],
            "confidence": 0.92,
        }
    ]


def _ml_prediction() -> dict[str, object]:
    """Build a sample ML prediction.

    Args:
        None.

    Returns:
        ML prediction dictionary.
    """
    return {
        "approval_probability": 0.93,
        "predicted_approval": True,
        "fraud_risk": 0.12,
        "medical_necessity_score": 0.88,
        "model_confidence": 0.91,
    }


def _features() -> dict[str, object]:
    """Build sample features.

    Args:
        None.

    Returns:
        Feature dictionary.
    """
    return {
        "patient_age": 55,
        "icd_codes": ["M54.16"],
        "cpt_codes": ["72148"],
        "diagnosis": "Lumbar radiculopathy",
        "procedure": "Lumbar spine MRI",
        "therapy_weeks": 6,
        "prior_authorization_required": True,
        "prior_authorization_obtained": True,
        "documentation_complete": True,
    }


def _citations() -> list[dict[str, object]]:
    """Build sample citations.

    Args:
        None.

    Returns:
        Citation list.
    """
    return [
        {
            "source": "Billing_Policy.pdf",
            "page": 12,
            "text": "MRI is covered after six weeks of conservative therapy.",
        }
    ]


def test_approve_explanation() -> None:
    """Verify approve explanation includes approval language."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_explanation(
        _decision("approve"),
        _rule_results(),
        _ml_prediction(),
        _features(),
    )

    assert explanation["final_decision"] == "approve"
    assert "recommended for approve" in explanation["plain_language_summary"]
    assert explanation["executive_explanation"]
    assert len(explanation["top_supporting_reasons"]) <= 3
    assert explanation["recommended_next_action"] == "Proceed with approval."


def test_deny_explanation() -> None:
    """Verify deny explanation includes deny next action."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_explanation(
        _decision("deny"),
        _rule_results(),
        _ml_prediction(),
        _features(),
    )

    assert explanation["final_decision"] == "deny"
    assert explanation["decision_drivers"]["why_decided"]
    assert explanation["confidence_rationale"]
    assert explanation["recommended_next_action"] == "Deny claim or request corrected documentation."


def test_manual_review_explanation() -> None:
    """Verify manual review explanation includes conflict explanation."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_explanation(
        _decision("manual_review"),
        _rule_results(),
        _ml_prediction(),
        _features(),
    )

    assert explanation["final_decision"] == "manual_review"
    assert explanation["conflict_explanation"]
    assert explanation["recommended_next_action"] == "Route claim to a human reviewer for manual validation."


def test_rule_explanation() -> None:
    """Verify rule explanation describes condition matching."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_rule_explanation(_rule_results())

    assert explanation[0]["rule_id"] == "RULE_001"
    assert explanation[0]["conditions"][0]["matched"] is True
    assert "satisfies" in explanation[0]["conditions"][0]["explanation"]


def test_ml_explanation() -> None:
    """Verify ML explanation includes core signals."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_ml_explanation(_ml_prediction())

    assert explanation["approval_probability"] == 0.93
    assert explanation["fraud_risk"] == 0.12
    assert explanation["ml_contribution"] == "ML supports approval."


def test_risk_explanation() -> None:
    """Verify risk flag explanations are generated."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_risk_explanation(["Missing CPT code"])

    assert explanation[0]["risk_flag"] == "Missing CPT code"
    assert "Procedure coding" in explanation[0]["explanation"]


def test_conflict_explanation() -> None:
    """Verify conflict explanations are generated."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_conflict_explanation(
        ["Rule engine recommends deny but ML predicts approval."]
    )

    assert explanation[0]["conflict"] == "Rule engine recommends deny but ML predicts approval."


def test_confidence_calculation() -> None:
    """Verify explanation confidence is clamped and positive."""
    explainer = ClaimExplainabilityEngine()

    confidence = explainer.calculate_explanation_confidence(
        _decision("approve"),
        _rule_results(),
        _ml_prediction(),
    )

    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.8


def test_audit_trail_generation() -> None:
    """Verify audit trail includes required fields."""
    explainer = ClaimExplainabilityEngine()

    audit_trail = explainer.generate_audit_trail(
        _decision("approve"),
        _rule_results(),
        _ml_prediction(),
        _features(),
        _citations(),
    )

    assert "decision_inputs" in audit_trail
    assert "rules_evaluated" in audit_trail
    assert audit_trail["explanation_version"] == "1.0"
    assert audit_trail["citations_used"][0]["source"] == "Billing_Policy.pdf"


def test_export_to_json(tmp_path: Path) -> None:
    """Verify JSON export."""
    explainer = ClaimExplainabilityEngine()
    explanation = explainer.generate_explanation(
        _decision("approve"),
        _rule_results(),
        _ml_prediction(),
        _features(),
    )
    output_path = tmp_path / "explanation.json"

    assert explainer.export_explanation(explanation, output_path) is True
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert exported["final_decision"] == "approve"


def test_export_to_markdown(tmp_path: Path) -> None:
    """Verify Markdown export."""
    explainer = ClaimExplainabilityEngine()
    explanation = explainer.generate_explanation(
        _decision("approve"),
        _rule_results(),
        _ml_prediction(),
        _features(),
    )
    output_path = tmp_path / "explanation.md"

    assert explainer.export_explanation(explanation, output_path) is True
    assert "# Claim Decision Explanation" in output_path.read_text(encoding="utf-8")


def test_missing_inputs() -> None:
    """Verify missing inputs return structured error explanation."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_explanation({}, [], {}, {})

    assert explanation["final_decision"] == "error"
    assert explanation["errors"]


def test_explanations_with_citations() -> None:
    """Verify citations appear in generated explanations."""
    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_explanation(
        _decision("approve"),
        _rule_results(),
        _ml_prediction(),
        _features(),
        _citations(),
    )

    assert explanation["supporting_citations"][0]["source"] == "Billing_Policy.pdf"
    assert explanation["supporting_citations"][0]["page"] == 12


def test_explanation_statistics() -> None:
    """Verify explanation statistics are calculated."""
    explainer = ClaimExplainabilityEngine()
    approve = explainer.generate_explanation(_decision("approve"), _rule_results(), _ml_prediction(), _features())
    deny = explainer.generate_explanation(_decision("deny"), _rule_results(), _ml_prediction(), _features())
    manual = explainer.generate_explanation(_decision("manual_review"), _rule_results(), _ml_prediction(), _features(), _citations())

    statistics = explainer.get_explanation_statistics([approve, deny, manual])

    assert statistics["total_explanations"] == 3
    assert statistics["approve_explanations"] == 1
    assert statistics["deny_explanations"] == 1
    assert statistics["manual_review_explanations"] == 1
    assert statistics["explanations_with_citations"] == 1

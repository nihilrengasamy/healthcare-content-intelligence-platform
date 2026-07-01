"""Unit tests for claim decision support logic."""

from __future__ import annotations

import json
from pathlib import Path

from modules.claim_decision import ClaimDecisionEngine


def _approve_rule() -> dict[str, object]:
    """Build an approve rule result.

    Args:
        None.

    Returns:
        Rule result dictionary.
    """
    return {
        "rule_id": "RULE_001",
        "matched": True,
        "decision": "approve",
        "reason": "All required conditions were satisfied.",
        "confidence": 0.92,
        "rule_type": "coverage",
    }


def _failed_rule() -> dict[str, object]:
    """Build a failed rule result.

    Args:
        None.

    Returns:
        Rule result dictionary.
    """
    return {
        "rule_id": "RULE_002",
        "matched": False,
        "decision": "approve",
        "reason": "Therapy requirement failed.",
        "confidence": 0.80,
        "rule_type": "medical_necessity",
    }


def _deny_rule() -> dict[str, object]:
    """Build a deny rule result.

    Args:
        None.

    Returns:
        Rule result dictionary.
    """
    return {
        "rule_id": "RULE_003",
        "matched": True,
        "decision": "deny",
        "reason": "Excluded service.",
        "confidence": 0.95,
        "rule_type": "exclusion",
    }


def _ml_prediction() -> dict[str, object]:
    """Build a favorable ML prediction.

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
    """Build complete claim features.

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


def test_approve_decision() -> None:
    """Verify favorable rule, ML, and feature signals approve."""
    engine = ClaimDecisionEngine()

    decision = engine.make_decision([_approve_rule()], _ml_prediction(), _features())

    assert decision["claim_decision"] == "approve"
    assert decision["decision_score"] >= 0.75
    assert decision["risk_flags"] == []
    assert decision["rule_summary"]["matched_rules"] == 1


def test_deny_decision() -> None:
    """Verify low score conditions can deny."""
    engine = ClaimDecisionEngine()
    ml_prediction = _ml_prediction()
    ml_prediction.update(
        {
            "approval_probability": 0.20,
            "predicted_approval": False,
            "fraud_risk": 0.30,
            "medical_necessity_score": 0.20,
            "model_confidence": 0.80,
        }
    )
    features = _features()
    features["icd_codes"] = []

    decision = engine.make_decision([_failed_rule()], ml_prediction, features)

    assert decision["claim_decision"] == "deny"
    assert "ICD codes are missing." in decision["risk_flags"]


def test_manual_review_decision() -> None:
    """Verify borderline score routes to manual review."""
    engine = ClaimDecisionEngine()
    ml_prediction = _ml_prediction()
    ml_prediction.update(
        {
            "approval_probability": 0.60,
            "predicted_approval": True,
            "fraud_risk": 0.35,
            "medical_necessity_score": 0.65,
            "model_confidence": 0.80,
        }
    )

    decision = engine.make_decision([_approve_rule(), _failed_rule()], ml_prediction, _features())

    assert decision["claim_decision"] == "manual_review"


def test_hard_deny_override() -> None:
    """Verify matched deny rules override ML approval."""
    engine = ClaimDecisionEngine()

    decision = engine.make_decision([_approve_rule(), _deny_rule()], _ml_prediction(), _features())

    assert decision["claim_decision"] == "deny"
    assert decision["rule_summary"]["deny_rules"] == 1


def test_high_fraud_risk() -> None:
    """Verify high fraud risk triggers denial."""
    engine = ClaimDecisionEngine()
    ml_prediction = _ml_prediction()
    ml_prediction["fraud_risk"] = 0.80

    decision = engine.make_decision([_approve_rule()], ml_prediction, _features())

    assert decision["claim_decision"] == "deny"
    assert "Fraud risk is elevated." in decision["risk_flags"]


def test_missing_documentation() -> None:
    """Verify incomplete documentation triggers manual review."""
    engine = ClaimDecisionEngine()
    features = _features()
    features["documentation_complete"] = False

    decision = engine.make_decision([_approve_rule()], _ml_prediction(), features)

    assert decision["claim_decision"] == "manual_review"
    assert "Required documentation is incomplete." in decision["risk_flags"]


def test_missing_prior_authorization() -> None:
    """Verify missing prior authorization triggers denial."""
    engine = ClaimDecisionEngine()
    features = _features()
    features["prior_authorization_obtained"] = False

    decision = engine.make_decision([_approve_rule()], _ml_prediction(), features)

    assert decision["claim_decision"] == "deny"
    assert "Prior authorization is required but not obtained." in decision["risk_flags"]


def test_deny_reasons_are_deduplicated_and_non_contradictory() -> None:
    """Verify deny reasons are coherent when matched rules coexist with hard risks."""
    engine = ClaimDecisionEngine()
    features = _features()
    ml_prediction = _ml_prediction()
    ml_prediction.update(
        {
            "approval_probability": 0.48,
            "predicted_approval": True,
            "fraud_risk": 0.95,
            "medical_necessity_score": 0.18,
            "model_confidence": 0.51,
        }
    )
    features["documentation_complete"] = True
    features["icd_codes"] = []
    features["cpt_codes"] = []

    decision = engine.make_decision([_approve_rule(), _failed_rule()], ml_prediction, features)

    assert decision["claim_decision"] == "deny"
    assert len(decision["reasons"]) == len(set(decision["reasons"]))
    assert "Required documentation is complete." not in decision["reasons"]
    assert any("higher-priority risk" in reason.lower() for reason in decision["reasons"])


def test_rule_ml_conflict() -> None:
    """Verify rule and ML conflicts route away from approval."""
    engine = ClaimDecisionEngine()
    ml_prediction = _ml_prediction()
    ml_prediction["predicted_approval"] = False
    ml_prediction["approval_probability"] = 0.30

    decision = engine.make_decision([_approve_rule()], ml_prediction, _features())

    assert decision["claim_decision"] == "manual_review"
    assert "Rule engine approves but ML predicts denial." in decision["conflicts"]


def test_decision_score_calculation() -> None:
    """Verify decision score is clamped and numeric."""
    engine = ClaimDecisionEngine()

    score = engine.calculate_decision_score([_approve_rule()], _ml_prediction(), _features())

    assert 0.0 <= score <= 1.0
    assert score > 0.75


def test_batch_decisions() -> None:
    """Verify batch decisions return one decision per claim."""
    engine = ClaimDecisionEngine()

    decisions = engine.make_batch_decisions(
        [[_approve_rule()], [_deny_rule()]],
        [_ml_prediction(), _ml_prediction()],
        [_features(), _features()],
    )

    assert isinstance(decisions, list)
    assert len(decisions) == 2
    assert decisions[0]["claim_decision"] == "approve"
    assert decisions[1]["claim_decision"] == "deny"


def test_batch_length_mismatch() -> None:
    """Verify mismatched batch inputs return structured error."""
    engine = ClaimDecisionEngine()

    result = engine.make_batch_decisions([[_approve_rule()]], [_ml_prediction()], [_features(), _features()])

    assert isinstance(result, dict)
    assert result["valid"] is False


def test_decision_statistics() -> None:
    """Verify decision statistics are calculated."""
    engine = ClaimDecisionEngine()
    approved = engine.make_decision([_approve_rule()], _ml_prediction(), _features())
    denied = engine.make_decision([_deny_rule()], _ml_prediction(), _features())
    manual = engine.make_decision([_approve_rule(), _failed_rule()], _ml_prediction(), _features())

    statistics = engine.get_decision_statistics([approved, denied, manual])

    assert statistics["total_claims"] == 3
    assert statistics["approved"] == 1
    assert statistics["denied"] == 1
    assert statistics["manual_review"] == 1
    assert statistics["average_confidence"] > 0


def test_save_decisions(tmp_path: Path) -> None:
    """Verify decisions save as JSON."""
    engine = ClaimDecisionEngine()
    decision = engine.make_decision([_approve_rule()], _ml_prediction(), _features())
    output_path = tmp_path / "decisions.json"

    assert engine.save_decisions([decision], output_path) is True
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved[0]["claim_decision"] == "approve"


def test_load_decisions(tmp_path: Path) -> None:
    """Verify decisions load from JSON."""
    engine = ClaimDecisionEngine()
    output_path = tmp_path / "decisions.json"
    output_path.write_text(json.dumps([{"claim_decision": "approve"}]), encoding="utf-8")

    decisions = engine.load_decisions(output_path)

    assert decisions == [{"claim_decision": "approve"}]


def test_invalid_inputs() -> None:
    """Verify invalid inputs return structured error decision."""
    engine = ClaimDecisionEngine()

    decision = engine.make_decision("bad", {}, {})  # type: ignore[arg-type]

    assert decision["claim_decision"] == "error"
    assert decision["errors"]

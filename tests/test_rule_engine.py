"""Unit tests for deterministic healthcare rule execution."""

from __future__ import annotations

import json
from pathlib import Path

from modules.rule_engine import HealthcareRuleEngine


def _matching_rule() -> dict[str, object]:
    """Build a matching coverage rule.

    Args:
        None.

    Returns:
        Rule dictionary.
    """
    return {
        "rule_id": "RULE_001",
        "rule_type": "coverage",
        "service": "Lumbar spine MRI",
        "condition_logic": "AND",
        "conditions": [
            {"field": "therapy_weeks", "operator": ">=", "value": 6, "unit": "weeks"},
            {
                "field": "prior_authorization_required",
                "operator": "==",
                "value": True,
            },
        ],
        "decision": "approve",
        "action": "Approve claim",
        "confidence": 0.92,
    }


def _features() -> dict[str, object]:
    """Build sample claim features.

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
    }


def test_evaluate_one_matching_rule() -> None:
    """Verify one matching rule returns approve."""
    engine = HealthcareRuleEngine()

    result = engine.evaluate_rule(_matching_rule(), _features())

    assert result["matched"] is True
    assert result["decision"] == "approve"
    assert result["reason"] == "All required conditions were satisfied."
    assert len(result["conditions_evaluated"]) == 2


def test_evaluate_one_non_matching_rule() -> None:
    """Verify one non-matching rule returns no_match."""
    engine = HealthcareRuleEngine()
    features = _features()
    features["therapy_weeks"] = 4

    result = engine.evaluate_rule(_matching_rule(), features)

    assert result["matched"] is False
    assert result["decision"] == "no_match"
    assert result["conditions_evaluated"][0]["matched"] is False


def test_evaluate_and_condition_logic() -> None:
    """Verify AND logic requires all conditions to match."""
    engine = HealthcareRuleEngine()

    result = engine.evaluate_rule(_matching_rule(), _features())

    assert result["matched"] is True


def test_evaluate_or_condition_logic() -> None:
    """Verify OR logic matches when at least one condition matches."""
    engine = HealthcareRuleEngine()
    rule = _matching_rule()
    rule["condition_logic"] = "OR"
    rule["conditions"] = [
        {"field": "therapy_weeks", "operator": ">=", "value": 10},
        {"field": "prior_authorization_required", "operator": "==", "value": True},
    ]

    result = engine.evaluate_rule(rule, _features())

    assert result["matched"] is True
    assert result["reason"] == "At least one required condition was satisfied."


def test_missing_feature_field() -> None:
    """Verify missing feature fields produce an unmatched condition reason."""
    engine = HealthcareRuleEngine()
    rule = _matching_rule()
    rule["conditions"] = [{"field": "missing_field", "operator": "exists", "value": True}]

    result = engine.evaluate_rule(rule, _features())

    assert result["matched"] is False
    assert "missing" in result["conditions_evaluated"][0]["reason"]


def test_unsupported_operator() -> None:
    """Verify unsupported operators do not crash."""
    engine = HealthcareRuleEngine()
    rule = _matching_rule()
    rule["conditions"] = [{"field": "therapy_weeks", "operator": "between", "value": [1, 6]}]

    result = engine.evaluate_rule(rule, _features())

    assert result["matched"] is False
    assert "Unsupported operator" in result["conditions_evaluated"][0]["reason"]


def test_invalid_rule_schema() -> None:
    """Verify invalid rules return structured errors."""
    engine = HealthcareRuleEngine()

    result = engine.evaluate_rule({"rule_id": "BAD"}, _features())

    assert result["decision"] == "error"
    assert result["errors"]


def test_contains_and_in_operators() -> None:
    """Verify collection operators support healthcare code matching."""
    engine = HealthcareRuleEngine()
    rule = {
        "rule_id": "RULE_002",
        "rule_type": "coding",
        "condition_logic": "AND",
        "conditions": [
            {"field": "cpt_codes", "operator": "contains", "value": "72148"},
            {"field": "icd_codes", "operator": "in", "value": ["M54.16", "M51.26"]},
        ],
        "decision": "review",
    }

    result = engine.evaluate_rule(rule, _features())

    assert result["matched"] is True


def test_batch_evaluation() -> None:
    """Verify batch evaluation evaluates each feature record."""
    engine = HealthcareRuleEngine()
    features_a = _features()
    features_b = _features()
    features_b["therapy_weeks"] = 3

    batch_results = engine.evaluate_batch([_matching_rule()], [features_a, features_b])

    assert len(batch_results) == 2
    assert batch_results[0]["results"][0]["matched"] is True
    assert batch_results[1]["results"][0]["matched"] is False


def test_statistics() -> None:
    """Verify rule engine statistics are calculated."""
    engine = HealthcareRuleEngine()
    results = [
        engine.evaluate_rule(_matching_rule(), _features()),
        engine.evaluate_rule(
            {
                **_matching_rule(),
                "rule_id": "RULE_002",
                "decision": "deny",
                "conditions": [{"field": "therapy_weeks", "operator": ">=", "value": 6}],
            },
            _features(),
        ),
        engine.evaluate_rule(
            {
                **_matching_rule(),
                "rule_id": "RULE_003",
                "decision": "review",
                "conditions": [{"field": "therapy_weeks", "operator": ">=", "value": 10}],
            },
            _features(),
        ),
    ]

    statistics = engine.get_rule_engine_statistics(results)

    assert statistics["total_rules_evaluated"] == 3
    assert statistics["matched_rules"] == 2
    assert statistics["unmatched_rules"] == 1
    assert statistics["approve_decisions"] == 1
    assert statistics["deny_decisions"] == 1
    assert statistics["manual_review_decisions"] == 0


def test_save_results(tmp_path: Path) -> None:
    """Verify rule evaluation results save as JSON."""
    engine = HealthcareRuleEngine()
    results = [engine.evaluate_rule(_matching_rule(), _features())]
    output_path = tmp_path / "results.json"

    assert engine.save_results(results, output_path) is True
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved[0]["rule_id"] == "RULE_001"


def test_load_rules(tmp_path: Path) -> None:
    """Verify JSON rules can be loaded."""
    engine = HealthcareRuleEngine()
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps([_matching_rule()]), encoding="utf-8")

    rules = engine.load_rules(rules_path)

    assert len(rules) == 1
    assert rules[0]["rule_id"] == "RULE_001"


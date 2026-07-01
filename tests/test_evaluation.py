"""Unit tests for healthcare AI evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from modules.evaluation import HealthcareAIEvaluator


def _retrieved_documents() -> list[dict[str, object]]:
    """Build retrieved documents.

    Args:
        None.

    Returns:
        Retrieved document list.
    """
    return [
        {
            "text": "Lumbar MRI is covered after six weeks of conservative therapy unless neurological deficits are present.",
            "metadata": {"source": "Billing_Policy.pdf", "page": 12},
        }
    ]


def _citations() -> list[dict[str, object]]:
    """Build citations.

    Args:
        None.

    Returns:
        Citation list.
    """
    return [{"source": "Billing_Policy.pdf", "page": 12}]


def _valid_rule() -> dict[str, object]:
    """Build valid rule.

    Args:
        None.

    Returns:
        Rule dictionary.
    """
    return {
        "rule_id": "RULE_001",
        "rule_type": "coverage",
        "conditions": [{"field": "therapy_weeks", "operator": ">=", "value": 6}],
        "decision": "approve",
        "source_text": "MRI covered after six weeks.",
        "confidence": 0.9,
    }


def _valid_features() -> dict[str, object]:
    """Build valid feature record.

    Args:
        None.

    Returns:
        Feature dictionary.
    """
    return {
        "icd_codes": ["M54.16"],
        "cpt_codes": ["72148"],
        "hcpcs_codes": ["J1100"],
        "prior_authorization_required": True,
        "therapy_weeks": 6,
        "contract_terms": {"allowed_amount": 750},
        "source_text": "M54.16 CPT 72148 HCPCS J1100",
        "confidence": 0.9,
    }


def _summary() -> dict[str, object]:
    """Build structured summary.

    Args:
        None.

    Returns:
        Summary dictionary.
    """
    return {
        "purpose": "Define MRI coverage.",
        "covered_services": ["Lumbar MRI"],
        "excluded_services": ["Experimental imaging"],
        "eligibility_criteria": ["Member has symptoms."],
        "medical_necessity": ["Six weeks therapy."],
        "prior_authorization": "Required.",
        "coding_requirements": ["CPT 72148"],
    }


def _explanation() -> dict[str, object]:
    """Build explanation output.

    Args:
        None.

    Returns:
        Explanation dictionary.
    """
    return {
        "final_decision": "approve",
        "plain_language_summary": "Claim approved because rules matched.",
        "decision_confidence": 0.9,
        "rule_based_reasoning": [{"rule_id": "RULE_001"}],
        "ml_reasoning": {"approval_probability": 0.93},
        "risk_explanation": [],
        "conflict_explanation": [],
        "supporting_citations": _citations(),
        "recommended_next_action": "Proceed with approval.",
        "audit_trail": {"explanation_version": "1.0"},
    }


def test_evaluate_good_rag_response() -> None:
    """Verify good RAG responses pass."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_rag_response(
        "When is lumbar MRI covered?",
        "Lumbar MRI is covered after six weeks of conservative therapy unless neurological deficits are present.",
        _retrieved_documents(),
        _citations(),
    )

    assert result["rag_quality_score"] >= 0.8
    assert result["hallucination_risk"] == "Low"


def test_evaluate_bad_rag_response() -> None:
    """Verify unsupported RAG responses fail or require review."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_rag_response(
        "When is lumbar MRI covered?",
        "Lumbar MRI is always approved immediately with CPT 99999 and pays $5000.",
        _retrieved_documents(),
        _citations(),
    )

    assert result["rag_quality_score"] < 0.7
    assert result["hallucination_risk"] in {"Medium", "High"}


def test_evaluate_missing_citations() -> None:
    """Verify missing citations receive zero citation score."""
    evaluator = HealthcareAIEvaluator()

    score = evaluator.evaluate_citation_quality("Answer", [], _retrieved_documents())

    assert score == 0.0


def test_evaluate_grounded_answer() -> None:
    """Verify grounded answer support is detected."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_groundedness(
        "Lumbar MRI is covered after six weeks of conservative therapy.",
        _retrieved_documents(),
    )

    assert result["groundedness_score"] == 1.0


def test_evaluate_hallucinated_answer() -> None:
    """Verify hallucinated specifics are flagged."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_hallucination_risk(
        "Lumbar MRI uses CPT 99999 and pays $5000.",
        _retrieved_documents(),
    )

    assert result["hallucination_risk"] in {"Medium", "High"}
    assert result["risk_reasons"]


def test_evaluate_valid_rules() -> None:
    """Verify valid rules score well."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_rule_extraction([_valid_rule()])

    assert result["valid_rules"] == 1
    assert result["rule_quality_score"] == 1.0


def test_evaluate_invalid_rules() -> None:
    """Verify invalid rules are reported."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_rule_extraction([{"rule_id": "BAD"}])

    assert result["invalid_rules"] == 1
    assert result["schema_errors"]


def test_evaluate_valid_features() -> None:
    """Verify valid features score well."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_feature_extraction(_valid_features())

    assert result["valid_features"] == 1
    assert result["feature_quality_score"] > 0.8


def test_evaluate_invalid_features() -> None:
    """Verify invalid feature formats are reported."""
    evaluator = HealthcareAIEvaluator()
    features = _valid_features()
    features["cpt_codes"] = ["BAD"]

    result = evaluator.evaluate_feature_extraction(features)

    assert result["format_errors"]
    assert result["feature_quality_score"] < 1.0


def test_evaluate_summary_completeness() -> None:
    """Verify complete summaries score well."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_summary(_summary())

    assert result["summary_quality_score"] == 1.0
    assert result["issues"] == []


def test_evaluate_explanation_quality() -> None:
    """Verify explanation quality scoring."""
    evaluator = HealthcareAIEvaluator()

    result = evaluator.evaluate_decision_explanation(_explanation(), {"claim_decision": "approve"})

    assert result["explanation_quality_score"] > 0.8
    assert result["issues"] == []


def test_evaluate_full_pipeline() -> None:
    """Verify full pipeline evaluation returns scores."""
    evaluator = HealthcareAIEvaluator()
    outputs = {
        "rag_response": {
            "question": "When is lumbar MRI covered?",
            "answer": "Lumbar MRI is covered after six weeks of conservative therapy.",
            "retrieved_documents": _retrieved_documents(),
            "citations": _citations(),
        },
        "rules": [_valid_rule()],
        "features": [_valid_features()],
        "summary": _summary(),
        "decision": {"claim_decision": "approve"},
        "explanation": _explanation(),
    }

    result = evaluator.evaluate_pipeline(outputs)

    assert result["overall_score"] > 0.7
    assert result["passed"] is True


def test_generate_evaluation_report() -> None:
    """Verify consolidated report generation."""
    evaluator = HealthcareAIEvaluator()

    report = evaluator.generate_evaluation_report({"rag_quality_score": 0.9, "issues": []})

    assert report["report_id"] == "EVAL_REPORT_001"
    assert report["overall_score"] == 0.9


def test_export_report_json(tmp_path: Path) -> None:
    """Verify JSON report export."""
    evaluator = HealthcareAIEvaluator()
    report = evaluator.generate_evaluation_report({"rag_quality_score": 0.9, "issues": []})
    output_path = tmp_path / "report.json"

    assert evaluator.export_report(report, output_path) is True
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert exported["overall_score"] == 0.9


def test_export_report_markdown(tmp_path: Path) -> None:
    """Verify Markdown report export."""
    evaluator = HealthcareAIEvaluator()
    report = evaluator.generate_evaluation_report({"rag_quality_score": 0.9, "issues": []})
    output_path = tmp_path / "report.md"

    assert evaluator.export_report(report, output_path) is True
    assert "# Healthcare AI Evaluation Report" in output_path.read_text(encoding="utf-8")


def test_evaluation_statistics() -> None:
    """Verify evaluation statistics aggregation."""
    evaluator = HealthcareAIEvaluator()

    statistics = evaluator.get_evaluation_statistics(
        [
            {"overall_score": 0.8, "rag_score": 0.9, "passed": True},
            {"overall_score": 0.6, "rag_score": 0.7, "passed": False},
        ]
    )

    assert statistics["total_evaluations"] == 2
    assert statistics["average_overall_score"] == 0.7
    assert statistics["pass_rate"] == 0.5


def test_empty_inputs() -> None:
    """Verify empty inputs return structured low scores."""
    evaluator = HealthcareAIEvaluator()

    rag = evaluator.evaluate_rag_response("", "", [], [])
    pipeline = evaluator.evaluate_pipeline({})

    assert rag["rag_quality_score"] == 0.0
    assert pipeline["passed"] is False


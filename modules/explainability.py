"""Explainability and audit trails for healthcare claim decisions.

This module turns claim decision outputs, rule evaluations, ML prediction
signals, features, and optional citations into transparent, deterministic, and
auditable explanations for healthcare payment integrity workflows.

Example:
    ```python
    from modules.explainability import ClaimExplainabilityEngine

    explainer = ClaimExplainabilityEngine()

    explanation = explainer.generate_explanation(
        decision_result=decision_result,
        rule_results=rule_results,
        ml_prediction=ml_prediction,
        features=features,
        citations=citations
    )

    print(explanation["plain_language_summary"])
    ```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator


class ExplanationInputConfidence(BaseModel):
    """Schema for validating confidence-like explanation inputs."""

    value: float

    @field_validator("value")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Validate confidence values.

        Args:
            value: Candidate confidence value.

        Returns:
            Valid confidence value.

        Raises:
            ValueError: If confidence is outside [0, 1].
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value


class ClaimExplainabilityEngine:
    """Generates human-readable explanations and audit trails."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the explainability engine.

        Args:
            logger: Optional logger instance.

        Returns:
            None.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.explanation_version = "1.0"

    def generate_explanation(
        self,
        decision_result: dict[str, Any],
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
        citations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate a complete claim decision explanation.

        Args:
            decision_result: Final claim decision output.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.
            citations: Optional supporting source citations.

        Returns:
            Explanation dictionary with plain-language summary, rule reasoning,
            ML reasoning, risks, conflicts, citations, missing information,
            next action, and audit trail.
        """
        self.logger.info("Explanation generation started.")
        validation_errors = self._validate_inputs(
            decision_result,
            rule_results,
            ml_prediction,
            features,
        )
        if validation_errors:
            return self._error_explanation(validation_errors)

        citations = citations or []
        rule_explanation = self.generate_rule_explanation(rule_results)
        ml_explanation = self.generate_ml_explanation(ml_prediction)
        conflicts = decision_result.get("conflicts", [])
        risk_flags = decision_result.get("risk_flags", [])
        conflict_explanation = self.generate_conflict_explanation(conflicts)
        risk_explanation = self.generate_risk_explanation(risk_flags)
        missing_information = self._detect_missing_information(features)
        explanation_confidence = self.calculate_explanation_confidence(
            decision_result,
            rule_results,
            ml_prediction,
        )
        final_decision = str(decision_result.get("claim_decision", "manual_review"))
        explanation = {
            "final_decision": final_decision,
            "plain_language_summary": self._build_plain_language_summary(
                decision_result,
                rule_results,
                ml_prediction,
                missing_information,
            ),
            "executive_explanation": self._build_executive_explanation(
                decision_result,
                rule_results,
                ml_prediction,
                missing_information,
            ),
            "decision_confidence": explanation_confidence,
            "rule_based_reasoning": rule_explanation,
            "top_supporting_reasons": self._top_supporting_reasons(decision_result),
            "decision_drivers": self._decision_driver_sections(
                decision_result,
                rule_explanation,
                risk_explanation,
                missing_information,
            ),
            "ml_reasoning": ml_explanation,
            "risk_explanation": risk_explanation,
            "conflict_explanation": conflict_explanation,
            "supporting_citations": self._normalize_citations(citations),
            "missing_information": missing_information,
            "confidence_rationale": self._confidence_rationale(
                explanation_confidence,
                decision_result,
                rule_results,
                ml_prediction,
                missing_information,
            ),
            "recommended_next_action": self._recommended_next_action(final_decision),
            "audit_trail": self.generate_audit_trail(
                decision_result,
                rule_results,
                ml_prediction,
                features,
                citations,
            ),
        }
        self.logger.info("Explanation generated.")
        return explanation

    def generate_rule_explanation(
        self,
        rule_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Explain rule evaluation results.

        Args:
            rule_results: Rule engine evaluation results.

        Returns:
            List of rule reasoning objects.
        """
        if not isinstance(rule_results, list):
            self.logger.warning("Rule results are missing or invalid.")
            return []

        explanations: list[dict[str, Any]] = []
        for result in rule_results:
            if not isinstance(result, dict):
                continue
            conditions = [
                self._explain_condition(condition)
                for condition in result.get("conditions_evaluated", [])
                if isinstance(condition, dict)
            ]
            explanations.append(
                {
                    "rule_id": result.get("rule_id", ""),
                    "matched": bool(result.get("matched", False)),
                    "decision": result.get("decision", ""),
                    "reason": result.get("reason", ""),
                    "confidence": self._safe_probability(result.get("confidence", 0.0)),
                    "conditions": conditions,
                    "impact": self._rule_impact(result),
                }
            )
        return explanations

    def generate_ml_explanation(self, ml_prediction: dict[str, Any]) -> dict[str, Any]:
        """Explain ML prediction contribution.

        Args:
            ml_prediction: ML prediction output.

        Returns:
            Structured ML explanation.
        """
        if not isinstance(ml_prediction, dict):
            return {}

        approval = self._safe_probability(ml_prediction.get("approval_probability", 0.0))
        fraud = self._safe_probability(ml_prediction.get("fraud_risk", 0.0))
        necessity = self._safe_probability(ml_prediction.get("medical_necessity_score", 0.0))
        confidence = self._safe_probability(ml_prediction.get("model_confidence", 0.0))
        predicted_approval = bool(ml_prediction.get("predicted_approval", False))
        return {
            "approval_probability": approval,
            "approval_probability_explanation": (
                f"The ML model estimated a {approval:.0%} approval probability."
            ),
            "fraud_risk": fraud,
            "fraud_risk_explanation": self._risk_level_text(fraud, "fraud risk"),
            "medical_necessity_score": necessity,
            "medical_necessity_explanation": self._risk_level_text(
                necessity,
                "medical necessity support",
            ),
            "model_confidence": confidence,
            "predicted_approval": predicted_approval,
            "ml_contribution": (
                "ML supports approval."
                if predicted_approval
                else "ML does not support automatic approval."
            ),
        }

    def generate_conflict_explanation(self, conflicts: list[str]) -> list[dict[str, str]]:
        """Explain detected rule/ML/data conflicts.

        Args:
            conflicts: Conflict descriptions from claim decision module.

        Returns:
            List of structured conflict explanations.
        """
        if not conflicts:
            return []
        return [
            {
                "conflict": str(conflict),
                "explanation": (
                    "This conflict should be reviewed because rule-based and "
                    "predictive signals do not fully align."
                ),
            }
            for conflict in conflicts
        ]

    def generate_risk_explanation(self, risk_flags: list[str]) -> list[dict[str, str]]:
        """Explain claim risk flags.

        Args:
            risk_flags: Risk flag descriptions.

        Returns:
            List of risk explanations.
        """
        if not risk_flags:
            return []
        return [
            {
                "risk_flag": str(flag),
                "explanation": self._risk_flag_text(str(flag)),
            }
            for flag in risk_flags
        ]

    def calculate_explanation_confidence(
        self,
        decision_result: dict[str, Any],
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
    ) -> float:
        """Calculate explanation confidence.

        Args:
            decision_result: Final claim decision output.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.

        Returns:
            Explanation confidence score between 0 and 1.
        """
        decision_confidence = self._safe_probability(decision_result.get("confidence", 0.0))
        model_confidence = self._safe_probability(ml_prediction.get("model_confidence", 0.0))
        matched_rules = sum(
            1 for result in rule_results if isinstance(result, dict) and result.get("matched") is True
        )
        failed_rules = sum(
            1 for result in rule_results if isinstance(result, dict) and result.get("matched") is not True
        )
        conflicts = decision_result.get("conflicts", [])
        confidence = (0.55 * decision_confidence) + (0.30 * model_confidence)
        confidence += min(0.10, matched_rules * 0.03)
        confidence -= min(0.20, failed_rules * 0.04)
        confidence -= min(0.25, len(conflicts) * 0.08 if isinstance(conflicts, list) else 0.0)
        calculated = self._clamp(confidence)
        self.logger.info("Explanation confidence calculated: %.2f", calculated)
        return round(calculated, 2)

    def generate_audit_trail(
        self,
        decision_result: dict[str, Any],
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
        citations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate structured audit trail.

        Args:
            decision_result: Final claim decision output.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.
            citations: Optional citations used to support explanation.

        Returns:
            Structured audit trail.
        """
        audit_trail = {
            "decision_inputs": self._redact_decision_inputs(decision_result),
            "rules_evaluated": rule_results if isinstance(rule_results, list) else [],
            "ml_outputs": ml_prediction if isinstance(ml_prediction, dict) else {},
            "features_used": self._redact_features(features),
            "citations_used": self._normalize_citations(citations or []),
            "decision_timestamp": "deterministic-not-recorded",
            "explanation_version": self.explanation_version,
        }
        self.logger.info("Audit trail generated.")
        return audit_trail

    def export_explanation(
        self,
        explanation: dict[str, Any],
        output_path: str | Path,
    ) -> bool:
        """Export explanation as JSON, Markdown, or text.

        Args:
            explanation: Explanation dictionary.
            output_path: Destination path. ``.md`` writes Markdown, ``.txt``
                writes text, and all other extensions write JSON.

        Returns:
            ``True`` when exported successfully; otherwise ``False``.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            suffix = path.suffix.lower()
            if suffix == ".md":
                path.write_text(self._to_markdown(explanation), encoding="utf-8")
            elif suffix == ".txt":
                path.write_text(self._to_text(explanation), encoding="utf-8")
            else:
                path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")
            self.logger.info("Export completed: %s", path)
            return True
        except (OSError, TypeError) as error:
            self.logger.error("Failed to export explanation to %s: %s", path, error)
            return False

    def get_explanation_statistics(
        self,
        explanations: list[dict[str, Any]],
    ) -> dict[str, int | float]:
        """Calculate explanation statistics.

        Args:
            explanations: Explanation dictionaries.

        Returns:
            Aggregate explanation statistics.
        """
        if not isinstance(explanations, list) or not explanations:
            return {
                "total_explanations": 0,
                "average_confidence": 0.0,
                "approve_explanations": 0,
                "deny_explanations": 0,
                "manual_review_explanations": 0,
                "explanations_with_conflicts": 0,
                "explanations_with_citations": 0,
            }

        valid = [item for item in explanations if isinstance(item, dict)]
        confidences = [
            float(item.get("decision_confidence", 0.0))
            for item in valid
            if isinstance(item.get("decision_confidence", 0.0), (int, float))
        ]
        return {
            "total_explanations": len(valid),
            "average_confidence": round(float(np.mean(confidences)), 2) if confidences else 0.0,
            "approve_explanations": self._count_final_decision(valid, "approve"),
            "deny_explanations": self._count_final_decision(valid, "deny"),
            "manual_review_explanations": self._count_final_decision(valid, "manual_review"),
            "explanations_with_conflicts": sum(1 for item in valid if item.get("conflict_explanation")),
            "explanations_with_citations": sum(1 for item in valid if item.get("supporting_citations")),
        }

    def _validate_inputs(
        self,
        decision_result: dict[str, Any],
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> list[str]:
        """Validate explanation inputs.

        Args:
            decision_result: Final claim decision output.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.

        Returns:
            List of validation errors.
        """
        errors: list[str] = []
        if not isinstance(decision_result, dict) or not decision_result:
            errors.append("decision_result must be a non-empty dictionary.")
        if not isinstance(rule_results, list):
            errors.append("rule_results must be a list.")
        if not isinstance(ml_prediction, dict) or not ml_prediction:
            errors.append("ml_prediction must be a non-empty dictionary.")
        if not isinstance(features, dict) or not features:
            errors.append("features must be a non-empty dictionary.")
        for field in ("confidence", "decision_score"):
            if isinstance(decision_result, dict) and field in decision_result:
                try:
                    ExplanationInputConfidence(value=float(decision_result[field]))
                except (ValueError, ValidationError) as error:
                    errors.append(f"decision_result.{field}: {error}")
        return errors

    def _build_plain_language_summary(
        self,
        decision_result: dict[str, Any],
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        missing_information: list[str],
    ) -> str:
        """Build deterministic plain-language summary.

        Args:
            decision_result: Final claim decision output.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            missing_information: Missing information descriptions.

        Returns:
            Human-readable explanation summary.
        """
        decision = str(decision_result.get("claim_decision", "manual_review"))
        reasons = decision_result.get("reasons", [])
        first_reasons = ", ".join(str(reason) for reason in reasons[:3]) if reasons else "the available evidence was reviewed"
        approval = self._safe_probability(ml_prediction.get("approval_probability", 0.0))
        fraud = self._safe_probability(ml_prediction.get("fraud_risk", 0.0))
        matched = sum(1 for result in rule_results if isinstance(result, dict) and result.get("matched") is True)
        failed = sum(1 for result in rule_results if isinstance(result, dict) and result.get("matched") is not True)
        summary = (
            f"The claim is recommended for {decision.replace('_', ' ')} because "
            f"{first_reasons}. Rule review found {matched} matched rule(s) and "
            f"{failed} failed rule(s). The ML model estimated a {approval:.0%} "
            f"approval probability with {fraud:.0%} fraud risk."
        )
        if missing_information:
            summary += f" Missing information includes: {', '.join(missing_information)}."
        return summary

    def _build_executive_explanation(
        self,
        decision_result: dict[str, Any],
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        missing_information: list[str],
    ) -> str:
        """Build a concise 3-5 sentence executive explanation.

        Args:
            decision_result: Final claim decision output.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            missing_information: Missing information descriptions.

        Returns:
            Professional executive explanation paragraph.
        """
        decision = str(decision_result.get("claim_decision", "manual_review")).replace("_", " ")
        approval = self._safe_probability(ml_prediction.get("approval_probability", 0.0))
        fraud = self._safe_probability(ml_prediction.get("fraud_risk", 0.0))
        necessity = self._safe_probability(ml_prediction.get("medical_necessity_score", 0.0))
        matched = sum(1 for result in rule_results if isinstance(result, dict) and result.get("matched") is True)
        failed = sum(1 for result in rule_results if isinstance(result, dict) and result.get("matched") is not True)
        top_reasons = self._top_supporting_reasons(decision_result)

        sentences = [
            f"The claim is recommended for {decision} after combining rule outcomes, structured claim features, and machine-learning support signals.",
            f"Rule review found {matched} matched rule(s) and {failed} failed rule(s), which shaped the decision more strongly than any single model score.",
            f"The ML layer estimated {approval:.0%} approval probability, {fraud:.0%} fraud risk, and {necessity:.0%} medical-necessity support.",
        ]
        if top_reasons:
            polished_reasons = [self._polish_reason_fragment(reason) for reason in top_reasons[:3]]
            sentences.append(f"The most important drivers were {', '.join(polished_reasons)}.")
        if missing_information:
            sentences.append(
                f"Missing or incomplete inputs that limited confidence included {', '.join(missing_information)}."
            )
        return " ".join(sentences[:5])

    def _top_supporting_reasons(self, decision_result: dict[str, Any]) -> list[str]:
        """Return the top clean supporting reasons.

        Args:
            decision_result: Final claim decision output.

        Returns:
            Up to three normalized reason strings.
        """
        reasons = decision_result.get("reasons", [])
        if not isinstance(reasons, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for reason in reasons:
            if not isinstance(reason, str):
                continue
            cleaned = reason.strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
            if len(normalized) == 3:
                break
        return normalized

    def _decision_driver_sections(
        self,
        decision_result: dict[str, Any],
        rule_explanation: list[dict[str, Any]],
        risk_explanation: list[dict[str, str]],
        missing_information: list[str],
    ) -> dict[str, list[str]]:
        """Build clean executive sections for the UI.

        Args:
            decision_result: Final claim decision output.
            rule_explanation: Structured rule explanations.
            risk_explanation: Structured risk explanations.
            missing_information: Missing input descriptions.

        Returns:
            Dictionary of executive explanation sections.
        """
        final_decision = str(decision_result.get("claim_decision", "manual_review"))
        why_decided: list[str] = []
        evidence_supported: list[str] = []
        if final_decision == "deny":
            why_decided.append("The claim was denied because higher-priority risk, completeness, or rule-failure signals outweighed approval indicators.")
        elif final_decision == "manual_review":
            why_decided.append("The claim was routed to manual review because the available signals were not strong enough for straight-through automation.")
        else:
            why_decided.append("The claim was approved because rule, feature, and ML signals were sufficiently aligned.")

        for item in rule_explanation:
            if item.get("matched") is True:
                rule_id = item.get("rule_id", "Rule")
                reason = item.get("reason", "")
                if reason:
                    evidence_supported.append(f"{rule_id}: {reason}")
            if len(evidence_supported) == 3:
                break
        if not evidence_supported:
            evidence_supported.append("The decision relied primarily on aggregated rule, feature, and ML signals.")

        missing_items = [f"Missing: {item}" for item in missing_information[:3]]
        if not missing_items:
            missing_items.append("No critical missing information was detected.")

        next_action = [self._recommended_next_action(final_decision)]
        if risk_explanation:
            next_action.append(risk_explanation[0].get("explanation", "Review flagged risks before finalizing action."))

        return {
            "why_decided": why_decided,
            "evidence_supported": evidence_supported,
            "missing_information": missing_items,
            "next_action_guidance": next_action,
        }

    def _confidence_rationale(
        self,
        explanation_confidence: float,
        decision_result: dict[str, Any],
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        missing_information: list[str],
    ) -> str:
        """Explain why explanation confidence is high or low.

        Args:
            explanation_confidence: Final explanation confidence.
            decision_result: Final claim decision output.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            missing_information: Missing input descriptions.

        Returns:
            Human-readable confidence rationale.
        """
        matched = sum(1 for result in rule_results if isinstance(result, dict) and result.get("matched") is True)
        failed = sum(1 for result in rule_results if isinstance(result, dict) and result.get("matched") is not True)
        model_confidence = self._safe_probability(ml_prediction.get("model_confidence", 0.0))
        conflicts = decision_result.get("conflicts", [])
        rationale_parts = [
            f"Explanation confidence is {explanation_confidence:.2f} because {matched} rule(s) matched and {failed} rule(s) failed.",
            f"The supporting ML confidence is {model_confidence:.2f}.",
        ]
        if conflicts:
            rationale_parts.append(f"Confidence is reduced by {len(conflicts)} cross-signal conflict(s).")
        if missing_information:
            rationale_parts.append("Confidence is also reduced by incomplete claim evidence.")
        return " ".join(rationale_parts)

    def _polish_reason_fragment(self, reason: str) -> str:
        """Convert a raw reason into cleaner sentence-fragment prose.

        Args:
            reason: Raw reason text.

        Returns:
            Polished sentence fragment.
        """
        cleaned = reason.strip().rstrip(".")
        replacements = {
            "The claim was denied because ": "",
            "Some approval-oriented rules matched, but ": "",
            "Medical necessity support was ": "medical necessity support was ",
            "Fraud risk remains ": "fraud risk remained ",
        }
        for source, target in replacements.items():
            if cleaned.startswith(source):
                cleaned = target + cleaned[len(source):]
                break
        if cleaned and cleaned[0].isupper():
            cleaned = cleaned[0].lower() + cleaned[1:]
        return cleaned

    def _explain_condition(self, condition: dict[str, Any]) -> dict[str, Any]:
        """Explain one evaluated rule condition.

        Args:
            condition: Condition evaluation dictionary.

        Returns:
            Condition explanation dictionary.
        """
        field = condition.get("field", "")
        expected = condition.get("expected", "")
        actual = condition.get("actual", None)
        matched = bool(condition.get("matched", False))
        return {
            "field": field,
            "expected": expected,
            "actual": actual,
            "matched": matched,
            "explanation": (
                f"{field} = {actual} satisfies the requirement {field} {expected}."
                if matched
                else f"{field} = {actual} does not satisfy the requirement {field} {expected}."
            ),
        }

    def _rule_impact(self, rule_result: dict[str, Any]) -> str:
        """Describe rule impact.

        Args:
            rule_result: Rule evaluation result.

        Returns:
            Rule impact text.
        """
        if rule_result.get("matched") is True:
            decision = str(rule_result.get("decision", "")).lower()
            if decision == "deny":
                return "This matched deny rule strongly supports denial."
            if decision == "approve":
                return "This matched approval rule supports approval."
            return "This matched rule supports review."
        return "This rule did not match and may reduce confidence."

    def _risk_level_text(self, value: float, label: str) -> str:
        """Build risk or score-level explanation.

        Args:
            value: Score value.
            label: Score label.

        Returns:
            Human-readable score explanation.
        """
        if value >= 0.75:
            level = "high"
        elif value >= 0.45:
            level = "moderate"
        else:
            level = "low"
        return f"The {label} is {level} at {value:.0%}."

    def _risk_flag_text(self, flag: str) -> str:
        """Build explanation for a risk flag.

        Args:
            flag: Risk flag text.

        Returns:
            Human-readable risk explanation.
        """
        lowered = flag.lower()
        if "authorization" in lowered:
            return "Prior authorization information should be verified before payment."
        if "documentation" in lowered:
            return "Required documentation is incomplete or missing."
        if "fraud" in lowered:
            return "The claim has elevated fraud risk and should be reviewed."
        if "cpt" in lowered:
            return "Procedure coding information is missing or incomplete."
        if "icd" in lowered:
            return "Diagnosis coding information is missing or incomplete."
        return "This risk factor may affect payment integrity and should be reviewed."

    def _detect_missing_information(self, features: dict[str, Any]) -> list[str]:
        """Detect missing decision-support information.

        Args:
            features: Extracted healthcare features.

        Returns:
            Missing information descriptions.
        """
        missing: list[str] = []
        if not features.get("icd_codes"):
            missing.append("ICD diagnosis codes")
        if not features.get("cpt_codes"):
            missing.append("CPT procedure codes")
        if features.get("documentation_complete") is False:
            missing.append("complete documentation")
        if (
            features.get("prior_authorization_required") is True
            and features.get("prior_authorization_obtained") is not True
        ):
            missing.append("prior authorization confirmation")
        return missing

    def _recommended_next_action(self, decision: str) -> str:
        """Return decision-specific recommended next action.

        Args:
            decision: Final decision value.

        Returns:
            Next action text.
        """
        return {
            "approve": "Proceed with approval.",
            "deny": "Deny claim or request corrected documentation.",
            "manual_review": "Route claim to a human reviewer for manual validation.",
        }.get(decision, "Review decision inputs before taking action.")

    def _normalize_citations(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize supporting citations.

        Args:
            citations: Citation dictionaries.

        Returns:
            Normalized citations.
        """
        normalized: list[dict[str, Any]] = []
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            normalized.append(
                {
                    "source": citation.get("source", ""),
                    "page": citation.get("page", citation.get("page_number", "")),
                    "text": citation.get("text", ""),
                }
            )
        return normalized

    def _redact_decision_inputs(self, decision_result: dict[str, Any]) -> dict[str, Any]:
        """Keep auditable decision fields while avoiding sensitive identifiers.

        Args:
            decision_result: Final claim decision output.

        Returns:
            Redacted decision input dictionary.
        """
        allowed = {
            "claim_decision",
            "decision_score",
            "confidence",
            "recommendation",
            "reasons",
            "risk_flags",
            "conflicts",
            "rule_summary",
            "ml_summary",
        }
        return {key: value for key, value in decision_result.items() if key in allowed}

    def _redact_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Keep useful feature fields while avoiding sensitive identifiers.

        Args:
            features: Extracted features.

        Returns:
            Redacted feature dictionary.
        """
        blocked = {"patient_name", "member_id", "claim_id", "raw_clinical_notes"}
        return {key: value for key, value in features.items() if key not in blocked}

    def _to_markdown(self, explanation: dict[str, Any]) -> str:
        """Convert explanation to Markdown.

        Args:
            explanation: Explanation dictionary.

        Returns:
            Markdown text.
        """
        lines = [
            "# Claim Decision Explanation",
            "",
            f"## Final Decision\n{explanation.get('final_decision', '')}",
            "",
            f"## Summary\n{explanation.get('plain_language_summary', '')}",
            "",
            f"## Recommended Next Action\n{explanation.get('recommended_next_action', '')}",
            "",
            "## Risk Flags",
        ]
        risk_items = explanation.get("risk_explanation", [])
        lines.extend(f"- {item.get('risk_flag', '')}: {item.get('explanation', '')}" for item in risk_items)
        if not risk_items:
            lines.append("- None")
        return "\n".join(lines).strip() + "\n"

    def _to_text(self, explanation: dict[str, Any]) -> str:
        """Convert explanation to plain text.

        Args:
            explanation: Explanation dictionary.

        Returns:
            Plain text explanation.
        """
        return (
            f"Final decision: {explanation.get('final_decision', '')}\n"
            f"Summary: {explanation.get('plain_language_summary', '')}\n"
            f"Next action: {explanation.get('recommended_next_action', '')}\n"
        )

    def _error_explanation(self, errors: list[str]) -> dict[str, Any]:
        """Build structured error explanation.

        Args:
            errors: Validation errors.

        Returns:
            Error explanation dictionary.
        """
        return {
            "final_decision": "error",
            "plain_language_summary": "Unable to generate explanation because required inputs are missing or invalid.",
            "decision_confidence": 0.0,
            "rule_based_reasoning": [],
            "ml_reasoning": {},
            "risk_explanation": [],
            "conflict_explanation": [],
            "supporting_citations": [],
            "missing_information": [],
            "recommended_next_action": "Review explanation inputs.",
            "audit_trail": {},
            "errors": errors,
        }

    def _safe_probability(self, value: Any) -> float:
        """Coerce a value into a clamped probability.

        Args:
            value: Candidate numeric value.

        Returns:
            Float clamped to [0, 1].
        """
        try:
            return self._clamp(float(value))
        except (TypeError, ValueError):
            return 0.0

    def _clamp(self, value: float) -> float:
        """Clamp a value to [0, 1].

        Args:
            value: Numeric value.

        Returns:
            Clamped float.
        """
        return float(np.clip(value, 0.0, 1.0))

    def _count_final_decision(self, explanations: list[dict[str, Any]], decision: str) -> int:
        """Count explanations by final decision.

        Args:
            explanations: Explanation dictionaries.
            decision: Final decision value.

        Returns:
            Count of matching explanations.
        """
        return sum(1 for item in explanations if item.get("final_decision") == decision)

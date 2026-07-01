"""Final claim decision support for healthcare payment integrity workflows.

This module combines rule engine outputs, machine learning prediction signals,
and extracted healthcare features to produce deterministic claim decision
support recommendations. It does not execute rules, train ML models, call LLMs,
or render any UI.

Example:
    ```python
    from modules.claim_decision import ClaimDecisionEngine

    engine = ClaimDecisionEngine()

    rule_results = [
        {
            "rule_id": "RULE_001",
            "matched": True,
            "decision": "approve",
            "reason": "Therapy requirement satisfied.",
            "confidence": 0.92
        }
    ]

    ml_prediction = {
        "approval_probability": 0.93,
        "predicted_approval": True,
        "fraud_risk": 0.12,
        "medical_necessity_score": 0.88,
        "model_confidence": 0.91
    }

    features = {
        "patient_age": 55,
        "icd_codes": ["M54.16"],
        "cpt_codes": ["72148"],
        "therapy_weeks": 6,
        "prior_authorization_required": True,
        "prior_authorization_obtained": True,
        "documentation_complete": True
    }

    decision = engine.make_decision(rule_results, ml_prediction, features)

    print(decision)
    ```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator


class MLPredictionSchema(BaseModel):
    """Schema for machine learning prediction signals."""

    approval_probability: float
    predicted_approval: bool
    fraud_risk: float
    medical_necessity_score: float
    model_confidence: float

    @field_validator(
        "approval_probability",
        "fraud_risk",
        "medical_necessity_score",
        "model_confidence",
    )
    @classmethod
    def validate_probability(cls, value: float) -> float:
        """Validate probability-like fields.

        Args:
            value: Candidate probability value.

        Returns:
            Validated probability value.

        Raises:
            ValueError: If value is outside [0, 1].
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("Value must be between 0 and 1.")
        return value


class RuleResultSchema(BaseModel):
    """Schema for one rule engine evaluation result."""

    rule_id: str = ""
    matched: bool = False
    decision: str = ""
    reason: str = ""
    confidence: float = 0.0
    rule_type: str = ""
    errors: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Validate rule confidence.

        Args:
            value: Candidate confidence value.

        Returns:
            Validated confidence value.

        Raises:
            ValueError: If value is outside [0, 1].
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("Rule confidence must be between 0 and 1.")
        return value


class ClaimDecisionEngine:
    """Combines rules, ML signals, and features into final claim decisions."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the claim decision engine.

        Args:
            logger: Optional logger instance.

        Returns:
            None.
        """
        self.logger = logger or logging.getLogger(__name__)

    def make_decision(
        self,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce one final claim decision.

        Args:
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features or claim attributes.

        Returns:
            Deterministic claim decision dictionary, or structured error output
            when inputs are invalid.
        """
        self.logger.info("Decision calculation started.")
        validation = self.validate_inputs(rule_results, ml_prediction, features)
        if not validation["valid"]:
            return self._error_decision(validation["errors"])

        score = self.calculate_decision_score(rule_results, ml_prediction, features)
        conflicts = self.detect_conflicts(rule_results, ml_prediction)
        risk_flags = self._detect_risk_flags(rule_results, ml_prediction, features)
        claim_decision = self.determine_decision(
            score,
            rule_results,
            ml_prediction,
            features,
        )
        if conflicts and claim_decision == "approve":
            claim_decision = "manual_review"

        confidence = self._calculate_confidence(
            rule_results,
            ml_prediction,
            conflicts,
            risk_flags,
            claim_decision,
        )
        reasons = self._build_reasons(
            claim_decision,
            score,
            rule_results,
            ml_prediction,
            features,
            conflicts,
            risk_flags,
        )
        decision = {
            "claim_decision": claim_decision,
            "decision_score": round(score, 2),
            "confidence": round(confidence, 2),
            "recommendation": self._recommendation_for_decision(claim_decision),
            "reasons": reasons,
            "risk_flags": risk_flags,
            "conflicts": conflicts,
            "rule_summary": self._summarize_rules(rule_results),
            "ml_summary": self._summarize_ml(ml_prediction),
        }
        self.logger.info("Decision score: %.2f", score)
        self.logger.info("Final decision: %s", claim_decision)
        if conflicts:
            self.logger.warning("Conflicts detected: %s", len(conflicts))
        if risk_flags:
            self.logger.warning("Risk flags detected: %s", len(risk_flags))
        return decision

    def make_batch_decisions(
        self,
        rule_results_list: list[list[dict[str, Any]]],
        ml_predictions: list[dict[str, Any]],
        feature_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Produce decisions for multiple claims.

        Args:
            rule_results_list: List of rule-result lists.
            ml_predictions: List of ML prediction dictionaries.
            feature_records: List of feature dictionaries.

        Returns:
            List of decision dictionaries, or structured error output when list
            lengths do not match.
        """
        if not (
            isinstance(rule_results_list, list)
            and isinstance(ml_predictions, list)
            and isinstance(feature_records, list)
        ):
            return self._batch_error("Batch inputs must all be lists.")

        lengths = {len(rule_results_list), len(ml_predictions), len(feature_records)}
        if len(lengths) != 1:
            return self._batch_error("Batch input lengths do not match.")

        decisions = [
            self.make_decision(rule_results, ml_prediction, features)
            for rule_results, ml_prediction, features in zip(
                rule_results_list,
                ml_predictions,
                feature_records,
            )
        ]
        statistics = self.get_decision_statistics(decisions)
        self.logger.info("Batch decision counts: %s", statistics)
        return decisions

    def calculate_decision_score(
        self,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> float:
        """Calculate weighted decision score.

        Args:
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.

        Returns:
            Weighted score clamped to [0, 1].
        """
        rule_score = self._calculate_rule_score(rule_results)
        ml_score = self._calculate_ml_score(ml_prediction)
        documentation_score = self._calculate_documentation_score(features)
        score = (0.60 * rule_score) + (0.30 * ml_score) + (0.10 * documentation_score)
        return self._clamp(score)

    def determine_decision(
        self,
        score: float,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> str:
        """Determine final decision from score and override logic.

        Args:
            score: Weighted decision score.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.

        Returns:
            One of ``approve``, ``deny``, or ``manual_review``.
        """
        if self._has_hard_deny(rule_results, ml_prediction, features):
            return "deny"
        if self._requires_manual_review(rule_results, ml_prediction, features):
            return "manual_review"
        if score >= 0.75:
            return "approve"
        if score <= 0.45:
            return "deny"
        return "manual_review"

    def detect_conflicts(
        self,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
    ) -> list[str]:
        """Detect conflicts between rule outputs and ML signals.

        Args:
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.

        Returns:
            List of conflict descriptions.
        """
        conflicts: list[str] = []
        matched_deny = self._matched_decisions(rule_results, {"deny", "denied"})
        matched_approve = self._matched_decisions(rule_results, {"approve", "approved"})
        failed_rules = self._failed_rules(rule_results)
        approval_probability = float(ml_prediction.get("approval_probability", 0.0))
        predicted_approval = bool(ml_prediction.get("predicted_approval", False))
        fraud_risk = float(ml_prediction.get("fraud_risk", 0.0))
        necessity = float(ml_prediction.get("medical_necessity_score", 0.0))

        if matched_deny and predicted_approval:
            conflicts.append("Rule engine recommends deny but ML predicts approval.")
        if matched_approve and not predicted_approval:
            conflicts.append("Rule engine approves but ML predicts denial.")
        if approval_probability >= 0.80 and len(failed_rules) >= 2:
            conflicts.append("ML approval probability is high but multiple rules failed.")
        if fraud_risk > 0.60 and matched_approve:
            conflicts.append("Fraud risk is high but approval rules passed.")
        if necessity < 0.50 and matched_approve:
            conflicts.append("Medical necessity score is low but approval rules matched.")
        if necessity >= 0.80 and matched_deny:
            conflicts.append("Medical necessity is high but deny rule matched.")
        return conflicts

    def validate_inputs(
        self,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate decision engine inputs.

        Args:
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.

        Returns:
            Validation dictionary with ``valid`` and ``errors``.
        """
        errors: list[str] = []
        if not isinstance(rule_results, list):
            errors.append("rule_results must be a list.")
        else:
            for index, rule_result in enumerate(rule_results):
                if not isinstance(rule_result, dict):
                    errors.append(f"rule_results[{index}] must be a dictionary.")
                    continue
                try:
                    RuleResultSchema.model_validate(rule_result)
                except ValidationError as error:
                    errors.extend(f"rule_results[{index}]: {item['msg']}" for item in error.errors())

        if not isinstance(ml_prediction, dict):
            errors.append("ml_prediction must be a dictionary.")
        else:
            try:
                MLPredictionSchema.model_validate(ml_prediction)
            except ValidationError as error:
                errors.extend(f"ml_prediction: {item['msg']}" for item in error.errors())

        if not isinstance(features, dict):
            errors.append("features must be a dictionary.")

        return {"valid": not errors, "errors": errors}

    def save_decisions(self, decisions: list[dict[str, Any]], output_path: str | Path) -> bool:
        """Save decisions as JSON.

        Args:
            decisions: Decision dictionaries.
            output_path: Destination JSON path.

        Returns:
            ``True`` when saved successfully; otherwise ``False``.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
            self.logger.info("Decisions saved: %s", path)
            return True
        except (OSError, TypeError) as error:
            self.logger.error("Failed to save decisions to %s: %s", path, error)
            return False

    def load_decisions(self, input_path: str | Path) -> list[dict[str, Any]]:
        """Load decisions from JSON.

        Args:
            input_path: Source JSON path.

        Returns:
            Loaded decision dictionaries, or an empty list on failure.
        """
        path = Path(input_path)
        if not path.exists():
            self.logger.error("Decision file does not exist: %s", path)
            return []
        try:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                self.logger.warning("Decision file is empty: %s", path)
                return []
            loaded = json.loads(content)
            return loaded if isinstance(loaded, list) else []
        except (json.JSONDecodeError, OSError) as error:
            self.logger.error("Failed to load decisions from %s: %s", path, error)
            return []

    def get_decision_statistics(self, decisions: list[dict[str, Any]]) -> dict[str, int | float]:
        """Calculate decision statistics.

        Args:
            decisions: Decision dictionaries.

        Returns:
            Aggregate decision statistics.
        """
        if not isinstance(decisions, list) or not decisions:
            return {
                "total_claims": 0,
                "approved": 0,
                "denied": 0,
                "manual_review": 0,
                "average_confidence": 0.0,
                "average_decision_score": 0.0,
                "high_risk_claims": 0,
            }

        valid_decisions = [decision for decision in decisions if isinstance(decision, dict)]
        confidence_values = [
            float(decision.get("confidence", 0.0))
            for decision in valid_decisions
            if isinstance(decision.get("confidence", 0.0), (int, float))
        ]
        score_values = [
            float(decision.get("decision_score", 0.0))
            for decision in valid_decisions
            if isinstance(decision.get("decision_score", 0.0), (int, float))
        ]
        return {
            "total_claims": len(valid_decisions),
            "approved": self._count_decision(valid_decisions, "approve"),
            "denied": self._count_decision(valid_decisions, "deny"),
            "manual_review": self._count_decision(valid_decisions, "manual_review"),
            "average_confidence": round(float(np.mean(confidence_values)), 2) if confidence_values else 0.0,
            "average_decision_score": round(float(np.mean(score_values)), 2) if score_values else 0.0,
            "high_risk_claims": self._count_high_risk_claims(valid_decisions),
        }

    def _calculate_rule_score(self, rule_results: list[dict[str, Any]]) -> float:
        """Calculate rule-derived score.

        Args:
            rule_results: Rule engine evaluation results.

        Returns:
            Rule score between 0 and 1.
        """
        if not rule_results:
            return 0.45

        score = 0.50
        for result in rule_results:
            matched = result.get("matched") is True
            decision = str(result.get("decision", "")).lower()
            rule_type = str(result.get("rule_type", "")).lower()
            confidence = self._clamp(float(result.get("confidence", 0.0)))
            if matched and decision in {"approve", "approved"}:
                score += 0.18 * max(confidence, 0.5)
            elif matched and decision in {"deny", "denied"}:
                score -= 0.55 * max(confidence, 0.5)
            elif matched and decision in {"review", "manual_review"}:
                score -= 0.05
            elif not matched:
                penalty = 0.12 if rule_type == "medical_necessity" else 0.08
                score -= penalty
        return self._clamp(score)

    def _calculate_ml_score(self, ml_prediction: dict[str, Any]) -> float:
        """Calculate ML-derived score.

        Args:
            ml_prediction: ML prediction output.

        Returns:
            ML score between 0 and 1.
        """
        approval = float(ml_prediction.get("approval_probability", 0.0))
        fraud = float(ml_prediction.get("fraud_risk", 0.0))
        necessity = float(ml_prediction.get("medical_necessity_score", 0.0))
        confidence = float(ml_prediction.get("model_confidence", 0.0))
        score = (0.45 * approval) + (0.25 * (1.0 - fraud)) + (0.20 * necessity) + (0.10 * confidence)
        return self._clamp(score)

    def _calculate_documentation_score(self, features: dict[str, Any]) -> float:
        """Calculate documentation and required-code score.

        Args:
            features: Extracted healthcare features.

        Returns:
            Documentation score between 0 and 1.
        """
        score = 0.50
        score += 0.25 if features.get("documentation_complete") is True else -0.20
        score += 0.10 if features.get("icd_codes") else -0.10
        score += 0.10 if features.get("cpt_codes") else -0.10
        if features.get("prior_authorization_required") is True:
            score += 0.15 if features.get("prior_authorization_obtained") is True else -0.35
        return self._clamp(score)

    def _has_hard_deny(
        self,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> bool:
        """Check hard deny conditions.

        Args:
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.

        Returns:
            Whether hard deny applies.
        """
        return bool(
            self._matched_decisions(rule_results, {"deny", "denied"})
            or float(ml_prediction.get("fraud_risk", 0.0)) > 0.75
            or not features.get("icd_codes")
            or not features.get("cpt_codes")
            or (
                features.get("prior_authorization_required") is True
                and features.get("prior_authorization_obtained") is not True
            )
        )

    def _requires_manual_review(
        self,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> bool:
        """Check manual review conditions.

        Args:
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.

        Returns:
            Whether manual review applies.
        """
        return bool(
            not rule_results
            or float(ml_prediction.get("model_confidence", 0.0)) < 0.55
            or features.get("documentation_complete") is False
            or any(result.get("decision") == "error" for result in rule_results)
        )

    def _detect_risk_flags(
        self,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
    ) -> list[str]:
        """Detect claim risk flags.

        Args:
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.

        Returns:
            Risk flag descriptions.
        """
        flags: list[str] = []
        if float(ml_prediction.get("fraud_risk", 0.0)) > 0.60:
            flags.append("Fraud risk is elevated.")
        if float(ml_prediction.get("medical_necessity_score", 0.0)) < 0.50:
            flags.append("Medical necessity score is low.")
        if features.get("documentation_complete") is False:
            flags.append("Required documentation is incomplete.")
        if not features.get("icd_codes"):
            flags.append("ICD codes are missing.")
        if not features.get("cpt_codes"):
            flags.append("CPT codes are missing.")
        if (
            features.get("prior_authorization_required") is True
            and features.get("prior_authorization_obtained") is not True
        ):
            flags.append("Prior authorization is required but not obtained.")
        if len(self._failed_rules(rule_results)) >= 2:
            flags.append("Multiple rules failed.")
        return self._deduplicate_preserve_order(flags)

    def _calculate_confidence(
        self,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        conflicts: list[str],
        risk_flags: list[str],
        decision: str,
    ) -> float:
        """Calculate final decision confidence.

        Args:
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            conflicts: Conflict descriptions.
            risk_flags: Risk flag descriptions.
            decision: Final decision.

        Returns:
            Confidence score between 0 and 1.
        """
        rule_confidences = [
            float(result.get("confidence", 0.0))
            for result in rule_results
            if isinstance(result.get("confidence", 0.0), (int, float))
        ]
        average_rule_confidence = float(np.mean(rule_confidences)) if rule_confidences else 0.45
        ml_confidence = float(ml_prediction.get("model_confidence", 0.0))
        matched_count = len([result for result in rule_results if result.get("matched") is True])
        failed_count = len(self._failed_rules(rule_results))
        confidence = (0.55 * average_rule_confidence) + (0.35 * ml_confidence)
        confidence += min(0.10, matched_count * 0.03)
        confidence -= min(0.20, failed_count * 0.05)
        confidence -= min(0.30, len(conflicts) * 0.10)
        confidence -= min(0.20, len(risk_flags) * 0.04)
        if decision == "deny" and self._matched_decisions(rule_results, {"deny", "denied"}):
            confidence += 0.08
        return self._clamp(confidence)

    def _build_reasons(
        self,
        decision: str,
        score: float,
        rule_results: list[dict[str, Any]],
        ml_prediction: dict[str, Any],
        features: dict[str, Any],
        conflicts: list[str],
        risk_flags: list[str],
    ) -> list[str]:
        """Build decision reasons.

        Args:
            decision: Final decision.
            score: Decision score.
            rule_results: Rule engine evaluation results.
            ml_prediction: ML prediction output.
            features: Extracted healthcare features.
            conflicts: Conflict descriptions.
            risk_flags: Risk flag descriptions.

        Returns:
            Reason descriptions.
        """
        matched_approve = self._matched_decisions(rule_results, {"approve", "approved"})
        matched_deny = self._matched_decisions(rule_results, {"deny", "denied"})
        failed_rules = self._failed_rules(rule_results)
        fraud_risk = float(ml_prediction.get("fraud_risk", 0.0))
        approval_probability = float(ml_prediction.get("approval_probability", 0.0))
        medical_necessity_score = float(ml_prediction.get("medical_necessity_score", 0.0))
        missing_icd = not bool(features.get("icd_codes"))
        missing_cpt = not bool(features.get("cpt_codes"))
        missing_prior_auth = (
            features.get("prior_authorization_required") is True
            and features.get("prior_authorization_obtained") is not True
        )

        reasons: list[str] = []

        if decision == "deny":
            if matched_deny:
                reasons.append("A deny or exclusion rule matched and overrode approval-oriented signals.")
            elif missing_prior_auth:
                reasons.append("The claim was denied because prior authorization was required but not obtained.")
            elif missing_icd or missing_cpt:
                reasons.append("The claim was denied because required coding evidence was incomplete.")
            elif fraud_risk > 0.75:
                reasons.append("The claim was denied because fraud risk exceeded the hard-stop threshold.")
            elif failed_rules:
                reasons.append("The claim was denied because multiple supporting rule conditions were not satisfied.")

            if matched_approve:
                reasons.append("Some approval-oriented rules matched, but higher-priority risk and completeness checks prevented approval.")
            if medical_necessity_score < 0.50:
                reasons.append("Medical necessity support was too weak to justify approval.")

        elif decision == "manual_review":
            if matched_approve:
                reasons.append("Some approval-oriented rules matched, but the evidence was not strong enough for straight-through approval.")
            if conflicts:
                reasons.append("Rule and ML signals are not fully aligned, so manual review is safer.")
            if 0.45 < score < 0.75:
                reasons.append("The combined decision score falls in the manual review range.")
            if features.get("documentation_complete") is False:
                reasons.append("Documentation needs human review before a final decision can be made.")

        else:
            if matched_approve:
                reasons.append("Coverage and supporting approval rules matched.")
            if features.get("prior_authorization_required") is True and features.get("prior_authorization_obtained") is True:
                reasons.append("Prior authorization requirements were satisfied.")
            if self._documentation_is_complete(features):
                reasons.append("Required coding and documentation elements are present.")
            if approval_probability >= 0.75:
                reasons.append("The ML model supports approval with a strong approval probability.")
            if fraud_risk <= 0.35:
                reasons.append("Fraud risk is low.")

        if failed_rules and decision != "deny":
            reasons.append("Some rules still failed and should be monitored.")
        if fraud_risk > 0.60 and decision != "approve":
            reasons.append("Fraud risk remains elevated.")
        if conflicts and decision == "approve":
            reasons.append("Approval was tempered by conflicting signals.")

        selected_risk_flags = [
            flag for flag in risk_flags
            if flag not in reasons and not self._is_reason_covered_by_flag(flag, reasons)
        ]
        reasons.extend(selected_risk_flags[:2])

        if not reasons:
            reasons.append(f"Decision determined by weighted score and {decision} threshold.")
        return self._deduplicate_preserve_order(reasons)

    def _summarize_rules(self, rule_results: list[dict[str, Any]]) -> dict[str, int]:
        """Summarize rule outcomes.

        Args:
            rule_results: Rule engine evaluation results.

        Returns:
            Rule summary dictionary.
        """
        return {
            "matched_rules": len([result for result in rule_results if result.get("matched") is True]),
            "failed_rules": len(self._failed_rules(rule_results)),
            "deny_rules": len(self._matched_decisions(rule_results, {"deny", "denied"})),
        }

    def _summarize_ml(self, ml_prediction: dict[str, Any]) -> dict[str, float]:
        """Summarize ML signals.

        Args:
            ml_prediction: ML prediction output.

        Returns:
            ML summary dictionary.
        """
        return {
            "approval_probability": float(ml_prediction.get("approval_probability", 0.0)),
            "fraud_risk": float(ml_prediction.get("fraud_risk", 0.0)),
            "medical_necessity_score": float(ml_prediction.get("medical_necessity_score", 0.0)),
        }

    def _matched_decisions(
        self,
        rule_results: list[dict[str, Any]],
        decisions: set[str],
    ) -> list[dict[str, Any]]:
        """Return matched rules with selected decisions.

        Args:
            rule_results: Rule engine evaluation results.
            decisions: Normalized decisions to match.

        Returns:
            Matching rule results.
        """
        return [
            result
            for result in rule_results
            if result.get("matched") is True and str(result.get("decision", "")).lower() in decisions
        ]

    def _failed_rules(self, rule_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return failed rule results.

        Args:
            rule_results: Rule engine evaluation results.

        Returns:
            Failed rule results.
        """
        return [result for result in rule_results if result.get("matched") is not True]

    def _recommendation_for_decision(self, decision: str) -> str:
        """Return recommendation text for a decision.

        Args:
            decision: Final decision.

        Returns:
            Recommendation string.
        """
        return {
            "approve": "Approve claim",
            "deny": "Deny claim",
            "manual_review": "Route claim for manual review",
        }.get(decision, "Unable to determine recommendation")

    def _documentation_is_complete(self, features: dict[str, Any]) -> bool:
        """Return whether documentation is complete enough to support approval messaging.

        Args:
            features: Extracted healthcare features.

        Returns:
            ``True`` when the documentation flag is complete and core coding
            evidence is present.
        """
        return bool(
            features.get("documentation_complete") is True
            and features.get("icd_codes")
            and features.get("cpt_codes")
        )

    def _is_reason_covered_by_flag(self, flag: str, reasons: list[str]) -> bool:
        """Return whether a risk flag is already covered semantically by reasons.

        Args:
            flag: Risk flag text.
            reasons: Existing reasons.

        Returns:
            ``True`` when the flag meaning is already present in reasons.
        """
        normalized_flag = flag.lower()
        normalized_reasons = " ".join(reason.lower() for reason in reasons)
        synonym_groups = {
            "fraud risk": ["fraud risk"],
            "medical necessity": ["medical necessity"],
            "prior authorization": ["prior authorization"],
            "icd codes": ["icd code", "coding evidence"],
            "cpt codes": ["cpt code", "coding evidence"],
            "documentation": ["documentation"],
            "multiple rules failed": ["multiple supporting rule conditions", "multiple rules failed"],
        }
        for anchor, synonyms in synonym_groups.items():
            if anchor in normalized_flag and any(synonym in normalized_reasons for synonym in synonyms):
                return True
        return False

    def _deduplicate_preserve_order(self, items: list[str]) -> list[str]:
        """Deduplicate text items while preserving order.

        Args:
            items: Text items to deduplicate.

        Returns:
            Order-preserving deduplicated list.
        """
        seen: set[str] = set()
        deduplicated: list[str] = []
        for item in items:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduplicated.append(item)
        return deduplicated

    def _error_decision(self, errors: list[str]) -> dict[str, Any]:
        """Build structured error output.

        Args:
            errors: Validation errors.

        Returns:
            Error decision dictionary.
        """
        return {
            "claim_decision": "error",
            "decision_score": 0.0,
            "confidence": 0.0,
            "recommendation": "Unable to make claim decision",
            "reasons": ["Invalid decision inputs."],
            "risk_flags": [],
            "conflicts": [],
            "rule_summary": {"matched_rules": 0, "failed_rules": 0, "deny_rules": 0},
            "ml_summary": {
                "approval_probability": 0.0,
                "fraud_risk": 0.0,
                "medical_necessity_score": 0.0,
            },
            "errors": errors,
        }

    def _batch_error(self, error: str) -> dict[str, Any]:
        """Build structured batch error output.

        Args:
            error: Error description.

        Returns:
            Batch error dictionary.
        """
        self.logger.error(error)
        return {"valid": False, "errors": [error], "decisions": []}

    def _count_decision(self, decisions: list[dict[str, Any]], decision_name: str) -> int:
        """Count decisions by name.

        Args:
            decisions: Decision dictionaries.
            decision_name: Decision value to count.

        Returns:
            Count of matching decisions.
        """
        return sum(1 for decision in decisions if decision.get("claim_decision") == decision_name)

    def _count_high_risk_claims(self, decisions: list[dict[str, Any]]) -> int:
        """Count high-risk claims.

        Args:
            decisions: Decision dictionaries.

        Returns:
            Count of high-risk decisions.
        """
        count = 0
        for decision in decisions:
            fraud_risk = float(decision.get("ml_summary", {}).get("fraud_risk", 0.0))
            if fraud_risk > 0.60 or decision.get("risk_flags"):
                count += 1
        return count

    def _clamp(self, value: float) -> float:
        """Clamp a numeric value to [0, 1].

        Args:
            value: Numeric value.

        Returns:
            Clamped float.
        """
        return float(np.clip(value, 0.0, 1.0))

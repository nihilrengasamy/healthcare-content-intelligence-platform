"""Evaluation utilities for healthcare AI platform outputs.

This module evaluates RAG responses, citation quality, groundedness,
hallucination risk, rule extraction quality, feature extraction quality,
summary completeness, decision explanation quality, and full pipeline quality.
It is deterministic by default and does not call external services.

Example:
    ```python
    from modules.evaluation import HealthcareAIEvaluator

    evaluator = HealthcareAIEvaluator()

    rag_eval = evaluator.evaluate_rag_response(
        question="When is lumbar MRI covered?",
        answer="Lumbar MRI is covered after six weeks of conservative therapy.",
        retrieved_documents=[
            {
                "text": "Lumbar MRI is covered after six weeks of conservative therapy.",
                "metadata": {
                    "source": "Billing_Policy.pdf",
                    "page": 12
                }
            }
        ],
        citations=[
            {
                "source": "Billing_Policy.pdf",
                "page": 12
            }
        ]
    )

    print(rag_eval)
    ```
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


class EvaluationScore(BaseModel):
    """Schema for validation of score values."""

    value: float

    @field_validator("value")
    @classmethod
    def validate_score(cls, value: float) -> float:
        """Validate score ranges.

        Args:
            value: Candidate score.

        Returns:
            Validated score.

        Raises:
            ValueError: If score is outside [0, 1].
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("Score must be between 0 and 1.")
        return value


class RuleQualitySchema(BaseModel):
    """Minimal schema for extracted rule quality checks."""

    rule_id: str
    rule_type: str
    conditions: list[dict[str, Any]]
    decision: str
    source_text: str = ""
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Validate confidence range.

        Args:
            value: Candidate confidence.

        Returns:
            Valid confidence.

        Raises:
            ValueError: If confidence is outside [0, 1].
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value


class FeatureQualitySchema(BaseModel):
    """Minimal schema for extracted feature quality checks."""

    icd_codes: list[str] = Field(default_factory=list)
    cpt_codes: list[str] = Field(default_factory=list)
    hcpcs_codes: list[str] = Field(default_factory=list)
    prior_authorization_required: bool | None = None
    therapy_weeks: int | None = None
    contract_terms: dict[str, Any] = Field(default_factory=dict)
    source_text: str = ""
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Validate confidence range.

        Args:
            value: Candidate confidence.

        Returns:
            Valid confidence.

        Raises:
            ValueError: If confidence is outside [0, 1].
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value


class HealthcareAIEvaluator:
    """Evaluates quality, reliability, and safety of healthcare AI outputs."""

    def __init__(
        self,
        enable_llm_judge: bool = False,
        llm: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            enable_llm_judge: Whether optional LLM-as-judge is enabled.
            llm: Optional LLM object for future qualitative evaluation.
            logger: Optional logger instance.

        Returns:
            None.
        """
        self.enable_llm_judge = enable_llm_judge
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def evaluate_rag_response(
        self,
        question: str,
        answer: str,
        retrieved_documents: list[dict[str, Any]],
        citations: list[dict[str, Any]] | None = None,
        expected_answer: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate RAG answer quality.

        Args:
            question: User question.
            answer: Generated answer.
            retrieved_documents: Retrieved source chunks.
            citations: Optional citations.
            expected_answer: Optional reference answer.

        Returns:
            Structured RAG evaluation result.
        """
        self.logger.info("Evaluation started for RAG response.")
        if not answer or not isinstance(answer, str):
            return {
                "rag_quality_score": 0.0,
                "groundedness_score": 0.0,
                "citation_score": 0.0,
                "hallucination_risk": "High",
                "answer_completeness": 0.0,
                "issues": ["Missing answer."],
                "recommendation": "Fail",
            }

        groundedness = self.evaluate_groundedness(answer, retrieved_documents)
        citation_score = self.evaluate_citation_quality(answer, citations or [], retrieved_documents)
        hallucination = self.evaluate_hallucination_risk(answer, retrieved_documents)
        completeness = self._answer_completeness(answer, question, expected_answer)
        rag_score = self._clamp(
            (0.40 * groundedness["groundedness_score"])
            + (0.25 * citation_score)
            + (0.25 * completeness)
            + (0.10 * (1.0 - hallucination["risk_score"]))
        )
        issues = []
        issues.extend(groundedness["unsupported_claims"])
        issues.extend(hallucination["risk_reasons"])
        if citation_score < 0.6:
            issues.append("Citation quality is low.")
        recommendation = "Pass" if rag_score >= 0.70 and hallucination["hallucination_risk"] != "High" else "Review"
        if rag_score < 0.50:
            recommendation = "Fail"
        result = {
            "rag_quality_score": round(rag_score, 2),
            "groundedness_score": groundedness["groundedness_score"],
            "citation_score": round(citation_score, 2),
            "hallucination_risk": hallucination["hallucination_risk"],
            "answer_completeness": round(completeness, 2),
            "issues": issues,
            "recommendation": recommendation,
        }
        self.logger.info("RAG evaluation completed.")
        return result

    def evaluate_citation_quality(
        self,
        answer: str,
        citations: list[dict[str, Any]] | None,
        retrieved_documents: list[dict[str, Any]],
    ) -> float:
        """Evaluate citation quality.

        Args:
            answer: Generated answer.
            citations: Citation dictionaries.
            retrieved_documents: Retrieved source chunks.

        Returns:
            Citation quality score from 0 to 1.
        """
        if not citations:
            return 0.0
        retrieved_keys = {
            (
                str(document.get("metadata", {}).get("source", "")),
                document.get("metadata", {}).get("page", document.get("metadata", {}).get("page_number", "")),
            )
            for document in retrieved_documents
        }
        score = 0.0
        for citation in citations:
            source = str(citation.get("source", ""))
            page = citation.get("page", citation.get("page_number", ""))
            if source:
                score += 0.25
            if page != "":
                score += 0.25
            if (source, page) in retrieved_keys:
                score += 0.35
            if self._term_overlap(answer, self._retrieved_text_for_source(retrieved_documents, source)) > 0.2:
                score += 0.15
        return self._clamp(score / max(len(citations), 1))

    def evaluate_groundedness(
        self,
        answer: str,
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate whether an answer is supported by retrieved context.

        Args:
            answer: Generated answer.
            retrieved_documents: Retrieved source chunks.

        Returns:
            Groundedness result with supported and unsupported claims.
        """
        context = self._combined_retrieved_text(retrieved_documents)
        if not answer or not context:
            return {
                "groundedness_score": 0.0,
                "unsupported_claims": ["Missing answer or retrieved context."],
                "supported_claims": [],
            }
        answer_sentences = self._sentences(answer)
        supported: list[str] = []
        unsupported: list[str] = []
        for sentence in answer_sentences:
            overlap = self._term_overlap(sentence, context)
            if overlap >= 0.45 or self._all_critical_tokens_supported(sentence, context):
                supported.append(sentence)
            else:
                unsupported.append(sentence)
        score = len(supported) / len(answer_sentences) if answer_sentences else 0.0
        return {
            "groundedness_score": round(score, 2),
            "unsupported_claims": unsupported,
            "supported_claims": supported,
        }

    def evaluate_hallucination_risk(
        self,
        answer: str,
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate hallucination risk.

        Args:
            answer: Generated answer.
            retrieved_documents: Retrieved source chunks.

        Returns:
            Hallucination risk result.
        """
        context = self._combined_retrieved_text(retrieved_documents)
        reasons: list[str] = []
        if not context:
            reasons.append("No retrieved source context is available.")
        unsupported_codes = self._unsupported_pattern_values(answer, context, self._code_pattern())
        unsupported_amounts = self._unsupported_pattern_values(answer, context, r"\$\s?\d[\d,]*(?:\.\d{2})?")
        if unsupported_codes:
            reasons.append(f"Unsupported specific codes: {', '.join(unsupported_codes)}.")
        if unsupported_amounts:
            reasons.append(f"Unsupported dollar amounts: {', '.join(unsupported_amounts)}.")
        groundedness = self.evaluate_groundedness(answer, retrieved_documents)["groundedness_score"]
        risk_score = self._clamp((1.0 - groundedness) + (0.15 * len(reasons)))
        risk = "Low" if risk_score < 0.35 else "Medium" if risk_score < 0.70 else "High"
        return {
            "hallucination_risk": risk,
            "risk_score": round(risk_score, 2),
            "risk_reasons": reasons,
        }

    def evaluate_rule_extraction(
        self,
        extracted_rules: list[dict[str, Any]],
        expected_rules: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate extracted JSON rules.

        Args:
            extracted_rules: Extracted rule dictionaries.
            expected_rules: Optional expected rule dictionaries.

        Returns:
            Rule extraction quality result.
        """
        if not isinstance(extracted_rules, list):
            return self._rule_eval_empty(["Rules must be a list."])
        valid_rules = 0
        schema_errors: list[str] = []
        missing_required: list[str] = []
        for index, rule in enumerate(extracted_rules):
            try:
                RuleQualitySchema.model_validate(rule)
                valid_rules += 1
            except ValidationError as error:
                schema_errors.extend(f"rule[{index}]: {item['msg']}" for item in error.errors())
                missing_required.extend(self._missing_rule_fields(rule))
        invalid_rules = len(extracted_rules) - valid_rules
        schema_score = valid_rules / len(extracted_rules) if extracted_rules else 0.0
        precision, recall, f1 = self._matching_metrics(extracted_rules, expected_rules, "rule_type")
        score = schema_score if f1 is None else (0.6 * schema_score) + (0.4 * f1)
        self.logger.info("Rule evaluation completed.")
        return {
            "rule_quality_score": round(score, 2),
            "valid_rules": valid_rules,
            "invalid_rules": invalid_rules,
            "missing_required_fields": sorted(set(missing_required)),
            "schema_errors": schema_errors,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def evaluate_feature_extraction(
        self,
        extracted_features: dict[str, Any] | list[dict[str, Any]],
        expected_features: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate extracted healthcare features.

        Args:
            extracted_features: Feature dictionary or feature dictionaries.
            expected_features: Optional expected features.

        Returns:
            Feature extraction quality result.
        """
        features_list = extracted_features if isinstance(extracted_features, list) else [extracted_features]
        if not features_list or not all(isinstance(item, dict) for item in features_list):
            return self._feature_eval_empty(["Features must be dictionaries."])

        valid_features = 0
        format_errors: list[str] = []
        missing_values: list[str] = []
        for index, features in enumerate(features_list):
            try:
                FeatureQualitySchema.model_validate(features)
                valid_features += 1
            except ValidationError as error:
                format_errors.extend(f"features[{index}]: {item['msg']}" for item in error.errors())
            format_errors.extend(self._code_format_errors(features))
            missing_values.extend(self._missing_feature_values(features))
        invalid_features = len(features_list) - valid_features + (1 if format_errors else 0)
        invalid_features = min(invalid_features, len(features_list))
        schema_score = max(0.0, 1.0 - (len(format_errors) / max(len(features_list) * 4, 1)))
        completeness_score = max(0.0, 1.0 - (len(missing_values) / max(len(features_list) * 5, 1)))
        precision, recall, f1 = self._matching_metrics(features_list, self._coerce_expected_list(expected_features), "service")
        score = (0.55 * schema_score) + (0.45 * completeness_score)
        if f1 is not None:
            score = (0.7 * score) + (0.3 * f1)
        self.logger.info("Feature evaluation completed.")
        return {
            "feature_quality_score": round(self._clamp(score), 2),
            "valid_features": valid_features,
            "invalid_features": invalid_features,
            "format_errors": format_errors,
            "missing_values": sorted(set(missing_values)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def evaluate_summary(
        self,
        summary: dict[str, Any],
        source_text: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate summary completeness and support.

        Args:
            summary: Structured summary dictionary.
            source_text: Optional source text.

        Returns:
            Summary quality result.
        """
        if not isinstance(summary, dict) or not summary:
            return {
                "summary_quality_score": 0.0,
                "coverage_score": 0.0,
                "completeness_score": 0.0,
                "unsupported_content": [],
                "issues": ["Missing summary."],
            }
        required_fields = [
            "purpose",
            "covered_services",
            "excluded_services",
            "eligibility_criteria",
            "medical_necessity",
            "prior_authorization",
            "coding_requirements",
        ]
        present = [field for field in required_fields if summary.get(field)]
        completeness = len(present) / len(required_fields)
        unsupported = self._unsupported_summary_content(summary, source_text or "")
        coverage_score = completeness
        quality = self._clamp((0.70 * completeness) + (0.30 * (1.0 - min(len(unsupported) / 5, 1))))
        issues = [f"Missing summary field: {field}." for field in required_fields if not summary.get(field)]
        return {
            "summary_quality_score": round(quality, 2),
            "coverage_score": round(coverage_score, 2),
            "completeness_score": round(completeness, 2),
            "unsupported_content": unsupported,
            "issues": issues,
        }

    def evaluate_decision_explanation(
        self,
        explanation: dict[str, Any],
        decision_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate explainability output quality.

        Args:
            explanation: Explanation dictionary.
            decision_result: Optional decision result.

        Returns:
            Explanation quality result.
        """
        if not isinstance(explanation, dict) or not explanation:
            return {
                "explanation_quality_score": 0.0,
                "completeness_score": 0.0,
                "auditability_score": 0.0,
                "issues": ["Missing explanation."],
            }
        required_fields = [
            "final_decision",
            "plain_language_summary",
            "decision_confidence",
            "rule_based_reasoning",
            "ml_reasoning",
            "recommended_next_action",
            "audit_trail",
        ]
        present = [field for field in required_fields if explanation.get(field)]
        completeness = len(present) / len(required_fields)
        auditability = self._auditability_score(explanation)
        issues = [f"Missing explanation field: {field}." for field in required_fields if not explanation.get(field)]
        if decision_result and explanation.get("final_decision") != decision_result.get("claim_decision"):
            issues.append("Explanation final decision does not match decision result.")
        quality = self._clamp((0.6 * completeness) + (0.4 * auditability))
        return {
            "explanation_quality_score": round(quality, 2),
            "completeness_score": round(completeness, 2),
            "auditability_score": round(auditability, 2),
            "issues": issues,
        }

    def evaluate_pipeline(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Evaluate full AI pipeline outputs.

        Args:
            outputs: Pipeline output dictionary.

        Returns:
            Consolidated pipeline evaluation.
        """
        if not isinstance(outputs, dict) or not outputs:
            return {
                "overall_score": 0.0,
                "overall_score_completed": 0.0,
                "evaluation_coverage": 0.0,
                "rag_score": 0.0,
                "rule_score": 0.0,
                "feature_score": 0.0,
                "summary_score": 0.0,
                "explanation_score": 0.0,
                "passed": False,
                "module_status": {},
                "pending_modules": ["rag", "rules", "features", "summary", "explanation"],
                "critical_issues": ["Missing pipeline outputs."],
                "recommendations": ["Provide module outputs before pipeline evaluation."],
            }
        rag_response = outputs.get("rag_response", {})
        rag_available = bool(
            isinstance(rag_response, dict)
            and rag_response.get("answer")
            and rag_response.get("retrieved_documents")
        )
        rules_output = outputs.get("rules", [])
        rules_available = bool(isinstance(rules_output, list) and rules_output)
        features_output = outputs.get("features", [])
        features_available = bool(
            (isinstance(features_output, list) and features_output)
            or (isinstance(features_output, dict) and features_output)
        )
        summary_output = outputs.get("summary", {})
        summary_available = bool(isinstance(summary_output, dict) and summary_output)
        explanation_output = outputs.get("explanation", {})
        decision_output = outputs.get("decision", {})
        explanation_available = bool(isinstance(explanation_output, dict) and explanation_output)

        rag_eval = (
            self.evaluate_rag_response(
                rag_response.get("question", ""),
                rag_response.get("answer", ""),
                rag_response.get("retrieved_documents", []),
                rag_response.get("citations", []),
            )
            if rag_available
            else {"rag_quality_score": 0.0, "issues": [], "recommendation": "Pending"}
        )
        rule_eval = (
            self.evaluate_rule_extraction(rules_output)
            if rules_available
            else {"rule_quality_score": 0.0, "schema_errors": [], "missing_required_fields": []}
        )
        feature_eval = (
            self.evaluate_feature_extraction(features_output)
            if features_available
            else {"feature_quality_score": 0.0, "format_errors": [], "missing_values": []}
        )
        summary_eval = (
            self.evaluate_summary(summary_output)
            if summary_available
            else {"summary_quality_score": 0.0, "issues": []}
        )
        explanation_eval = (
            self.evaluate_decision_explanation(explanation_output, decision_output)
            if explanation_available
            else {"explanation_quality_score": 0.0, "issues": []}
        )
        scores = {
            "rag_score": rag_eval.get("rag_quality_score", 0.0),
            "rule_score": rule_eval.get("rule_quality_score", 0.0),
            "feature_score": feature_eval.get("feature_quality_score", 0.0),
            "summary_score": summary_eval.get("summary_quality_score", 0.0),
            "explanation_score": explanation_eval.get("explanation_quality_score", 0.0),
        }
        module_status = {
            "rag": {
                "status": "completed" if rag_available else "pending",
                "score": round(float(scores["rag_score"]), 2),
                "label": "Policy Chat / RAG",
            },
            "rules": {
                "status": "completed" if rules_available else "pending",
                "score": round(float(scores["rule_score"]), 2),
                "label": "Rule Extraction",
            },
            "features": {
                "status": "completed" if features_available else "pending",
                "score": round(float(scores["feature_score"]), 2),
                "label": "Feature Extraction",
            },
            "summary": {
                "status": "completed" if summary_available else "pending",
                "score": round(float(scores["summary_score"]), 2),
                "label": "Summarization",
            },
            "explanation": {
                "status": "completed" if explanation_available else "pending",
                "score": round(float(scores["explanation_score"]), 2),
                "label": "Explainability",
            },
        }
        completed_scores = [
            status["score"]
            for status in module_status.values()
            if status["status"] == "completed"
        ]
        overall = float(np.mean(completed_scores)) if completed_scores else 0.0
        evaluation_coverage = len(completed_scores) / len(module_status) if module_status else 0.0
        pending_modules = [
            key for key, value in module_status.items() if value["status"] != "completed"
        ]
        critical_issues = self._collect_critical_issues(
            rag_eval,
            rule_eval,
            feature_eval,
            summary_eval,
            explanation_eval,
        )
        recommendations = self._recommendations_from_issues(critical_issues, pending_modules)
        self.logger.info("Pipeline evaluation completed.")
        return {
            "overall_score": round(overall, 2),
            "overall_score_completed": round(overall, 2),
            "evaluation_coverage": round(evaluation_coverage, 2),
            **{key: round(float(value), 2) for key, value in scores.items()},
            "passed": overall >= 0.70 and evaluation_coverage >= 0.80 and not any("High" in issue for issue in critical_issues),
            "module_status": module_status,
            "pending_modules": pending_modules,
            "critical_issues": critical_issues,
            "recommendations": recommendations,
        }

    def generate_evaluation_report(self, evaluations: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """Generate consolidated evaluation report.

        Args:
            evaluations: Evaluation dictionary or list of evaluations.

        Returns:
            Consolidated evaluation report.
        """
        evaluation_list = evaluations if isinstance(evaluations, list) else [evaluations]
        scores = self._extract_scores(evaluation_list)
        overall_score = float(np.mean(scores)) if scores else 0.0
        issues = self._extract_issues(evaluation_list)
        report = {
            "report_id": "EVAL_REPORT_001",
            "timestamp": "deterministic-not-recorded",
            "overall_score": round(overall_score, 2),
            "module_scores": self._module_scores(evaluation_list),
            "strengths": self._strengths_from_scores(evaluation_list),
            "issues": issues,
            "recommendations": self._recommendations_from_issues(issues),
            "pass_fail": "Pass" if overall_score >= 0.70 and not issues else "Review",
        }
        return report

    def export_report(self, report: dict[str, Any], output_path: str | Path) -> bool:
        """Export evaluation report as JSON, Markdown, or CSV.

        Args:
            report: Evaluation report dictionary.
            output_path: Destination path.

        Returns:
            ``True`` when export succeeds; otherwise ``False``.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            suffix = path.suffix.lower()
            if suffix == ".md":
                path.write_text(self._report_to_markdown(report), encoding="utf-8")
            elif suffix == ".csv":
                self._report_to_csv(report, path)
            else:
                path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.logger.info("Report exported: %s", path)
            return True
        except (OSError, TypeError) as error:
            self.logger.error("Report export failed: %s", error)
            return False

    def get_evaluation_statistics(self, evaluations: list[dict[str, Any]]) -> dict[str, int | float]:
        """Calculate statistics across evaluations.

        Args:
            evaluations: Evaluation dictionaries.

        Returns:
            Aggregate evaluation statistics.
        """
        if not isinstance(evaluations, list) or not evaluations:
            return {
                "total_evaluations": 0,
                "average_overall_score": 0.0,
                "average_rag_score": 0.0,
                "average_rule_score": 0.0,
                "average_feature_score": 0.0,
                "average_summary_score": 0.0,
                "average_explanation_score": 0.0,
                "pass_rate": 0.0,
            }
        return {
            "total_evaluations": len(evaluations),
            "average_overall_score": self._average_key(evaluations, "overall_score"),
            "average_rag_score": self._average_key(evaluations, "rag_score", "rag_quality_score"),
            "average_rule_score": self._average_key(evaluations, "rule_score", "rule_quality_score"),
            "average_feature_score": self._average_key(evaluations, "feature_score", "feature_quality_score"),
            "average_summary_score": self._average_key(evaluations, "summary_score", "summary_quality_score"),
            "average_explanation_score": self._average_key(evaluations, "explanation_score", "explanation_quality_score"),
            "pass_rate": round(
                sum(1 for item in evaluations if item.get("passed") is True or item.get("recommendation") == "Pass") / len(evaluations),
                2,
            ),
        }

    def _term_overlap(self, left: str, right: str) -> float:
        """Calculate token overlap ratio.

        Args:
            left: First text.
            right: Second text.

        Returns:
            Overlap ratio.
        """
        left_terms = self._content_terms(left)
        right_terms = self._content_terms(right)
        if not left_terms:
            return 0.0
        return len(left_terms & right_terms) / len(left_terms)

    def _content_terms(self, text: str) -> set[str]:
        """Extract meaningful lowercase terms.

        Args:
            text: Source text.

        Returns:
            Set of content terms.
        """
        stopwords = {
            "the",
            "is",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "for",
            "in",
            "after",
            "with",
            "when",
            "are",
            "be",
        }
        return {
            token.lower()
            for token in re.findall(r"\b[A-Za-z0-9.$%]+\b", text)
            if token.lower() not in stopwords and len(token) > 1
        }

    def _combined_retrieved_text(self, retrieved_documents: list[dict[str, Any]]) -> str:
        """Combine retrieved document text.

        Args:
            retrieved_documents: Retrieved documents.

        Returns:
            Combined retrieved text.
        """
        if not isinstance(retrieved_documents, list):
            return ""
        return " ".join(str(document.get("text", "")) for document in retrieved_documents if isinstance(document, dict))

    def _retrieved_text_for_source(self, retrieved_documents: list[dict[str, Any]], source: str) -> str:
        """Get retrieved text for a cited source.

        Args:
            retrieved_documents: Retrieved documents.
            source: Source document name.

        Returns:
            Matching retrieved text.
        """
        return " ".join(
            str(document.get("text", ""))
            for document in retrieved_documents
            if document.get("metadata", {}).get("source", "") == source
        )

    def _sentences(self, text: str) -> list[str]:
        """Split text into simple sentences.

        Args:
            text: Source text.

        Returns:
            Sentence list.
        """
        return [sentence.strip() for sentence in re.split(r"[.!?]+", text) if sentence.strip()]

    def _all_critical_tokens_supported(self, answer: str, context: str) -> bool:
        """Check whether specific codes, amounts, and numbers are in context.

        Args:
            answer: Answer text.
            context: Retrieved context.

        Returns:
            Whether all critical tokens are supported.
        """
        critical = re.findall(self._code_pattern(), answer, flags=re.IGNORECASE)
        critical.extend(re.findall(r"\$\s?\d[\d,]*(?:\.\d{2})?", answer))
        critical.extend(re.findall(r"\b\d+\s+(?:week|weeks|month|months|day|days)\b", answer, flags=re.IGNORECASE))
        return all(token.lower() in context.lower() for token in critical)

    def _unsupported_pattern_values(self, answer: str, context: str, pattern: str) -> list[str]:
        """Find pattern values in answer missing from context.

        Args:
            answer: Answer text.
            context: Retrieved context.
            pattern: Regex pattern.

        Returns:
            Unsupported values.
        """
        values = re.findall(pattern, answer, flags=re.IGNORECASE)
        return sorted({value for value in values if str(value).lower() not in context.lower()})

    def _code_pattern(self) -> str:
        """Return code regex pattern.

        Args:
            None.

        Returns:
            Regex pattern for ICD/CPT/HCPCS-like codes.
        """
        return r"\b(?:[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?|\d{5}|[A-Z]\d{4})\b"

    def _answer_completeness(self, answer: str, question: str, expected_answer: str | None) -> float:
        """Estimate answer completeness.

        Args:
            answer: Generated answer.
            question: User question.
            expected_answer: Optional reference answer.

        Returns:
            Completeness score.
        """
        if expected_answer:
            return self._term_overlap(expected_answer, answer)
        question_terms = self._content_terms(question)
        answer_terms = self._content_terms(answer)
        if not question_terms:
            return min(1.0, len(answer_terms) / 8)
        return self._clamp((len(question_terms & answer_terms) / len(question_terms)) + min(0.5, len(answer_terms) / 20))

    def _missing_rule_fields(self, rule: Any) -> list[str]:
        """Find missing required rule fields.

        Args:
            rule: Candidate rule.

        Returns:
            Missing field names.
        """
        if not isinstance(rule, dict):
            return ["rule"]
        return [field for field in ("rule_id", "rule_type", "conditions", "decision") if not rule.get(field)]

    def _rule_eval_empty(self, errors: list[str]) -> dict[str, Any]:
        """Build empty rule evaluation result.

        Args:
            errors: Error messages.

        Returns:
            Rule evaluation result.
        """
        return {
            "rule_quality_score": 0.0,
            "valid_rules": 0,
            "invalid_rules": 0,
            "missing_required_fields": [],
            "schema_errors": errors,
            "precision": None,
            "recall": None,
            "f1_score": None,
        }

    def _feature_eval_empty(self, errors: list[str]) -> dict[str, Any]:
        """Build empty feature evaluation result.

        Args:
            errors: Error messages.

        Returns:
            Feature evaluation result.
        """
        return {
            "feature_quality_score": 0.0,
            "valid_features": 0,
            "invalid_features": 0,
            "format_errors": errors,
            "missing_values": [],
            "precision": None,
            "recall": None,
            "f1_score": None,
        }

    def _code_format_errors(self, features: dict[str, Any]) -> list[str]:
        """Check code format validity.

        Args:
            features: Feature dictionary.

        Returns:
            Format error messages.
        """
        errors: list[str] = []
        for code in features.get("icd_codes", []):
            if not re.fullmatch(r"[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?", str(code), flags=re.IGNORECASE):
                errors.append(f"Invalid ICD code: {code}.")
        for code in features.get("cpt_codes", []):
            if not re.fullmatch(r"\d{5}", str(code)):
                errors.append(f"Invalid CPT code: {code}.")
        for code in features.get("hcpcs_codes", []):
            if not re.fullmatch(r"[A-Z]\d{4}", str(code), flags=re.IGNORECASE):
                errors.append(f"Invalid HCPCS code: {code}.")
        return errors

    def _missing_feature_values(self, features: dict[str, Any]) -> list[str]:
        """Find important missing feature values.

        Args:
            features: Feature dictionary.

        Returns:
            Missing value names.
        """
        missing: list[str] = []
        for field in ("source_text", "confidence"):
            if features.get(field) in (None, "", []):
                missing.append(field)
        return missing

    def _coerce_expected_list(self, expected: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Coerce expected data to list.

        Args:
            expected: Expected item or list.

        Returns:
            Expected list or ``None``.
        """
        if expected is None:
            return None
        return expected if isinstance(expected, list) else [expected]

    def _matching_metrics(
        self,
        extracted: list[dict[str, Any]],
        expected: list[dict[str, Any]] | None,
        key: str,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute simple precision, recall, and F1.

        Args:
            extracted: Extracted records.
            expected: Expected records.
            key: Key used for matching.

        Returns:
            Precision, recall, and F1, or ``None`` values when expected absent.
        """
        if not expected:
            return None, None, None
        extracted_values = {str(item.get(key, "")) for item in extracted if isinstance(item, dict)}
        expected_values = {str(item.get(key, "")) for item in expected if isinstance(item, dict)}
        true_positive = len(extracted_values & expected_values)
        precision = true_positive / len(extracted_values) if extracted_values else 0.0
        recall = true_positive / len(expected_values) if expected_values else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        return round(precision, 2), round(recall, 2), round(f1, 2)

    def _unsupported_summary_content(self, summary: dict[str, Any], source_text: str) -> list[str]:
        """Find summary values unsupported by source text.

        Args:
            summary: Summary dictionary.
            source_text: Source text.

        Returns:
            Unsupported summary snippets.
        """
        if not source_text:
            return []
        unsupported: list[str] = []
        for value in summary.values():
            values = value if isinstance(value, list) else [value]
            for item in values:
                text = str(item)
                if text and self._term_overlap(text, source_text) < 0.25:
                    unsupported.append(text)
        return unsupported[:10]

    def _auditability_score(self, explanation: dict[str, Any]) -> float:
        """Calculate explanation auditability score.

        Args:
            explanation: Explanation dictionary.

        Returns:
            Auditability score.
        """
        score = 0.0
        if explanation.get("rule_based_reasoning"):
            score += 0.25
        if explanation.get("ml_reasoning"):
            score += 0.20
        if explanation.get("audit_trail"):
            score += 0.25
        if explanation.get("supporting_citations"):
            score += 0.20
        if explanation.get("recommended_next_action"):
            score += 0.10
        return self._clamp(score)

    def _collect_critical_issues(self, *evaluations: dict[str, Any]) -> list[str]:
        """Collect issues from module evaluations.

        Args:
            *evaluations: Evaluation dictionaries.

        Returns:
            Critical issue list.
        """
        issues: list[str] = []
        for evaluation in evaluations:
            for key in ("issues", "schema_errors", "format_errors", "risk_reasons"):
                values = evaluation.get(key, [])
                if isinstance(values, list):
                    issues.extend(str(value) for value in values)
            if evaluation.get("hallucination_risk") == "High":
                issues.append("High hallucination risk detected.")
        return issues

    def _recommendations_from_issues(
        self,
        issues: list[str],
        pending_modules: list[str] | None = None,
    ) -> list[str]:
        """Generate recommendations from issues.

        Args:
            issues: Issue descriptions.
            pending_modules: Modules that have not yet been evaluated.

        Returns:
            Recommendation descriptions.
        """
        pending_modules = pending_modules or []
        if not issues and not pending_modules:
            return ["No major quality issues detected."]
        recommendations: list[str] = []
        joined = " ".join(issues).lower()
        pending_text = " ".join(pending_modules).lower()
        if "citation" in joined:
            recommendations.append("Improve citation coverage and source-page mapping.")
        if "unsupported" in joined or "hallucination" in joined:
            recommendations.append("Strengthen grounding checks and retrieved context relevance.")
        if "schema" in joined or "field" in joined:
            recommendations.append("Validate structured outputs before downstream use.")
        if "rag" in pending_text:
            recommendations.append("Run Policy Chat to evaluate grounding, citations, and hallucination risk.")
        if "explanation" in pending_text:
            recommendations.append("Generate Explainability output to complete auditability review.")
        if not recommendations:
            recommendations.append("Review flagged outputs before production use.")
        return recommendations

    def _extract_scores(self, evaluations: list[dict[str, Any]]) -> list[float]:
        """Extract score values from evaluations.

        Args:
            evaluations: Evaluation dictionaries.

        Returns:
            Score list.
        """
        score_keys = [
            "overall_score",
            "rag_quality_score",
            "rule_quality_score",
            "feature_quality_score",
            "summary_quality_score",
            "explanation_quality_score",
        ]
        scores: list[float] = []
        for evaluation in evaluations:
            if isinstance(evaluation, dict):
                scores.extend(float(evaluation[key]) for key in score_keys if isinstance(evaluation.get(key), (int, float)))
        return scores

    def _extract_issues(self, evaluations: list[dict[str, Any]]) -> list[str]:
        """Extract issue strings from evaluations.

        Args:
            evaluations: Evaluation dictionaries.

        Returns:
            Issue list.
        """
        issues: list[str] = []
        for evaluation in evaluations:
            if isinstance(evaluation, dict):
                issues.extend(self._collect_critical_issues(evaluation))
        return issues

    def _module_scores(self, evaluations: list[dict[str, Any]]) -> dict[str, float]:
        """Extract module scores for report.

        Args:
            evaluations: Evaluation dictionaries.

        Returns:
            Module score mapping.
        """
        module_scores: dict[str, float] = {}
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            for key, value in evaluation.items():
                if key.endswith("_score") and isinstance(value, (int, float)):
                    module_scores[key] = float(value)
        return module_scores

    def _strengths_from_scores(self, evaluations: list[dict[str, Any]]) -> list[str]:
        """Generate strengths from high scores.

        Args:
            evaluations: Evaluation dictionaries.

        Returns:
            Strength descriptions.
        """
        strengths: list[str] = []
        for key, value in self._module_scores(evaluations).items():
            if value >= 0.85:
                strengths.append(f"{key} is strong.")
        return strengths

    def _report_to_markdown(self, report: dict[str, Any]) -> str:
        """Convert report to Markdown.

        Args:
            report: Evaluation report.

        Returns:
            Markdown report.
        """
        lines = [
            "# Healthcare AI Evaluation Report",
            "",
            f"Overall score: {report.get('overall_score', 0.0)}",
            f"Pass/fail: {report.get('pass_fail', '')}",
            "",
            "## Issues",
        ]
        issues = report.get("issues", [])
        lines.extend(f"- {issue}" for issue in issues) if issues else lines.append("- None")
        lines.append("")
        lines.append("## Recommendations")
        recommendations = report.get("recommendations", [])
        lines.extend(f"- {item}" for item in recommendations) if recommendations else lines.append("- None")
        return "\n".join(lines).strip() + "\n"

    def _report_to_csv(self, report: dict[str, Any], path: Path) -> None:
        """Write report summary to CSV.

        Args:
            report: Evaluation report.
            path: Destination CSV path.

        Returns:
            None.
        """
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["field", "value"])
            for key in ("report_id", "timestamp", "overall_score", "pass_fail"):
                writer.writerow([key, report.get(key, "")])

    def _average_key(self, evaluations: list[dict[str, Any]], *keys: str) -> float:
        """Average the first available key across evaluations.

        Args:
            evaluations: Evaluation dictionaries.
            *keys: Candidate score keys.

        Returns:
            Average score.
        """
        values: list[float] = []
        for evaluation in evaluations:
            for key in keys:
                if isinstance(evaluation.get(key), (int, float)):
                    values.append(float(evaluation[key]))
                    break
        return round(float(np.mean(values)), 2) if values else 0.0

    def _clamp(self, value: float) -> float:
        """Clamp value to [0, 1].

        Args:
            value: Numeric value.

        Returns:
            Clamped float.
        """
        return float(np.clip(value, 0.0, 1.0))

"""Deterministic healthcare business rule execution.

This module evaluates structured JSON healthcare rules against extracted
features or sample claim records. It does not perform ML prediction, final
claim decisioning, Streamlit rendering, or LLM calls.

Example:
    ```python
    from modules.rule_engine import HealthcareRuleEngine

    engine = HealthcareRuleEngine()

    rules = [
        {
            "rule_id": "RULE_001",
            "rule_type": "coverage",
            "service": "Lumbar spine MRI",
            "condition_logic": "AND",
            "conditions": [
                {"field": "therapy_weeks", "operator": ">=", "value": 6},
                {
                    "field": "prior_authorization_required",
                    "operator": "==",
                    "value": True
                }
            ],
            "decision": "approve",
            "action": "Approve claim",
            "confidence": 0.92
        }
    ]

    features = {
        "patient_age": 55,
        "icd_codes": ["M54.16"],
        "cpt_codes": ["72148"],
        "diagnosis": "Lumbar radiculopathy",
        "procedure": "Lumbar spine MRI",
        "therapy_weeks": 6,
        "prior_authorization_required": True
    }

    results = engine.evaluate_rules(rules, features)

    print(results)
    ```
"""

from __future__ import annotations

import json
import logging
import operator
import re
from pathlib import Path
from typing import Any, Callable


class HealthcareRuleEngine:
    """Executes structured healthcare business rules against feature records."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the healthcare rule engine.

        Args:
            logger: Optional logger instance.

        Returns:
            None.
        """
        self.logger = logger or logging.getLogger(__name__)
        self._comparison_operators: dict[str, Callable[[Any, Any], bool]] = {
            "==": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
        }
        self._field_aliases: dict[str, tuple[str, ...]] = {
            "cpt_code": ("cpt_codes",),
            "cpt_codes": ("cpt_codes",),
            "cpt": ("cpt_codes",),
            "icd_code": ("icd_codes",),
            "icd_codes": ("icd_codes",),
            "icd_10_diagnosis_code": ("icd_codes",),
            "diagnosis_code": ("icd_codes",),
            "hcpcs_code": ("hcpcs_codes",),
            "hcpcs_codes": ("hcpcs_codes",),
            "prior_authorization": ("prior_authorization_required",),
            "prior_authorization_required": ("prior_authorization_required",),
            "therapy_duration": ("therapy_weeks",),
            "conservative_therapy_duration": ("therapy_weeks",),
            "therapy_weeks": ("therapy_weeks",),
            "service_type": ("source_text", "service", "procedure"),
            "service": ("service", "procedure", "source_text"),
            "procedure": ("procedure", "service"),
            "diagnosis": ("diagnosis", "icd_codes"),
            "clinical_indication": ("diagnosis", "source_text"),
            "symptom": ("diagnosis", "source_text"),
            "medical_necessity": ("medical_necessity_criteria", "source_text"),
            "documentation": ("documentation_required", "source_text"),
            "provider_information": ("documentation_required", "source_text"),
            "ordering_provider_attestation": ("documentation_required", "source_text"),
            "provider_type": ("source_text",),
            "time_period": ("frequency_limit", "source_text"),
            "frequency_limit": ("frequency_limit", "source_text"),
            "repeat_imaging": ("source_text",),
            "symptoms": ("source_text", "diagnosis"),
            "regulation": ("source_text",),
            "history_and_physical_examination": ("documentation_required", "source_text"),
            "clinical_indication_and_suspected_diagnosis": ("documentation_required", "diagnosis", "source_text"),
            "duration_and_type_of_conservative_therapy": ("documentation_required", "therapy_weeks", "source_text"),
            "neurological_examination_findings": ("documentation_required", "source_text"),
            "prior_imaging_results": ("documentation_required", "source_text"),
            "emergency_department_imaging": ("source_text",),
            "inpatient_services": ("source_text",),
            "progressive_neurological_deficit": ("source_text", "diagnosis"),
            "expected_change_in_clinical_management": ("source_text",),
        }

    def load_rules(self, input_path: str | Path) -> list[dict[str, Any]]:
        """Load JSON rules saved by the rule extractor.

        Args:
            input_path: Path to a JSON file containing rule dictionaries.

        Returns:
            Loaded rules, or an empty list if loading fails.
        """
        path = Path(input_path)
        if not path.exists():
            self.logger.error("Rules file does not exist: %s", path)
            return []

        try:
            rules = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rules, list):
                self.logger.error("Rules JSON must contain a list: %s", path)
                return []
            self.logger.info("Rules loaded: %s", len(rules))
            return [rule for rule in rules if isinstance(rule, dict)]
        except (json.JSONDecodeError, OSError) as error:
            self.logger.error("Failed to load rules from %s: %s", path, error)
            return []

    def save_results(self, results: list[dict[str, Any]], output_path: str | Path) -> bool:
        """Save rule evaluation results as JSON.

        Args:
            results: Rule evaluation results.
            output_path: Destination JSON file path.

        Returns:
            ``True`` when saved successfully; otherwise ``False``.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            self.logger.info("Rule evaluation results saved: %s", path)
            return True
        except (OSError, TypeError) as error:
            self.logger.error("Failed to save rule results to %s: %s", path, error)
            return False

    def evaluate_rule(
        self,
        rule: dict[str, Any],
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate one JSON rule against one feature dictionary.

        Args:
            rule: Structured business rule.
            features: Extracted healthcare features or claim attributes.

        Returns:
            Structured rule evaluation result with per-condition details.
        """
        self.logger.info("Rule evaluation started.")
        rule_validation = self.validate_rule(rule)
        feature_validation = self.validate_features(features)

        if not rule_validation["valid"]:
            return self._error_result(
                rule if isinstance(rule, dict) else {},
                "Invalid rule schema.",
                rule_validation["errors"],
            )

        if not feature_validation["valid"]:
            return self._error_result(
                rule,
                "Invalid feature input.",
                feature_validation["errors"],
            )

        conditions = rule.get("conditions", [])
        condition_results = [
            self._evaluate_condition(condition, features)
            for condition in conditions
            if isinstance(condition, dict)
        ]
        condition_logic = str(rule.get("condition_logic", "AND")).upper()
        matched = self._combine_condition_results(condition_results, condition_logic)
        reason = self._build_reason(matched, condition_results, condition_logic)

        result = {
            "rule_id": rule.get("rule_id", ""),
            "rule_type": rule.get("rule_type", ""),
            "matched": matched,
            "decision": rule.get("decision", "") if matched else "no_match",
            "action": rule.get("action", "") if matched else "",
            "reason": reason,
            "conditions_evaluated": condition_results,
            "confidence": rule.get("confidence", 0.0),
            "service": rule.get("service", ""),
            "errors": [],
        }
        self.logger.info("Rule evaluation completed: %s matched=%s", result["rule_id"], matched)
        return result

    def evaluate_rules(
        self,
        rules: list[dict[str, Any]],
        features: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Evaluate multiple rules against one feature dictionary.

        Args:
            rules: Rule dictionaries.
            features: Extracted healthcare features or claim attributes.

        Returns:
            List of rule evaluation results.
        """
        if not isinstance(rules, list) or not rules:
            self.logger.warning("No rules provided for evaluation.")
            return []

        results = [self.evaluate_rule(rule, features) for rule in rules]
        matched_count = sum(1 for result in results if result.get("matched") is True)
        self.logger.info("Number of matched rules: %s", matched_count)
        return results

    def evaluate_batch(
        self,
        rules: list[dict[str, Any]],
        feature_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate multiple rules against multiple feature records.

        Args:
            rules: Rule dictionaries.
            feature_records: Feature dictionaries or claim records.

        Returns:
            Batch evaluation results grouped by feature record index.
        """
        if not isinstance(feature_records, list) or not feature_records:
            self.logger.warning("No feature records provided for batch evaluation.")
            return []

        batch_results: list[dict[str, Any]] = []
        for index, features in enumerate(feature_records):
            record_results = self.evaluate_rules(rules, features)
            batch_results.append(
                {
                    "record_index": index,
                    "record_id": features.get("claim_id", features.get("record_id", "")) if isinstance(features, dict) else "",
                    "results": record_results,
                }
            )
        return batch_results

    def validate_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """Validate the required structure of one rule.

        Args:
            rule: Candidate rule dictionary.

        Returns:
            Validation result with ``valid`` and ``errors`` fields.
        """
        errors: list[str] = []
        if not isinstance(rule, dict):
            return {"valid": False, "errors": ["Rule must be a dictionary."]}

        for field in ("rule_id", "rule_type", "conditions", "decision"):
            if field not in rule:
                errors.append(f"Missing required field: {field}.")

        if "conditions" in rule and not isinstance(rule["conditions"], list):
            errors.append("conditions must be a list.")

        condition_logic = str(rule.get("condition_logic", "AND")).upper()
        if condition_logic not in {"AND", "OR"}:
            errors.append("condition_logic must be AND or OR.")

        if errors:
            self.logger.warning("Invalid rule: %s", errors)
        return {"valid": not errors, "errors": errors}

    def validate_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Validate that features are a usable dictionary.

        Args:
            features: Candidate feature dictionary.

        Returns:
            Validation result with ``valid`` and ``errors`` fields.
        """
        if not isinstance(features, dict):
            return {"valid": False, "errors": ["Features must be a dictionary."]}
        if not features:
            return {"valid": False, "errors": ["Features dictionary cannot be empty."]}
        return {"valid": True, "errors": []}

    def get_rule_engine_statistics(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Calculate rule engine evaluation statistics.

        Args:
            results: Rule evaluation results.

        Returns:
            Summary statistics for evaluated, matched, unmatched, and decision
            counts.
        """
        flattened_results = self._flatten_results(results)
        matched_results = [
            result for result in flattened_results if result.get("matched") is True
        ]
        unmatched_results = [
            result for result in flattened_results if result.get("matched") is not True
        ]

        return {
            "total_rules_evaluated": len(flattened_results),
            "matched_rules": len(matched_results),
            "unmatched_rules": len(unmatched_results),
            "approve_decisions": self._count_decisions(matched_results, {"approve", "approved"}),
            "deny_decisions": self._count_decisions(matched_results, {"deny", "denied"}),
            "manual_review_decisions": self._count_decisions(
                matched_results,
                {"review", "manual_review", "flag_for_review"},
            ),
        }

    def _evaluate_condition(
        self,
        condition: dict[str, Any],
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate one condition against features.

        Args:
            condition: Rule condition dictionary.
            features: Feature dictionary.

        Returns:
            Structured condition evaluation result.
        """
        field = str(condition.get("field", ""))
        condition_operator = str(condition.get("operator", "=="))
        expected_value = condition.get("value")
        resolved_field, actual_value = self._resolve_feature_value(field, features)

        if not field:
            return self._condition_result(condition, None, False, "Condition field is missing.")

        if resolved_field is None:
            if condition_operator == "not_exists":
                return self._condition_result(condition, None, True, "Field does not exist.")
            return self._condition_result(
                condition,
                None,
                False,
                f"Field '{field}' is missing from features.",
            )

        try:
            matched = self._apply_operator(actual_value, condition_operator, expected_value)
            reason = "Condition satisfied." if matched else "Condition not satisfied."
            result = self._condition_result(condition, actual_value, matched, reason)
            result["resolved_field"] = resolved_field
            return result
        except ValueError as error:
            if "Unsupported operator" in str(error):
                self.logger.warning("Unsupported operator: %s", condition_operator)
            result = self._condition_result(condition, actual_value, False, str(error))
            result["resolved_field"] = resolved_field or field
            return result
        except (TypeError, ValueError) as error:
            result = self._condition_result(condition, actual_value, False, f"Type mismatch: {error}")
            result["resolved_field"] = resolved_field or field
            return result

    def _apply_operator(self, actual_value: Any, condition_operator: str, expected_value: Any) -> bool:
        """Apply a supported rule operator.

        Args:
            actual_value: Feature value.
            condition_operator: Rule operator.
            expected_value: Expected value.

        Returns:
            Whether the condition matches.

        Raises:
            ValueError: If the operator is unsupported.
        """
        normalized_operator = self._normalize_operator(condition_operator)
        if normalized_operator in self._comparison_operators:
            return self._compare_values(actual_value, expected_value, normalized_operator)
        if normalized_operator == "in":
            return self._contains(expected_value, actual_value)
        if normalized_operator == "not_in":
            return not self._contains(expected_value, actual_value)
        if normalized_operator == "contains":
            return self._contains(actual_value, expected_value)
        if normalized_operator == "not_contains":
            return not self._contains(actual_value, expected_value)
        if normalized_operator == "exists":
            return actual_value is not None
        if normalized_operator == "not_exists":
            return actual_value is None
        raise ValueError(f"Unsupported operator: {condition_operator}.")

    def _normalize_operator(self, condition_operator: str) -> str:
        """Normalize human-readable operators into executable operators."""
        normalized = " ".join(str(condition_operator).strip().lower().replace("_", " ").split())
        operator_map = {
            "equals": "==",
            "equal to": "==",
            "is": "==",
            "not equals": "!=",
            "not equal to": "!=",
            "in": "in",
            "not in": "not_in",
            "contains": "contains",
            "not contains": "not_contains",
            "exists": "exists",
            "presence": "exists",
            "absence": "not_exists",
            "required": "exists",
            "greater than or equal to": ">=",
            "less than or equal to": "<=",
            "greater than": ">",
            "less than": "<",
            "within": "contains",
            "same": "contains",
            "not changed": "not_contains",
        }
        return operator_map.get(normalized, normalized)

    def _compare_values(self, actual_value: Any, expected_value: Any, condition_operator: str) -> bool:
        """Compare actual and expected values.

        Args:
            actual_value: Feature value.
            expected_value: Expected value.
            condition_operator: Comparison operator.

        Returns:
            Comparison result.
        """
        comparator = self._comparison_operators[condition_operator]
        if isinstance(actual_value, list):
            expected_items = self._to_comparable_list(expected_value)
            actual_items = self._to_comparable_list(actual_value)
            if not expected_items:
                return False
            if condition_operator == "==":
                return any(item in actual_items for item in expected_items)
            if condition_operator == "!=":
                return all(item not in actual_items for item in expected_items)
        if isinstance(actual_value, bool):
            coerced_expected = self._coerce_bool_like(expected_value)
            if coerced_expected is not None:
                return comparator(actual_value, coerced_expected)
        if isinstance(actual_value, str):
            coerced_expected = self._coerce_string_expectation(expected_value)
            if coerced_expected is not None:
                if condition_operator == "==":
                    return coerced_expected in actual_value.lower()
                if condition_operator == "!=":
                    return coerced_expected not in actual_value.lower()
        if condition_operator in {">", ">=", "<", "<="}:
            actual_number = self._coerce_number_like(actual_value)
            expected_number = self._coerce_number_like(expected_value)
            if actual_number is None or expected_number is None:
                raise TypeError("Numeric comparison values could not be parsed.")
            return comparator(actual_number, expected_number)
        return comparator(actual_value, expected_value)

    def _contains(self, container: Any, item: Any) -> bool:
        """Evaluate containment across strings and collections.

        Args:
            container: Container value.
            item: Item value.

        Returns:
            Whether ``container`` contains ``item``.
        """
        if container is None:
            return False
        if isinstance(container, str):
            items = self._to_comparable_list(item)
            container_text = container.lower()
            return any(candidate in container_text for candidate in items)
        if isinstance(container, (list, tuple, set)):
            container_items = self._to_comparable_list(container)
            item_values = self._to_comparable_list(item)
            return any(candidate in container_items for candidate in item_values)
        return container == item

    def _resolve_feature_value(
        self,
        field: str,
        features: dict[str, Any],
    ) -> tuple[str | None, Any]:
        """Resolve a rule condition field name to the best feature value."""
        normalized_field = self._normalize_field_name(field)
        if normalized_field == "diagnosis" and self._looks_like_code(condition_value := features.get("diagnosis", "")):
            pass
        candidate_keys = [normalized_field, *self._field_aliases.get(normalized_field, ())]
        if normalized_field == "diagnosis":
            candidate_keys = self._prioritize_diagnosis_keys(candidate_keys, features)
        for key in candidate_keys:
            if key in features and features.get(key) not in (None, "", [], {}):
                return key, features.get(key)

        source_text = str(features.get("source_text", "")).lower()
        documentation = self._to_comparable_list(features.get("documentation_required", []))

        if normalized_field == "service_type" and source_text:
            if "outpatient" in source_text:
                return "source_text", "outpatient"
            if "inpatient" in source_text:
                return "source_text", "inpatient"
        if normalized_field == "provider_type" and source_text:
            if "freestanding imaging provider" in source_text:
                return "source_text", "freestanding imaging provider"
        if normalized_field == "medical_necessity" and source_text:
            if "medically necessary" in source_text or "medical necessity" in source_text:
                return "source_text", "medically necessary"
        if normalized_field == "progressive_neurological_deficit" and source_text:
            if "progressive neurological deficit" in source_text:
                return "source_text", "progressive neurological deficit"
        if normalized_field == "expected_change_in_clinical_management" and source_text:
            if "expected to change clinical management" in source_text or "expected to affect treatment planning" in source_text:
                return "source_text", "expected change in clinical management"
        if normalized_field == "emergency_department_imaging" and source_text:
            if "emergency department imaging" in source_text:
                return "source_text", "emergency department imaging"
        if normalized_field == "inpatient_services" and source_text:
            if "inpatient services" in source_text:
                return "source_text", "inpatient services"
        if normalized_field in {"time_period", "frequency_limit"} and source_text:
            if "12-month" in source_text:
                return "source_text", "12 months"
            if "24-month" in source_text:
                return "source_text", "24 months"
        if normalized_field in {"clinical_indication", "clinical_indication_and_suspected_diagnosis"} and source_text:
            if "same clinical indication" in source_text:
                return "source_text", "same clinical indication"
        if normalized_field in {"medical_necessity"} and source_text:
            if "medically necessary" in source_text or "medical necessity" in source_text:
                return "source_text", "medical necessity present"
        if normalized_field in {"provider_information", "ordering_provider_attestation"} and documentation:
            return "documentation_required", documentation
        if normalized_field in {
            "history_and_physical_examination",
            "clinical_indication_and_suspected_diagnosis",
            "duration_and_type_of_conservative_therapy",
            "neurological_examination_findings",
            "prior_imaging_results",
        } and documentation:
            return "documentation_required", documentation
        if normalized_field in {"time_period", "frequency_limit"} and source_text:
            return "source_text", source_text
        if normalized_field in {"symptom", "symptoms", "clinical_indication", "regulation"} and source_text:
            return "source_text", source_text
        if "_" in normalized_field and source_text:
            return "source_text", source_text
        return None, None

    def _prioritize_diagnosis_keys(
        self,
        candidate_keys: list[str],
        features: dict[str, Any],
    ) -> list[str]:
        """Prefer code-based diagnosis matching when ICD codes are available."""
        if features.get("icd_codes"):
            prioritized = ["icd_codes"]
            prioritized.extend(key for key in candidate_keys if key != "icd_codes")
            return prioritized
        return candidate_keys

    def _normalize_field_name(self, field: str) -> str:
        """Normalize human-readable condition fields into lookup keys."""
        normalized = field.strip().lower().replace("-", "_").replace("/", "_")
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("__", "_")
        alias_map = {
            "cpt_code": "cpt_code",
            "cpt_codes": "cpt_codes",
            "icd_10_diagnosis_code": "icd_10_diagnosis_code",
            "supported_icd_10_diagnosis_code": "icd_10_diagnosis_code",
            "hcpcs_code": "hcpcs_code",
            "prior_authorization": "prior_authorization",
            "medical_necessity": "medical_necessity",
            "conservative_therapy_duration": "conservative_therapy_duration",
            "therapy_duration": "therapy_duration",
            "service_type": "service_type",
            "provider_information": "provider_information",
            "ordering_provider_attestation": "ordering_provider_attestation",
            "time_period": "time_period",
            "clinical_indication": "clinical_indication",
        }
        return alias_map.get(normalized, normalized)

    def _to_comparable_list(self, value: Any) -> list[str]:
        """Convert strings, lists, and scalars into normalized comparable items."""
        if isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        elif isinstance(value, str):
            raw_items = [item for item in value.replace(";", ",").split(",") if item.strip()]
        elif value in (None, ""):
            raw_items = []
        else:
            raw_items = [value]
        return [" ".join(str(item).strip().lower().split()) for item in raw_items if str(item).strip()]

    def _coerce_bool_like(self, value: Any) -> bool | None:
        """Coerce common text expectations into booleans."""
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "yes", "present", "required"}:
            return True
        if text in {"false", "no", "absent", "not required"}:
            return False
        return None

    def _coerce_string_expectation(self, value: Any) -> str | None:
        """Normalize text expectations for string containment checks."""
        if value in (None, ""):
            return None
        normalized = " ".join(str(value).strip().lower().split())
        prefixes = [
            "is ",
            "in ",
            "within ",
            "presence ",
            "equals ",
            "greater than or equal to ",
            "less than ",
            "same ",
        ]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break
        return normalized

    def _coerce_number_like(self, value: Any) -> float | None:
        """Extract a numeric value from ints, floats, or text like 'six weeks'."""
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        if not text:
            return None
        digit_match = "".join(character for character in text if character.isdigit() or character == ".")
        if digit_match:
            try:
                return float(digit_match)
            except ValueError:
                pass
        word_numbers = {
            "one": 1.0,
            "two": 2.0,
            "three": 3.0,
            "four": 4.0,
            "five": 5.0,
            "six": 6.0,
            "seven": 7.0,
            "eight": 8.0,
            "nine": 9.0,
            "ten": 10.0,
            "twelve": 12.0,
        }
        for word, number in word_numbers.items():
            if re.search(rf"\b{word}\b", text):
                return number
        return None

    def _looks_like_code(self, value: Any) -> bool:
        """Return whether a value looks like a medical code."""
        text = str(value).strip().upper()
        return bool(text and any(character.isdigit() for character in text))

    def _combine_condition_results(
        self,
        condition_results: list[dict[str, Any]],
        condition_logic: str,
    ) -> bool:
        """Combine condition results using AND or OR logic.

        Args:
            condition_results: Per-condition evaluation records.
            condition_logic: Rule-level logical connector.

        Returns:
            Overall rule match status.
        """
        if not condition_results:
            return False
        if condition_logic == "OR":
            return any(result.get("matched") is True for result in condition_results)
        return all(result.get("matched") is True for result in condition_results)

    def _condition_result(
        self,
        condition: dict[str, Any],
        actual_value: Any,
        matched: bool,
        reason: str,
    ) -> dict[str, Any]:
        """Build a condition evaluation result.

        Args:
            condition: Rule condition.
            actual_value: Feature value found during evaluation.
            matched: Whether the condition matched.
            reason: Human-readable condition reason.

        Returns:
            Structured condition evaluation result.
        """
        return {
            "field": condition.get("field", ""),
            "operator": condition.get("operator", ""),
            "expected": self._format_expected(condition),
            "actual": actual_value,
            "matched": matched,
            "reason": reason,
            "description": condition.get("description", ""),
        }

    def _format_expected(self, condition: dict[str, Any]) -> str:
        """Format a condition expectation for explainability.

        Args:
            condition: Rule condition.

        Returns:
            Expected value string.
        """
        condition_operator = condition.get("operator", "")
        value = condition.get("value", "")
        unit = condition.get("unit", "")
        suffix = f" {unit}" if unit else ""
        return f"{condition_operator} {value}{suffix}".strip()

    def _build_reason(
        self,
        matched: bool,
        condition_results: list[dict[str, Any]],
        condition_logic: str,
    ) -> str:
        """Build a rule-level reason.

        Args:
            matched: Overall rule match status.
            condition_results: Per-condition results.
            condition_logic: Rule-level logical connector.

        Returns:
            Human-readable rule reason.
        """
        if matched and condition_logic == "AND":
            return "All required conditions were satisfied."
        if matched and condition_logic == "OR":
            return "At least one required condition was satisfied."
        failed = [
            result.get("field", "")
            for result in condition_results
            if result.get("matched") is not True
        ]
        if failed:
            return f"Rule did not match because these conditions failed: {', '.join(failed)}."
        return "Rule did not match."

    def _error_result(
        self,
        rule: dict[str, Any],
        reason: str,
        errors: list[str],
    ) -> dict[str, Any]:
        """Build a structured error result.

        Args:
            rule: Rule dictionary when available.
            reason: Error reason.
            errors: Validation or execution errors.

        Returns:
            Structured rule evaluation error result.
        """
        return {
            "rule_id": rule.get("rule_id", "") if isinstance(rule, dict) else "",
            "rule_type": rule.get("rule_type", "") if isinstance(rule, dict) else "",
            "matched": False,
            "decision": "error",
            "action": "",
            "reason": reason,
            "conditions_evaluated": [],
            "confidence": rule.get("confidence", 0.0) if isinstance(rule, dict) else 0.0,
            "service": rule.get("service", "") if isinstance(rule, dict) else "",
            "errors": errors,
        }

    def _flatten_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten direct or batch rule evaluation results.

        Args:
            results: Direct rule results or batch results.

        Returns:
            Flat list of rule evaluation results.
        """
        if not isinstance(results, list):
            return []
        flattened: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("results"), list):
                flattened.extend(
                    item for item in result["results"] if isinstance(item, dict)
                )
            elif isinstance(result, dict):
                flattened.append(result)
        return flattened

    def _count_decisions(self, results: list[dict[str, Any]], decisions: set[str]) -> int:
        """Count matched results by normalized decision value.

        Args:
            results: Rule evaluation results.
            decisions: Decision names to count.

        Returns:
            Decision count.
        """
        return sum(
            1
            for result in results
            if str(result.get("decision", "")).lower() in decisions
        )

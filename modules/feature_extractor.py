"""Healthcare feature extraction for policy intelligence workflows.

This module extracts structured, model-ready healthcare features from policy
text, LangChain documents, summarizer outputs, and rule extractor outputs.
Features are validated with Pydantic and can be exported to JSON, CSV, or a
Pandas DataFrame for downstream ML, rule engine, claim decision, explainability,
and dashboard modules.

Example:
    ```python
    from modules.feature_extractor import HealthcareFeatureExtractor

    extractor = HealthcareFeatureExtractor()

    text = '''
    Lumbar spine MRI CPT 72148 is covered for diagnosis M54.16
    after six weeks of conservative therapy. Prior authorization is required.
    Allowed amount is $750.
    '''

    features = extractor.extract_features_from_text(text)
    df = extractor.to_dataframe([features])

    extractor.save_features([features], "output/features.json")

    print(df)
    ```
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator


class ContractTerms(BaseModel):
    """Pydantic schema for payer-provider contract terms."""

    allowed_amount: float | None = None
    copay: float | None = None
    coinsurance: float | None = None
    currency: str = "USD"


class HealthcareFeatures(BaseModel):
    """Pydantic schema for one healthcare feature record."""

    patient_age: int | None = None
    age_requirement: str | None = None
    gender_requirement: str | None = None
    icd_codes: list[str] = Field(default_factory=list)
    cpt_codes: list[str] = Field(default_factory=list)
    hcpcs_codes: list[str] = Field(default_factory=list)
    diagnosis: str = ""
    procedure: str = ""
    service: str = ""
    therapy_weeks: int | None = None
    prior_authorization_required: bool | None = None
    medical_necessity_criteria: list[str] = Field(default_factory=list)
    excluded_services: list[str] = Field(default_factory=list)
    covered_services: list[str] = Field(default_factory=list)
    frequency_limit: str = ""
    contract_terms: ContractTerms = Field(default_factory=ContractTerms)
    provider_specialty: str = ""
    documentation_required: list[str] = Field(default_factory=list)
    coverage_type: str = ""
    effective_date: str = ""
    termination_date: str = ""
    source_text: str = ""
    source_document: str = ""
    page_number: int | None = None
    document_type: str = ""
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Validate confidence score.

        Args:
            value: Candidate confidence score.

        Returns:
            Confidence score.

        Raises:
            ValueError: If confidence is outside the 0 to 1 range.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value

    @field_validator("patient_age")
    @classmethod
    def validate_patient_age(cls, value: int | None) -> int | None:
        """Validate patient age.

        Args:
            value: Candidate patient age.

        Returns:
            Patient age or ``None``.

        Raises:
            ValueError: If patient age is negative.
        """
        if value is not None and value < 0:
            raise ValueError("patient_age cannot be negative.")
        return value


class HealthcareFeatureExtractor:
    """Extracts structured healthcare features from content and rules."""

    def __init__(
        self,
        llm: Any | None = None,
        llm_model: str = "gpt-4.1",
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        max_retries: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the healthcare feature extractor.

        Args:
            llm: Optional LangChain-compatible chat model.
            llm_model: OpenAI model name.
            temperature: LLM temperature.
            request_timeout: LLM request timeout in seconds.
            max_retries: Number of retries for failed LLM calls.
            logger: Optional logger instance.

        Returns:
            None.

        Raises:
            ValueError: If ``max_retries`` is negative.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")

        self.llm = llm
        self.llm_model = llm_model
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)
        self._last_processing_time_seconds = 0.0

    def initialize_llm(self) -> Any | None:
        """Initialize an OpenAI chat model for feature extraction.

        Args:
            None.

        Returns:
            LangChain-compatible chat model when available; otherwise ``None``.

        Raises:
            This method logs failures and returns ``None`` instead of raising.
        """
        if self.llm is not None:
            return self.llm

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OPENAI_API_KEY is not configured.")
            return None

        try:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.llm_model,
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=api_key,
            )
            self.logger.info("Feature extraction LLM initialized: %s", self.llm_model)
            return self.llm
        except Exception as error:
            self.logger.error("Failed to initialize feature extraction LLM: %s", error)
            return None

    def extract_features_from_text(self, text: str) -> dict[str, Any]:
        """Extract structured healthcare features from raw text.

        Args:
            text: Raw healthcare content text.

        Returns:
            Structured healthcare feature dictionary.

        Raises:
            This method logs invalid input, API failures, and JSON failures
            instead of raising.
        """
        start_time = time.perf_counter()
        if not isinstance(text, str) or not text.strip():
            self.logger.warning("Empty text provided for feature extraction.")
            self._last_processing_time_seconds = 0.0
            return self._empty_features()

        self.logger.info("Feature extraction started from raw text.")
        regex_features = self._extract_regex_features(text)
        llm_features: dict[str, Any] = {}
        llm = self.initialize_llm()
        if llm is not None:
            llm_features = self._call_llm_for_features(text, llm)

        merged_features = self._merge_features(
            self._empty_features(),
            llm_features,
            regex_features,
        )
        merged_features["source_text"] = text
        merged_features = self._post_process_feature_record(merged_features)

        self._last_processing_time_seconds = time.perf_counter() - start_time
        self.logger.info(
            "Number of feature records extracted: 1 in %.2f seconds.",
            self._last_processing_time_seconds,
        )
        return merged_features

    def extract_features_from_documents(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Extract features from LangChain documents.

        Args:
            documents: LangChain ``Document`` objects.

        Returns:
            List of structured feature dictionaries with source metadata.
        """
        if not isinstance(documents, list):
            self.logger.error("Documents input must be a list.")
            return []

        features_list: list[dict[str, Any]] = []
        for document in documents:
            if not isinstance(document, Document):
                self.logger.warning("Invalid document skipped: %s", type(document).__name__)
                continue
            if not document.page_content or not document.page_content.strip():
                self.logger.warning("Empty document skipped during feature extraction.")
                continue

            features = self.extract_features_from_text(document.page_content)
            metadata = document.metadata or {}
            features["source_document"] = str(
                metadata.get("filename") or metadata.get("source") or ""
            )
            features["page_number"] = self._coerce_page_number(
                metadata.get("page_number", metadata.get("page"))
            )
            features["document_type"] = str(metadata.get("document_type") or "")
            features_list.append(features)

        consolidated_features = self.consolidate_feature_records(features_list)
        self.logger.info(
            "Number of document feature records extracted: %s, consolidated to: %s",
            len(features_list),
            len(consolidated_features),
        )
        return consolidated_features

    def extract_features_from_summary(
        self,
        summary: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract features from a structured summarizer output.

        Args:
            summary: Structured summary dictionary or summarizer result list.

        Returns:
            Structured feature dictionary.
        """
        payload = self._extract_summary_payload(summary)
        if not payload:
            self.logger.warning("Missing summary provided for feature extraction.")
            return self._empty_features()

        text = self._summary_to_text(payload)
        features = self._extract_regex_features(text)
        features["document_type"] = str(payload.get("document_type", ""))
        features["covered_services"] = self._coerce_string_list(payload.get("covered_services", []))
        features["excluded_services"] = self._coerce_string_list(payload.get("excluded_services", []))
        features["medical_necessity_criteria"] = self._coerce_string_list(
            payload.get("medical_necessity", [])
        )
        features["documentation_required"] = self._coerce_string_list(
            payload.get("required_documentation", [])
        )
        prior_auth = str(payload.get("prior_authorization", "")).lower()
        if prior_auth:
            features["prior_authorization_required"] = (
                "required" in prior_auth and "not required" not in prior_auth
            )
        key_dates = self._coerce_string_list(payload.get("key_dates", []))
        if key_dates:
            features["effective_date"] = key_dates[0]
        features["source_text"] = text
        return self._post_process_feature_record(features)

    def extract_features_from_rules(self, rules: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract model-ready features from JSON business rules.

        Args:
            rules: Rule dictionaries generated by ``rule_extractor.py``.

        Returns:
            Structured feature dictionary.
        """
        if not isinstance(rules, list) or not rules:
            self.logger.warning("Missing rules provided for feature extraction.")
            return self._empty_features()

        features = self._empty_features()
        source_fragments: list[str] = []
        confidences: list[float] = []

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rule_type = str(rule.get("rule_type", ""))
            source_fragments.append(str(rule.get("source_text", "")))
            if isinstance(rule.get("confidence"), (int, float)):
                confidences.append(float(rule["confidence"]))
            if rule.get("service"):
                features["service"] = str(rule["service"])
            if rule_type == "coverage":
                features["coverage_type"] = "covered"
                if rule.get("service"):
                    features["covered_services"].append(str(rule["service"]))
            if rule_type == "exclusion":
                features["coverage_type"] = "excluded"
                if rule.get("service"):
                    features["excluded_services"].append(str(rule["service"]))
            if rule_type == "prior_authorization":
                features["prior_authorization_required"] = True
            if rule_type == "coding":
                regex_features = self._extract_regex_features(str(rule.get("source_text", "")))
                features = self._merge_features(features, regex_features)

            for condition in rule.get("conditions", []):
                if isinstance(condition, dict):
                    self._apply_condition_to_features(features, condition)

        features["source_text"] = " ".join(fragment for fragment in source_fragments if fragment)
        if confidences:
            features["confidence"] = round(sum(confidences) / len(confidences), 2)
        return self._post_process_feature_record(features)

    def consolidate_feature_records(
        self,
        features_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize and consolidate fragmented feature records.

        Args:
            features_list: Raw extracted feature records.

        Returns:
            Cleaner, consolidated feature records suitable for UI display and
            downstream processing.
        """
        if not isinstance(features_list, list):
            return []

        normalized_records = [
            self._post_process_feature_record(features)
            for features in features_list
            if isinstance(features, dict)
        ]
        normalized_records = [
            record for record in normalized_records if self._has_meaningful_features(record)
        ]
        if not normalized_records:
            return []

        policy_level_record = self._build_policy_level_record(normalized_records)
        primary_record = self._select_primary_record(normalized_records)
        secondary_records: list[dict[str, Any]] = []

        for record in normalized_records:
            if record is primary_record:
                continue
            if self._should_merge_into_primary(primary_record, record):
                primary_record = self._merge_records(primary_record, record)
            else:
                secondary_records.append(record)

        consolidated = [self._post_process_feature_record(policy_level_record)]
        if self._record_is_materially_distinct(primary_record, consolidated[0]):
            consolidated.append(self._post_process_feature_record(primary_record))
        for record in secondary_records:
            candidate = self._post_process_feature_record(record)
            if self._record_is_materially_distinct(candidate, consolidated[0]):
                consolidated.append(candidate)

        unique_records: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for record in consolidated:
            dedupe_key = json.dumps(
                {
                    "procedure": record.get("procedure", ""),
                    "service": record.get("service", ""),
                    "coverage_type": record.get("coverage_type", ""),
                    "icd_codes": record.get("icd_codes", []),
                    "cpt_codes": record.get("cpt_codes", []),
                    "hcpcs_codes": record.get("hcpcs_codes", []),
                    "therapy_weeks": record.get("therapy_weeks"),
                    "prior_authorization_required": record.get("prior_authorization_required"),
                },
                sort_keys=True,
                default=str,
            )
            if dedupe_key not in seen_keys:
                seen_keys.add(dedupe_key)
                unique_records.append(record)

        return unique_records

    def validate_feature_schema(self, features: dict[str, Any]) -> dict[str, Any]:
        """Validate one feature object using the Pydantic schema.

        Args:
            features: Candidate feature dictionary.

        Returns:
            Validation result with ``valid`` and ``errors`` fields.
        """
        try:
            HealthcareFeatures.model_validate(features)
            return {"valid": True, "errors": []}
        except ValidationError as error:
            errors = [item["msg"] for item in error.errors()]
            self.logger.warning("Invalid feature record: %s", errors)
            return {"valid": False, "errors": errors}

    def validate_features(self, features_list: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Validate multiple feature records.

        Args:
            features_list: Candidate feature dictionaries.

        Returns:
            Dictionary containing ``valid_features`` and ``invalid_features``.
        """
        if not isinstance(features_list, list):
            self.logger.error("features_list must be a list.")
            return {"valid_features": [], "invalid_features": []}

        valid_features: list[dict[str, Any]] = []
        invalid_features: list[dict[str, Any]] = []
        for features in features_list:
            validation = self.validate_feature_schema(features)
            if validation["valid"]:
                valid_features.append(features)
            else:
                invalid_features.append({"features": features, "errors": validation["errors"]})

        self.logger.info(
            "Validation results: %s valid, %s invalid.",
            len(valid_features),
            len(invalid_features),
        )
        return {"valid_features": valid_features, "invalid_features": invalid_features}

    def to_dataframe(self, features_list: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert feature records into a Pandas DataFrame.

        Args:
            features_list: Feature dictionaries.

        Returns:
            Pandas DataFrame suitable for downstream ML pipelines.
        """
        if not isinstance(features_list, list):
            self.logger.error("features_list must be a list.")
            return pd.DataFrame()
        return pd.DataFrame([self._flatten_features(features) for features in features_list])

    def save_features(self, features_list: list[dict[str, Any]], output_path: str | Path) -> bool:
        """Save feature records as JSON or CSV.

        Args:
            features_list: Feature dictionaries to save.
            output_path: Destination path. ``.csv`` writes CSV; all other
                extensions write JSON.

        Returns:
            ``True`` when saved successfully; otherwise ``False``.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".csv":
                self.to_dataframe(features_list).to_csv(path, index=False)
            else:
                path.write_text(json.dumps(features_list, indent=2), encoding="utf-8")
            self.logger.info("Features saved: %s", path)
            return True
        except Exception as error:
            self.logger.error("Failed to save features to %s: %s", path, error)
            return False

    def load_features(self, input_path: str | Path) -> list[dict[str, Any]]:
        """Load feature records from JSON or CSV.

        Args:
            input_path: Source JSON or CSV path.

        Returns:
            Loaded feature dictionaries, or an empty list on failure.
        """
        path = Path(input_path)
        if not path.exists():
            self.logger.error("Feature file does not exist: %s", path)
            return []

        try:
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path).to_dict(orient="records")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, list) else []
        except Exception as error:
            self.logger.error("Failed to load features from %s: %s", path, error)
            return []

    def get_feature_statistics(self, features_list: list[dict[str, Any]]) -> dict[str, int | float]:
        """Calculate feature extraction statistics.

        Args:
            features_list: Feature dictionaries.

        Returns:
            Statistics for feature coverage and average confidence.
        """
        if not isinstance(features_list, list) or not features_list:
            return {
                "total_feature_records": 0,
                "records_with_icd_codes": 0,
                "records_with_cpt_codes": 0,
                "records_with_prior_auth": 0,
                "records_with_contract_terms": 0,
                "average_confidence": 0.0,
            }

        confidences = [
            float(features.get("confidence", 0.0))
            for features in features_list
            if isinstance(features.get("confidence", 0.0), (int, float))
        ]
        return {
            "total_feature_records": len(features_list),
            "records_with_icd_codes": sum(1 for item in features_list if item.get("icd_codes")),
            "records_with_cpt_codes": sum(1 for item in features_list if item.get("cpt_codes")),
            "records_with_prior_auth": sum(
                1 for item in features_list if item.get("prior_authorization_required") is True
            ),
            "records_with_contract_terms": sum(
                1 for item in features_list if self._has_contract_terms(item)
            ),
            "average_confidence": round(sum(confidences) / len(confidences), 2)
            if confidences
            else 0.0,
        }

    def _call_llm_for_features(self, text: str, llm: Any) -> dict[str, Any]:
        """Call an LLM to extract healthcare features.

        Args:
            text: Source healthcare content.
            llm: LangChain-compatible chat model.

        Returns:
            Parsed feature dictionary, or an empty dictionary on failure.
        """
        messages = self._build_messages(text)
        for attempt in range(1, self.max_retries + 2):
            try:
                self.logger.info("Submitting feature extraction API request. Attempt: %s", attempt)
                response = llm.invoke(messages)
                content = self._extract_response_content(response)
                parsed = json.loads(self._strip_code_fences(content))
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, RuntimeError, TimeoutError, ValueError) as error:
                self.logger.error("Feature extraction API attempt failed: %s", error)
        return {}

    def _build_messages(self, text: str) -> list[SystemMessage | HumanMessage]:
        """Build LLM prompt messages for feature extraction.

        Args:
            text: Healthcare content text.

        Returns:
            LangChain system and human messages.
        """
        system_prompt = (
            "You are an expert healthcare policy analyst and healthcare feature "
            "extraction engine. Your task is to extract structured healthcare "
            "features from healthcare policy, guideline, coding, and contract "
            "text. Extract only information explicitly supported by the text. "
            "Do not invent missing values. If a value is missing, return null. "
            "Return valid JSON only."
        )
        human_prompt = (
            "Extract ICD codes, CPT codes, HCPCS codes, diagnoses, procedures, "
            "services, age requirements, therapy duration, prior authorization, "
            "medical necessity criteria, exclusions, documentation requirements, "
            "payer contract terms, reimbursement terms, and effective dates. "
            "Return one JSON object using the required schema.\n\n"
            f"Healthcare content:\n{text}"
        )
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    def _extract_response_content(self, response: Any) -> str:
        """Extract text from an LLM response.

        Args:
            response: LangChain response or response-like object.

        Returns:
            Response content.

        Raises:
            ValueError: If response content is unavailable.
        """
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response did not contain text content.")
        return content.strip()

    def _extract_regex_features(self, text: str) -> dict[str, Any]:
        """Extract healthcare features with deterministic regex patterns.

        Args:
            text: Source healthcare text.

        Returns:
            Feature dictionary populated with regex-derived values.
        """
        features = self._empty_features()
        features["icd_codes"] = self._unique_matches(
            r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b",
            text,
        )
        features["cpt_codes"] = self._unique_matches(r"\b\d{5}\b", text)
        features["hcpcs_codes"] = self._unique_matches(r"\b[A-Z]\d{4}\b", text)
        features["therapy_weeks"] = self._extract_therapy_weeks(text)
        features["prior_authorization_required"] = self._extract_prior_auth(text)
        features["contract_terms"] = self._extract_contract_terms(text)
        features["effective_date"] = self._first_or_empty(self._extract_dates(text))
        features["coverage_type"] = self._extract_coverage_type(text)
        features["procedure"] = self._extract_procedure(text)
        features["service"] = features["procedure"]
        features["documentation_required"] = self._extract_documentation_requirements(text)
        features["source_text"] = text
        self.logger.info(
            "Regex extraction results: ICD=%s CPT=%s HCPCS=%s",
            len(features["icd_codes"]),
            len(features["cpt_codes"]),
            len(features["hcpcs_codes"]),
        )
        return features

    def _empty_features(self) -> dict[str, Any]:
        """Create an empty feature record using the required schema.

        Args:
            None.

        Returns:
            Empty feature dictionary.
        """
        return {
            "patient_age": None,
            "age_requirement": None,
            "gender_requirement": None,
            "icd_codes": [],
            "cpt_codes": [],
            "hcpcs_codes": [],
            "diagnosis": "",
            "procedure": "",
            "service": "",
            "therapy_weeks": None,
            "prior_authorization_required": None,
            "medical_necessity_criteria": [],
            "excluded_services": [],
            "covered_services": [],
            "frequency_limit": "",
            "contract_terms": {
                "allowed_amount": None,
                "copay": None,
                "coinsurance": None,
                "currency": "USD",
            },
            "provider_specialty": "",
            "documentation_required": [],
            "coverage_type": "",
            "effective_date": "",
            "termination_date": "",
            "source_text": "",
            "source_document": "",
            "page_number": None,
            "document_type": "",
            "confidence": 0.0,
        }

    def _merge_features(self, *feature_sets: dict[str, Any]) -> dict[str, Any]:
        """Merge feature dictionaries without discarding populated values.

        Args:
            *feature_sets: Feature dictionaries ordered by increasing priority.

        Returns:
            Merged feature dictionary.
        """
        merged = self._empty_features()
        for features in feature_sets:
            if not isinstance(features, dict):
                continue
            for key, value in features.items():
                if key == "contract_terms" and isinstance(value, dict):
                    merged["contract_terms"].update(
                        {term_key: term_value for term_key, term_value in value.items() if term_value not in (None, "", [])}
                    )
                elif isinstance(merged.get(key), list):
                    merged[key] = self._merge_lists(merged[key], value)
                elif value not in (None, "", []):
                    merged[key] = value
        return merged

    def _merge_records(self, primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
        """Merge two feature records and re-normalize the result."""
        merged = self._merge_features(primary, secondary)
        return self._post_process_feature_record(merged)

    def _merge_lists(self, existing: list[Any], value: Any) -> list[Any]:
        """Merge list values with de-duplication.

        Args:
            existing: Existing list.
            value: New list or scalar.

        Returns:
            Merged list.
        """
        incoming = value if isinstance(value, list) else [value] if value not in (None, "") else []
        merged: list[Any] = []
        for item in existing + incoming:
            if item not in merged:
                merged.append(item)
        return merged

    def _apply_condition_to_features(self, features: dict[str, Any], condition: dict[str, Any]) -> None:
        """Apply a rule condition to a feature dictionary.

        Args:
            features: Feature dictionary to mutate.
            condition: Rule condition.

        Returns:
            None.
        """
        field = str(condition.get("field", ""))
        value = condition.get("value")
        if field in {"therapy_weeks", "therapy_duration"}:
            features["therapy_weeks"] = self._coerce_int(value)
        elif field == "prior_authorization":
            features["prior_authorization_required"] = bool(value)
        elif field in {"icd_codes", "cpt_codes", "hcpcs_codes"}:
            features[field] = self._merge_lists(features[field], value)
        elif field in {"allowed_amount", "copay", "coinsurance"}:
            features["contract_terms"][field] = self._coerce_float(value)

    def _flatten_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested feature fields for DataFrame output.

        Args:
            features: Feature dictionary.

        Returns:
            Flattened feature dictionary.
        """
        flattened = dict(features)
        contract_terms = flattened.pop("contract_terms", {}) or {}
        for key, value in contract_terms.items():
            flattened[f"contract_{key}"] = value
        return flattened

    def _extract_summary_payload(self, summary: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """Extract the summary payload from supported summarizer shapes.

        Args:
            summary: Summary dictionary or summarizer result list.

        Returns:
            Summary payload dictionary.
        """
        if isinstance(summary, list):
            if not summary or not isinstance(summary[0], dict):
                return {}
            payload = summary[0].get("summary", summary[0])
            return payload if isinstance(payload, dict) else {}
        if isinstance(summary, dict):
            payload = summary.get("summary", summary)
            return payload if isinstance(payload, dict) else {}
        return {}

    def _summary_to_text(self, payload: dict[str, Any]) -> str:
        """Convert a structured summary to text.

        Args:
            payload: Summary payload.

        Returns:
            Multiline summary text.
        """
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"{key}: {'; '.join(str(item) for item in value)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _coerce_string_list(self, value: Any) -> list[str]:
        """Coerce a value into a string list.

        Args:
            value: Candidate value.

        Returns:
            List of strings.
        """
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if value in (None, ""):
            return []
        return [str(value)]

    def _coerce_page_number(self, value: Any) -> int | None:
        """Coerce page metadata into an integer.

        Args:
            value: Candidate page value.

        Returns:
            Integer page or ``None``.
        """
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_int(self, value: Any) -> int | None:
        """Coerce a value into an integer.

        Args:
            value: Candidate numeric value.

        Returns:
            Integer or ``None``.
        """
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> float | None:
        """Coerce a value into a float.

        Args:
            value: Candidate numeric value.

        Returns:
            Float or ``None``.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _unique_matches(self, pattern: str, text: str) -> list[str]:
        """Return unique regex matches in source order.

        Args:
            pattern: Regex pattern.
            text: Source text.

        Returns:
            Unique matches.
        """
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        unique: list[str] = []
        for match in matches:
            value = str(match).upper()
            if value not in unique:
                unique.append(value)
        return unique

    def _extract_therapy_weeks(self, text: str) -> int | None:
        """Extract therapy duration in weeks.

        Args:
            text: Source text.

        Returns:
            Therapy weeks or ``None``.
        """
        word_numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        match = re.search(
            r"\b(\d+|one|two|three|four|five|six)\s+(week|weeks)\b",
            text.lower(),
        )
        if not match:
            return None
        raw_value = match.group(1)
        return int(raw_value) if raw_value.isdigit() else word_numbers[raw_value]

    def _extract_prior_auth(self, text: str) -> bool | None:
        """Extract prior authorization requirement.

        Args:
            text: Source text.

        Returns:
            Prior authorization requirement or ``None``.
        """
        lowered = text.lower()
        if "prior authorization" not in lowered and "preauthorization" not in lowered:
            return None
        return not ("not required" in lowered or "no prior authorization" in lowered)

    def _extract_contract_terms(self, text: str) -> dict[str, Any]:
        """Extract contract and reimbursement terms.

        Args:
            text: Source text.

        Returns:
            Contract terms dictionary.
        """
        terms = {
            "allowed_amount": None,
            "copay": None,
            "coinsurance": None,
            "currency": "USD",
        }
        amount = re.search(r"\$\s?([0-9]+(?:\.[0-9]{1,2})?)", text)
        if amount:
            terms["allowed_amount"] = float(amount.group(1))
        percentage = re.search(r"\b([0-9]{1,3})%\b", text)
        if percentage:
            terms["coinsurance"] = float(percentage.group(1))
        copay = re.search(r"copay(?:ment)?\s+(?:is\s+)?\$\s?([0-9]+(?:\.[0-9]{1,2})?)", text, re.IGNORECASE)
        if copay:
            terms["copay"] = float(copay.group(1))
        return terms

    def _extract_dates(self, text: str) -> list[str]:
        """Extract common date formats.

        Args:
            text: Source text.

        Returns:
            Date strings.
        """
        numeric_dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
        named_dates = re.findall(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            text,
            flags=re.IGNORECASE,
        )
        return numeric_dates + named_dates

    def _extract_coverage_type(self, text: str) -> str:
        """Extract coverage type from text.

        Args:
            text: Source text.

        Returns:
            Coverage type string.
        """
        lowered = text.lower()
        if "excluded" in lowered or "not covered" in lowered:
            return "excluded"
        if "covered" in lowered:
            return "covered"
        return ""

    def _extract_procedure(self, text: str) -> str:
        """Extract a simple procedure phrase.

        Args:
            text: Source text.

        Returns:
            Procedure string.
        """
        patterns = [
            (r"\blumbar spine mri\b", "Lumbar Spine MRI"),
            (r"\bscreening mri\b", "Screening MRI"),
            (r"\broutine repeat imaging\b", "Routine Repeat Imaging"),
            (r"\brepeat lumbar mri\b", "Repeat Lumbar MRI"),
            (r"\bexperimental imaging protocols?\b", "Experimental Imaging Protocols"),
            (r"\bmri lumbar spine\b", "Lumbar Spine MRI"),
        ]
        lowered = text.lower()
        for pattern, label in patterns:
            if re.search(pattern, lowered):
                return label

        match = re.search(r"\b([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,4} (?:MRI|CT))\b", text)
        candidate = match.group(1).strip() if match else ""
        return candidate if self._is_meaningful_procedure(candidate) else ""

    def _extract_documentation_requirements(self, text: str) -> list[str]:
        """Extract documentation requirements from policy language."""
        requirements: list[str] = []
        lowered = text.lower()
        patterns = [
            ("valid cpt or hcpcs", "Valid CPT or HCPCS procedure code"),
            ("supported icd-10", "Supported ICD-10 diagnosis code"),
            ("documentation supporting medical necessity", "Documentation supporting medical necessity"),
            ("ordering provider information", "Ordering provider information"),
            ("rendering provider information", "Rendering provider information"),
            ("date of onset", "Date of symptom onset"),
            ("history and physical examination", "History and physical examination"),
        ]
        for marker, label in patterns:
            if marker in lowered and label not in requirements:
                requirements.append(label)
        return requirements

    def _post_process_feature_record(self, features: dict[str, Any]) -> dict[str, Any]:
        """Normalize one feature record for cleaner downstream use."""
        record = self._merge_features(features)
        record["icd_codes"] = self._normalize_code_list(record.get("icd_codes", []))
        record["cpt_codes"] = self._normalize_code_list(record.get("cpt_codes", []))
        record["hcpcs_codes"] = self._normalize_code_list(record.get("hcpcs_codes", []))
        record["covered_services"] = self._normalize_string_list(record.get("covered_services", []))
        record["excluded_services"] = self._normalize_string_list(record.get("excluded_services", []))
        record["medical_necessity_criteria"] = self._normalize_string_list(
            record.get("medical_necessity_criteria", [])
        )
        record["documentation_required"] = self._normalize_string_list(
            record.get("documentation_required", [])
        )

        procedure_candidate = self._select_procedure_candidate(record)
        record["procedure"] = procedure_candidate
        if procedure_candidate:
            record["service"] = procedure_candidate
        elif not self._is_meaningful_procedure(str(record.get("service", ""))):
            record["service"] = ""

        if not record.get("diagnosis"):
            record["diagnosis"] = self._derive_diagnosis(record)

        if not record.get("coverage_type"):
            if record.get("excluded_services"):
                record["coverage_type"] = "excluded"
            elif record.get("covered_services") or record.get("procedure"):
                record["coverage_type"] = "covered"
        elif record.get("coverage_type") == "covered" and record.get("excluded_services"):
            record["coverage_type"] = "covered_with_exclusions"

        record["confidence"] = self._calculate_feature_confidence(record)
        return record

    def _build_policy_level_record(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one consolidated policy-level feature record."""
        policy_record = self._empty_features()
        source_documents: list[str] = []
        source_pages: list[int] = []

        for record in records:
            policy_record = self._merge_features(policy_record, record)
            source_document = str(record.get("source_document", "")).strip()
            if source_document and source_document not in source_documents:
                source_documents.append(source_document)
            page_number = self._coerce_page_number(record.get("page_number"))
            if page_number is not None and page_number not in source_pages:
                source_pages.append(page_number)

        policy_record["procedure"] = self._select_procedure_candidate(policy_record)
        policy_record["service"] = policy_record["procedure"]
        if source_documents:
            policy_record["source_document"] = source_documents[0]
        if source_pages:
            policy_record["page_number"] = min(source_pages)
        return self._post_process_feature_record(policy_record)

    def _select_primary_record(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Select the strongest record as the primary consolidated feature object."""
        return max(records, key=self._record_strength)

    def _record_strength(self, record: dict[str, Any]) -> tuple[int, int, int, int]:
        """Return a sortable strength score for a feature record."""
        return (
            1 if record.get("procedure") else 0,
            len(record.get("cpt_codes", [])) + len(record.get("icd_codes", [])),
            1 if record.get("therapy_weeks") is not None else 0,
            1 if record.get("prior_authorization_required") is not None else 0,
        )

    def _record_is_materially_distinct(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        """Return whether two records are meaningfully different for display."""
        left_key = (
            str(left.get("procedure", "")),
            str(left.get("coverage_type", "")),
            tuple(left.get("icd_codes", [])),
            tuple(left.get("cpt_codes", [])),
            left.get("therapy_weeks"),
            left.get("prior_authorization_required"),
        )
        right_key = (
            str(right.get("procedure", "")),
            str(right.get("coverage_type", "")),
            tuple(right.get("icd_codes", [])),
            tuple(right.get("cpt_codes", [])),
            right.get("therapy_weeks"),
            right.get("prior_authorization_required"),
        )
        return left_key != right_key

    def _should_merge_into_primary(
        self,
        primary: dict[str, Any],
        candidate: dict[str, Any],
    ) -> bool:
        """Return whether a candidate record should merge into the primary one."""
        primary_procedure = str(primary.get("procedure", "")).strip().lower()
        candidate_procedure = str(candidate.get("procedure", "")).strip().lower()
        if not candidate_procedure:
            return True
        if candidate_procedure == primary_procedure:
            return True
        if candidate_procedure == "imaging":
            return True
        if candidate_procedure.startswith("lumbar") and primary_procedure.startswith("lumbar"):
            return True
        if candidate.get("coverage_type") == "covered" and primary.get("coverage_type") != "excluded":
            return True
        return False

    def _select_procedure_candidate(self, record: dict[str, Any]) -> str:
        """Return the best normalized procedure for a feature record."""
        candidates = [
            str(record.get("procedure", "")),
            str(record.get("service", "")),
            *[str(item) for item in record.get("covered_services", [])],
            *[str(item) for item in record.get("excluded_services", [])],
            self._extract_procedure(str(record.get("source_text", ""))),
        ]
        for candidate in candidates:
            normalized = self._normalize_procedure_candidate(candidate)
            if normalized:
                return normalized
        return ""

    def _normalize_procedure_candidate(self, candidate: str) -> str:
        """Normalize a possible procedure value and reject policy statements."""
        normalized = " ".join(str(candidate).split()).strip()
        if not normalized:
            return ""

        label_map = {
            "sample lumbar spine mri": "Lumbar Spine MRI",
            "lumbar spine mri": "Lumbar Spine MRI",
            "mri lumbar spine": "Lumbar Spine MRI",
            "screening mri": "Screening MRI",
            "routine repeat imaging": "Routine Repeat Imaging",
            "repeat lumbar mri": "Repeat Lumbar MRI",
            "experimental imaging protocols": "Experimental Imaging Protocols",
            "imaging": "",
        }
        mapped = label_map.get(normalized.lower())
        if mapped is not None:
            return mapped

        return normalized if self._is_meaningful_procedure(normalized) else ""

    def _is_meaningful_procedure(self, value: str) -> bool:
        """Return whether a candidate looks like a real procedure or service."""
        candidate = " ".join(str(value).split()).strip()
        if not candidate:
            return False

        lowered = candidate.lower()
        blocked_markers = [
            "claims must include",
            "valid cpt or hcpcs",
            "duration and type of conservative therapy",
            "date of onset",
            "history and physical examination",
            "prior authorization required",
            "documentation supporting",
            "ordering provider information",
            "rendering provider information",
        ]
        if any(marker in lowered for marker in blocked_markers):
            return False

        allowed_markers = ["mri", "ct", "imaging", "scan", "radiograph", "ultrasound"]
        return any(marker in lowered for marker in allowed_markers)

    def _normalize_code_list(self, values: Any) -> list[str]:
        """Normalize and deduplicate code lists."""
        if not isinstance(values, list):
            values = [values] if values not in (None, "") else []
        normalized: list[str] = []
        for value in values:
            text = str(value).strip().upper()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def _normalize_string_list(self, values: Any) -> list[str]:
        """Normalize and deduplicate string lists."""
        if not isinstance(values, list):
            values = [values] if values not in (None, "") else []
        normalized: list[str] = []
        for value in values:
            text = " ".join(str(value).split()).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def _derive_diagnosis(self, record: dict[str, Any]) -> str:
        """Derive a diagnosis label from known codes or source text."""
        diagnosis_map = {
            "M54.16": "Lumbar Radiculopathy",
            "M48.061": "Lumbar Spinal Stenosis",
        }
        for code in record.get("icd_codes", []):
            if code in diagnosis_map:
                return diagnosis_map[code]

        source_text = str(record.get("source_text", "")).lower()
        if "radiculopathy" in source_text:
            return "Lumbar Radiculopathy"
        if "stenosis" in source_text:
            return "Lumbar Spinal Stenosis"
        if "low back pain" in source_text:
            return "Low Back Pain"
        return ""

    def _calculate_feature_confidence(self, features: dict[str, Any]) -> float:
        """Calculate a richer confidence score for feature extraction."""
        if not self._has_meaningful_features(features):
            return 0.0

        score = 0.45
        if features.get("procedure"):
            score += 0.12
        if features.get("icd_codes"):
            score += 0.08
        if features.get("cpt_codes"):
            score += 0.08
        if features.get("hcpcs_codes"):
            score += 0.06
        if features.get("therapy_weeks") is not None:
            score += 0.06
        if features.get("prior_authorization_required") is not None:
            score += 0.06
        if features.get("coverage_type"):
            score += 0.05
        if self._has_contract_terms(features):
            score += 0.06
        if features.get("documentation_required"):
            score += 0.05
        if features.get("medical_necessity_criteria"):
            score += 0.05
        if features.get("effective_date"):
            score += 0.03

        return round(min(score, 0.95), 2)

    def _first_or_empty(self, values: list[str]) -> str:
        """Return the first value or an empty string.

        Args:
            values: Candidate values.

        Returns:
            First value or empty string.
        """
        return values[0] if values else ""

    def _strip_code_fences(self, content: str) -> str:
        """Remove Markdown code fences from model output.

        Args:
            content: Raw model response.

        Returns:
            Content without wrapping code fences.
        """
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped

    def _has_meaningful_features(self, features: dict[str, Any]) -> bool:
        """Determine whether a feature record contains extracted values.

        Args:
            features: Feature dictionary.

        Returns:
            ``True`` when meaningful fields are populated.
        """
        return bool(
            features.get("icd_codes")
            or features.get("cpt_codes")
            or features.get("hcpcs_codes")
            or features.get("therapy_weeks") is not None
            or features.get("prior_authorization_required") is not None
            or self._has_contract_terms(features)
            or features.get("coverage_type")
        )

    def _has_contract_terms(self, features: dict[str, Any]) -> bool:
        """Determine whether contract terms contain reimbursement values.

        Args:
            features: Feature dictionary.

        Returns:
            ``True`` when contract terms include amount, copay, or coinsurance.
        """
        terms = features.get("contract_terms", {})
        return bool(
            isinstance(terms, dict)
            and (
                terms.get("allowed_amount") is not None
                or terms.get("copay") is not None
                or terms.get("coinsurance") is not None
            )
        )

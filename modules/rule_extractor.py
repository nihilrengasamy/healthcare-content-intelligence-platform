"""Healthcare policy rule extraction.

This module converts natural-language healthcare policy content into
structured, machine-readable JSON business rules. The extracted rules are
validated with Pydantic and are designed for downstream rule engine,
feature extraction, claim decision, and explainability modules.

Example:
    ```python
    from modules.rule_extractor import HealthcareRuleExtractor

    extractor = HealthcareRuleExtractor()

    text = '''
    Lumbar spine MRI is covered after six weeks of conservative therapy
    unless neurological deficits are present. Prior authorization is required.
    '''

    rules = extractor.extract_rules_from_text(text)
    validated = extractor.validate_rules(rules)

    extractor.save_rules(validated["valid_rules"], "output/rules.json")

    print(rules)
    ```
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()


RuleType = Literal[
    "coverage",
    "medical_necessity",
    "prior_authorization",
    "billing",
    "coding",
    "contract",
    "exclusion",
    "frequency_limit",
    "age_limit",
]


class RuleCondition(BaseModel):
    """Pydantic schema for one rule condition."""

    field: str = ""
    operator: str = ""
    value: Any = ""
    unit: str = ""
    logic: str = "AND"
    description: str = ""

    @field_validator("logic")
    @classmethod
    def validate_logic(cls, value: str) -> str:
        """Validate condition logic.

        Args:
            value: Candidate logical connector.

        Returns:
            Uppercase logical connector.

        Raises:
            ValueError: If the value is not AND or OR.
        """
        normalized = value.upper() if isinstance(value, str) else value
        if normalized not in {"AND", "OR"}:
            raise ValueError("logic must be AND or OR.")
        return normalized


class BusinessRule(BaseModel):
    """Pydantic schema for one healthcare business rule."""

    rule_id: str
    rule_type: RuleType
    document_type: str = ""
    service: str = ""
    condition_logic: str = "AND"
    conditions: list[RuleCondition] = Field(default_factory=list)
    decision: str = ""
    action: str = ""
    exceptions: list[str] = Field(default_factory=list)
    required_documentation: list[str] = Field(default_factory=list)
    source_text: str = ""
    source_document: str = ""
    page_number: int | None = None
    confidence: float = 0.0

    @field_validator("condition_logic")
    @classmethod
    def validate_condition_logic(cls, value: str) -> str:
        """Validate top-level condition logic.

        Args:
            value: Candidate logical connector.

        Returns:
            Uppercase logical connector.

        Raises:
            ValueError: If the value is not AND or OR.
        """
        normalized = value.upper() if isinstance(value, str) else value
        if normalized not in {"AND", "OR"}:
            raise ValueError("condition_logic must be AND or OR.")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Validate confidence score.

        Args:
            value: Candidate confidence value.

        Returns:
            Confidence score between 0 and 1.

        Raises:
            ValueError: If confidence is outside the valid range.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        return value


class HealthcareRuleExtractor:
    """Extracts structured healthcare business rules from policy content."""

    def __init__(
        self,
        llm: Any | None = None,
        llm_model: str = "gpt-4.1",
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        max_retries: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the rule extractor.

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
        """Initialize an OpenAI chat model for rule extraction.

        Args:
            None.

        Returns:
            LangChain-compatible chat model when available; otherwise ``None``.

        Raises:
            This method logs failures and returns ``None`` instead of raising.
        """
        if self.llm is not None:
            return self.llm

        try:
            openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
            groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

            if openai_api_key:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    api_key=openai_api_key,
                )
                self.logger.info("Rule extraction LLM initialized with OpenAI: %s", self.llm_model)
                return self.llm

            if groq_api_key:
                from langchain_groq import ChatGroq

                groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
                self.llm = ChatGroq(
                    model=groq_model,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    api_key=groq_api_key,
                )
                self.logger.info("Rule extraction LLM initialized with Groq: %s", groq_model)
                return self.llm

            self.logger.error("Neither OPENAI_API_KEY nor GROQ_API_KEY is configured.")
            return None
        except Exception as error:
            self.logger.error("Failed to initialize rule extraction LLM: %s", error)
            return None

    def extract_rules_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract structured rules from raw healthcare policy text.

        Args:
            text: Raw healthcare policy text.

        Returns:
            List of structured rule dictionaries.

        Raises:
            This method logs invalid input, API failures, and JSON failures
            instead of raising.
        """
        start_time = time.perf_counter()
        if not isinstance(text, str) or not text.strip():
            self.logger.warning("Empty text provided for rule extraction.")
            self._last_processing_time_seconds = 0.0
            return []

        self.logger.info("Rule extraction started from raw text.")
        llm = self.initialize_llm()
        if llm is None:
            rules = self._extract_rules_deterministically(text)
            self._last_processing_time_seconds = time.perf_counter() - start_time
            self.logger.info("Number of rules extracted: %s", len(rules))
            return rules

        raw_rules = self._call_llm_for_rules(text, llm)
        rules = self._normalize_rules(raw_rules, source_text=text)
        self._last_processing_time_seconds = time.perf_counter() - start_time
        self.logger.info(
            "Number of rules extracted: %s in %.2f seconds.",
            len(rules),
            self._last_processing_time_seconds,
        )
        return rules

    def extract_rules_from_documents(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Extract rules from LangChain documents.

        Args:
            documents: LangChain ``Document`` objects from upstream modules.

        Returns:
            List of structured rule dictionaries with source metadata.
        """
        if not isinstance(documents, list):
            self.logger.error("Documents input must be a list.")
            return []

        extracted_rules: list[dict[str, Any]] = []
        for document in documents:
            if not isinstance(document, Document):
                self.logger.warning("Invalid document skipped: %s", type(document).__name__)
                continue
            if not document.page_content or not document.page_content.strip():
                self.logger.warning("Empty document skipped during rule extraction.")
                continue

            document_rules = self.extract_rules_from_text(document.page_content)
            for rule in document_rules:
                metadata = document.metadata or {}
                rule["source_document"] = str(
                    metadata.get("filename") or metadata.get("source") or ""
                )
                rule["page_number"] = self._coerce_page_number(
                    metadata.get("page_number", metadata.get("page"))
                )
                rule["document_type"] = str(
                    metadata.get("document_type") or rule.get("document_type", "")
                )
            extracted_rules.extend(document_rules)

        normalized_rules = self._post_process_rules(extracted_rules)
        normalized_rules = self._assign_rule_ids(normalized_rules, force=True)
        self.logger.info("Number of document rules extracted: %s", len(normalized_rules))
        return normalized_rules

    def extract_rules_from_summary(
        self,
        summary: dict[str, Any] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract rules from a structured summarizer output.

        Args:
            summary: Structured summary dictionary or summarizer result list.

        Returns:
            List of structured rule dictionaries derived from summary fields.
        """
        payload = self._extract_summary_payload(summary)
        if not payload:
            self.logger.warning("Missing summary provided for rule extraction.")
            return []

        rules: list[dict[str, Any]] = []
        rules.extend(
            self._rules_from_list_field(
                payload,
                "covered_services",
                "coverage",
                "approve",
                "Covered service from summary.",
            )
        )
        rules.extend(
            self._rules_from_list_field(
                payload,
                "excluded_services",
                "exclusion",
                "deny",
                "Excluded service from summary.",
            )
        )
        rules.extend(
            self._rules_from_list_field(
                payload,
                "eligibility_criteria",
                "coverage",
                "review",
                "Eligibility criterion from summary.",
            )
        )
        rules.extend(
            self._rules_from_list_field(
                payload,
                "medical_necessity",
                "medical_necessity",
                "review",
                "Medical necessity criterion from summary.",
            )
        )
        rules.extend(
            self._rules_from_list_field(
                payload,
                "coding_requirements",
                "coding",
                "review",
                "Coding requirement from summary.",
            )
        )

        prior_authorization = str(payload.get("prior_authorization", "")).strip()
        if prior_authorization:
            rules.append(
                self._build_rule(
                    rule_type="prior_authorization",
                    service="",
                    source_text=prior_authorization,
                    decision="review",
                    action="Require prior authorization review.",
                    conditions=[
                        {
                            "field": "prior_authorization",
                            "operator": "==",
                            "value": "required" in prior_authorization.lower(),
                            "unit": "",
                            "logic": "AND",
                            "description": prior_authorization,
                        }
                    ],
                    confidence=0.85,
                )
            )

        normalized_rules = self._post_process_rules(rules)
        normalized_rules = self._assign_rule_ids(normalized_rules, force=True)
        self.logger.info("Number of summary rules extracted: %s", len(normalized_rules))
        return normalized_rules

    def validate_rule_schema(self, rule: dict[str, Any]) -> dict[str, Any]:
        """Validate one extracted rule against the Pydantic schema.

        Args:
            rule: Candidate rule dictionary.

        Returns:
            Validation result with ``valid`` and ``errors`` fields.
        """
        try:
            BusinessRule.model_validate(rule)
            return {"valid": True, "errors": []}
        except ValidationError as error:
            errors = [item["msg"] for item in error.errors()]
            self.logger.warning("Invalid rule schema: %s", errors)
            return {"valid": False, "errors": errors}

    def validate_rules(self, rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Validate a collection of extracted rules.

        Args:
            rules: Rule dictionaries to validate.

        Returns:
            Dictionary containing ``valid_rules`` and ``invalid_rules`` lists.
        """
        valid_rules: list[dict[str, Any]] = []
        invalid_rules: list[dict[str, Any]] = []

        if not isinstance(rules, list):
            self.logger.error("Rules input must be a list.")
            return {"valid_rules": [], "invalid_rules": []}

        for rule in rules:
            validation = self.validate_rule_schema(rule)
            if validation["valid"]:
                valid_rules.append(rule)
            else:
                invalid_rules.append({"rule": rule, "errors": validation["errors"]})

        self.logger.info(
            "Validation results: %s valid, %s invalid.",
            len(valid_rules),
            len(invalid_rules),
        )
        return {"valid_rules": valid_rules, "invalid_rules": invalid_rules}

    def save_rules(self, rules: list[dict[str, Any]], output_path: str | Path) -> bool:
        """Save rules to a JSON file.

        Args:
            rules: Rule dictionaries to save.
            output_path: Destination JSON path.

        Returns:
            ``True`` when saved successfully; otherwise ``False``.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
            self.logger.info("Rules saved: %s", path)
            return True
        except Exception as error:
            self.logger.error("Failed to save rules to %s: %s", path, error)
            return False

    def load_rules(self, input_path: str | Path) -> list[dict[str, Any]]:
        """Load rules from a JSON file.

        Args:
            input_path: Source JSON path.

        Returns:
            Loaded rule dictionaries, or an empty list on failure.
        """
        path = Path(input_path)
        if not path.exists():
            self.logger.error("Rules file does not exist: %s", path)
            return []

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                self.logger.error("Rules file does not contain a list: %s", path)
                return []
            return loaded
        except (json.JSONDecodeError, OSError) as error:
            self.logger.error("Failed to load rules from %s: %s", path, error)
            return []

    def get_rule_statistics(self, rules: list[dict[str, Any]]) -> dict[str, int | float]:
        """Calculate rule extraction statistics.

        Args:
            rules: Rule dictionaries.

        Returns:
            Statistics by rule type plus average confidence.
        """
        if not isinstance(rules, list) or not rules:
            return {
                "total_rules": 0,
                "coverage_rules": 0,
                "medical_necessity_rules": 0,
                "prior_authorization_rules": 0,
                "billing_rules": 0,
                "coding_rules": 0,
                "contract_rules": 0,
                "average_confidence": 0.0,
            }

        def count(rule_type: str) -> int:
            return sum(1 for rule in rules if rule.get("rule_type") == rule_type)

        confidences = [
            float(rule.get("confidence", 0.0))
            for rule in rules
            if isinstance(rule.get("confidence", 0.0), (int, float))
        ]
        average_confidence = (
            round(sum(confidences) / len(confidences), 2) if confidences else 0.0
        )
        return {
            "total_rules": len(rules),
            "coverage_rules": count("coverage"),
            "medical_necessity_rules": count("medical_necessity"),
            "prior_authorization_rules": count("prior_authorization"),
            "billing_rules": count("billing"),
            "coding_rules": count("coding"),
            "contract_rules": count("contract"),
            "average_confidence": average_confidence,
        }

    def _call_llm_for_rules(self, text: str, llm: Any) -> list[dict[str, Any]]:
        """Call the LLM to extract structured rules.

        Args:
            text: Source healthcare policy text.
            llm: LangChain-compatible chat model.

        Returns:
            Raw rule dictionaries returned by the LLM, or an empty list.
        """
        messages = self._build_messages(text)
        for attempt in range(1, self.max_retries + 2):
            try:
                self.logger.info("Submitting rule extraction API request. Attempt: %s", attempt)
                response = llm.invoke(messages)
                content = self._extract_response_content(response)
                return self._parse_rules_json(content)
            except (json.JSONDecodeError, RuntimeError, TimeoutError, ValueError) as error:
                self.logger.error("Rule extraction API attempt failed: %s", error)
        return []

    def _build_messages(self, text: str) -> list[SystemMessage | HumanMessage]:
        """Build LLM prompt messages for rule extraction.

        Args:
            text: Healthcare policy text.

        Returns:
            LangChain system and human messages.
        """
        system_prompt = (
            "You are an expert healthcare policy analyst and rule extraction engine. "
            "Your task is to convert healthcare policy text into structured "
            "machine-readable business rules. Extract only rules that are "
            "explicitly supported by the text. Do not invent rules. Do not infer "
            "unsupported medical requirements. Preserve exceptions and conditions. "
            "Return valid JSON only."
        )
        human_prompt = (
            "Extract healthcare business rules from the text. Include services, "
            "conditions, thresholds, diagnosis requirements, CPT codes, ICD codes, "
            "age requirements, therapy duration, prior authorization requirements, "
            "documentation requirements, exclusions, exceptions, and contract terms. "
            "Return a JSON array of rule objects using this schema:\n"
            "{\n"
            '  "rule_id": "", "rule_type": "", "document_type": "", '
            '"service": "", "condition_logic": "AND", "conditions": ['
            '{"field": "", "operator": "", "value": "", "unit": "", '
            '"logic": "AND", "description": ""}], "decision": "", '
            '"action": "", "exceptions": [], "required_documentation": [], '
            '"source_text": "", "source_document": "", "page_number": null, '
            '"confidence": 0.0\n'
            "}\n\n"
            f"Healthcare policy text:\n{text}"
        )
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from an LLM response.

        Args:
            response: LangChain response or response-like object.

        Returns:
            Text response content.

        Raises:
            ValueError: If no usable response content exists.
        """
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response did not contain text content.")
        return content.strip()

    def _parse_rules_json(self, content: str) -> list[dict[str, Any]]:
        """Parse LLM JSON output into rule dictionaries.

        Args:
            content: Raw LLM response content.

        Returns:
            Parsed rule dictionaries.

        Raises:
            json.JSONDecodeError: If JSON parsing fails.
            ValueError: If parsed JSON shape is unsupported.
        """
        parsed = json.loads(self._strip_code_fences(content))
        if isinstance(parsed, dict) and isinstance(parsed.get("rules"), list):
            parsed = parsed["rules"]
        if not isinstance(parsed, list):
            raise ValueError("Rule extraction response must be a JSON array.")
        return [item for item in parsed if isinstance(item, dict)]

    def _strip_code_fences(self, content: str) -> str:
        """Remove Markdown code fences from model output.

        Args:
            content: Raw model response.

        Returns:
            Content without wrapping Markdown code fences.
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

    def _normalize_rules(
        self,
        rules: list[dict[str, Any]],
        source_text: str = "",
    ) -> list[dict[str, Any]]:
        """Normalize raw rule dictionaries to the required schema.

        Args:
            rules: Raw rule dictionaries.
            source_text: Fallback source text.

        Returns:
            Normalized rule dictionaries.
        """
        normalized_rules = [
            self._build_rule(
                rule_type=str(rule.get("rule_type", "coverage")),
                document_type=str(rule.get("document_type", "")),
                service=str(rule.get("service", "")),
                condition_logic=str(rule.get("condition_logic", "AND")),
                conditions=rule.get("conditions", []),
                decision=str(rule.get("decision", "")),
                action=str(rule.get("action", "")),
                exceptions=self._coerce_string_list(rule.get("exceptions", [])),
                required_documentation=self._coerce_string_list(
                    rule.get("required_documentation", [])
                ),
                source_text=str(rule.get("source_text") or source_text),
                source_document=str(rule.get("source_document", "")),
                page_number=self._coerce_page_number(rule.get("page_number")),
                confidence=self._coerce_confidence(rule.get("confidence", 0.0)),
            )
            for rule in rules
        ]
        return self._assign_rule_ids(self._post_process_rules(normalized_rules), force=True)

    def _build_rule(
        self,
        rule_type: str,
        service: str,
        source_text: str,
        document_type: str = "",
        condition_logic: str = "AND",
        conditions: list[dict[str, Any]] | None = None,
        decision: str = "",
        action: str = "",
        exceptions: list[str] | None = None,
        required_documentation: list[str] | None = None,
        source_document: str = "",
        page_number: int | None = None,
        confidence: float = 0.8,
    ) -> dict[str, Any]:
        """Build one rule dictionary using the required schema.

        Args:
            rule_type: Rule category.
            service: Healthcare service.
            source_text: Supporting source text.
            document_type: Source document type.
            condition_logic: Top-level condition logic.
            conditions: Rule conditions.
            decision: Rule decision.
            action: Rule action.
            exceptions: Rule exceptions.
            required_documentation: Required documentation list.
            source_document: Source document name.
            page_number: Source page number.
            confidence: Rule extraction confidence.

        Returns:
            Rule dictionary.
        """
        return {
            "rule_id": "",
            "rule_type": self._normalize_rule_type(rule_type),
            "document_type": document_type,
            "service": service.strip(),
            "condition_logic": condition_logic.upper() if condition_logic else "AND",
            "conditions": self._normalize_conditions(conditions or []),
            "decision": decision.strip(),
            "action": action.strip(),
            "exceptions": exceptions or [],
            "required_documentation": required_documentation or [],
            "source_text": source_text,
            "source_document": source_document,
            "page_number": page_number,
            "confidence": confidence,
        }

    def _extract_rules_deterministically(self, text: str) -> list[dict[str, Any]]:
        """Extract obvious rules without an LLM fallback.

        Args:
            text: Healthcare policy text.

        Returns:
            Heuristic rule dictionaries.
        """
        rules: list[dict[str, Any]] = []
        service = self._extract_service(text)
        duration = self._extract_duration(text)

        if "covered" in text.lower():
            conditions: list[dict[str, Any]] = []
            if duration:
                value, unit = duration
                conditions.append(
                    {
                        "field": "therapy_duration",
                        "operator": ">=",
                        "value": value,
                        "unit": unit,
                        "logic": "AND",
                        "description": "Therapy duration requirement.",
                    }
                )
            rules.append(
                self._build_rule(
                    rule_type="coverage",
                    service=service,
                    source_text=text,
                    decision="approve",
                    action="Approve when conditions are met.",
                    conditions=conditions,
                    confidence=0.7,
                )
            )

        if "prior authorization" in text.lower():
            rules.append(
                self._build_rule(
                    rule_type="prior_authorization",
                    service=service,
                    source_text=text,
                    decision="review",
                    action="Require prior authorization before service.",
                    conditions=[
                        {
                            "field": "prior_authorization",
                            "operator": "==",
                            "value": "required" in text.lower(),
                            "unit": "",
                            "logic": "AND",
                            "description": "Prior authorization requirement.",
                        }
                    ],
                    confidence=0.75,
                )
            )

        return self._assign_rule_ids(self._post_process_rules(rules), force=True)

    def _extract_service(self, text: str) -> str:
        """Extract a simple service phrase from text.

        Args:
            text: Source policy text.

        Returns:
            Service phrase when detected.
        """
        patterns = [
            r"(lumbar(?:\s+spine)?\s+mri)",
            r"(cervical(?:\s+spine)?\s+mri)",
            r"(thoracic(?:\s+spine)?\s+mri)",
            r"(magnetic resonance imaging)",
            r"(computed tomography|ct scan)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._title_case_service(match.group(1))

        match = re.search(r"([A-Z][A-Za-z ]+(?:MRI|CT))", text)
        return self._title_case_service(match.group(1)) if match else ""

    def _extract_duration(self, text: str) -> tuple[int, str] | None:
        """Extract simple duration requirements from text.

        Args:
            text: Source policy text.

        Returns:
            Duration value and unit, or ``None``.
        """
        word_numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        match = re.search(
            r"\b(\d+|one|two|three|four|five|six)\s+(week|weeks|month|months|day|days)\b",
            text.lower(),
        )
        if not match:
            return None
        raw_value = match.group(1)
        value = int(raw_value) if raw_value.isdigit() else word_numbers[raw_value]
        unit = match.group(2)
        if unit.endswith("s"):
            unit = unit[:-1]
        return value, unit

    def _rules_from_list_field(
        self,
        payload: dict[str, Any],
        field_name: str,
        rule_type: str,
        decision: str,
        description: str,
    ) -> list[dict[str, Any]]:
        """Create rules from a list-valued summary field.

        Args:
            payload: Summary payload.
            field_name: Summary field name.
            rule_type: Rule type to assign.
            decision: Rule decision.
            description: Condition description.

        Returns:
            Rule dictionaries.
        """
        values = payload.get(field_name, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []

        rules: list[dict[str, Any]] = []
        for value in values:
            text_value = str(value).strip()
            if not text_value:
                continue
            rules.append(
                self._build_rule(
                    rule_type=rule_type,
                    service=text_value if rule_type in {"coverage", "exclusion"} else "",
                    source_text=text_value,
                    document_type=str(payload.get("document_type", "")),
                    decision=decision,
                    action=description,
                    conditions=[
                        {
                            "field": field_name,
                            "operator": "contains",
                            "value": text_value,
                            "unit": "",
                            "logic": "AND",
                            "description": description,
                        }
                    ],
                    confidence=0.8,
                )
            )
        return rules

    def _extract_summary_payload(
        self,
        summary: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract a summary payload from supported summary shapes.

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

    def _normalize_conditions(self, conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize rule conditions.

        Args:
            conditions: Raw condition dictionaries.

        Returns:
            Normalized condition dictionaries.
        """
        normalized: list[dict[str, Any]] = []
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            normalized.append(
                {
                    "field": str(condition.get("field", "")),
                    "operator": str(condition.get("operator", "")),
                    "value": condition.get("value", ""),
                    "unit": str(condition.get("unit", "")),
                    "logic": str(condition.get("logic", "AND")).upper(),
                    "description": str(condition.get("description", "")),
                }
            )
        return normalized

    def _normalize_rule_type(self, rule_type: str) -> str:
        """Normalize a rule type to supported values.

        Args:
            rule_type: Raw rule type.

        Returns:
            Supported rule type.
        """
        normalized = rule_type.lower().strip().replace("-", "_").replace(" ", "_")
        alias_map = {
            "prior_authoriz": "prior_authorization",
            "prior_auth": "prior_authorization",
            "medicalnecessity": "medical_necessity",
            "medical_necessity_rule": "medical_necessity",
            "frequency": "frequency_limit",
            "age": "age_limit",
        }
        normalized = alias_map.get(normalized, normalized)
        supported = {
            "coverage",
            "medical_necessity",
            "prior_authorization",
            "billing",
            "coding",
            "contract",
            "exclusion",
            "frequency_limit",
            "age_limit",
        }
        return normalized if normalized in supported else "coverage"

    def _infer_rule_type(self, rule_type: str, source_text: str) -> str:
        """Infer a stronger rule type from source text when labels are weak."""
        normalized = self._normalize_rule_type(rule_type)
        lowered = source_text.lower()

        if any(token in lowered for token in ["prior authorization", "preauthorization"]):
            return "prior_authorization"
        if any(token in lowered for token in ["cpt", "hcpcs", "icd", "procedure code", "diagnosis code"]):
            return "coding"
        if any(
            token in lowered
            for token in ["claim must include", "documentation", "rendering provider", "ordering provider"]
        ):
            return "billing" if normalized == "coverage" else normalized
        if any(token in lowered for token in ["not covered", "excluded", "is not covered"]):
            return "exclusion"
        if any(token in lowered for token in ["once per", "frequency", "repeat imaging"]):
            return "frequency_limit"
        if any(token in lowered for token in ["medically necessary", "medical necessity"]):
            return "medical_necessity"
        return normalized

    def _assign_rule_ids(
        self,
        rules: list[dict[str, Any]],
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Assign stable sequential rule IDs where missing.

        Args:
            rules: Rule dictionaries.
            force: Whether to overwrite existing IDs.

        Returns:
            Rule dictionaries with IDs.
        """
        for index, rule in enumerate(rules, start=1):
            if force or not rule.get("rule_id"):
                rule["rule_id"] = f"RULE_{index:03d}"
        return rules

    def _post_process_rules(self, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize, deduplicate, and harden extracted rules for presentation.

        Args:
            rules: Raw or partially normalized rule dictionaries.

        Returns:
            Cleaned rule dictionaries with stronger consistency.
        """
        processed: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rule in rules:
            if not isinstance(rule, dict):
                continue

            normalized_rule = dict(rule)
            normalized_rule["source_text"] = str(normalized_rule.get("source_text", "")).strip()
            normalized_rule["rule_type"] = self._infer_rule_type(
                str(normalized_rule.get("rule_type", "coverage")),
                normalized_rule["source_text"],
            )
            normalized_rule["service"] = self._normalize_service(
                str(normalized_rule.get("service", "")),
                normalized_rule["source_text"],
                normalized_rule["rule_type"],
            )
            normalized_rule["decision"] = self._normalize_decision(
                str(normalized_rule.get("decision", "")),
                normalized_rule["rule_type"],
                normalized_rule["source_text"],
            )
            normalized_rule["conditions"] = self._normalize_conditions(
                normalized_rule.get("conditions", [])
            )
            normalized_rule["conditions"] = self._ensure_conditions(normalized_rule)
            normalized_rule["action"] = self._normalize_action(
                str(normalized_rule.get("action", "")),
                normalized_rule["decision"],
                normalized_rule["rule_type"],
            )

            dedupe_key = json.dumps(
                {
                    "rule_type": normalized_rule.get("rule_type", ""),
                    "service": normalized_rule.get("service", ""),
                    "decision": normalized_rule.get("decision", ""),
                    "conditions": normalized_rule.get("conditions", []),
                    "source_text": normalized_rule.get("source_text", ""),
                    "page_number": normalized_rule.get("page_number"),
                },
                sort_keys=True,
                default=str,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            processed.append(normalized_rule)
        return processed

    def _normalize_service(self, service: str, source_text: str, rule_type: str) -> str:
        """Normalize service labels so service and requirement text stay separate."""
        service_text = service.strip()
        source = source_text.strip()

        extracted_service = self._extract_service(service_text) or self._extract_service(source)
        if extracted_service:
            return extracted_service

        lowered_service = service_text.lower()
        lowered_source = source.lower()
        requirement_markers = [
            "claims must include",
            "must include",
            "documentation",
            "prior authorization",
            "duration and type",
            "conservative therapy",
            "history and physical",
        ]
        if any(marker in lowered_service for marker in requirement_markers):
            if "mri" in lowered_source:
                extracted = self._extract_service(source)
                if extracted:
                    return extracted
            if rule_type in {"coding", "billing"}:
                return "Coding or Billing Requirement"
            if rule_type == "prior_authorization":
                return "Prior Authorization Requirement"
            if rule_type == "medical_necessity":
                return "Medical Necessity Requirement"
            return "Policy Requirement"

        return service_text or "Policy Requirement"

    def _normalize_decision(self, decision: str, rule_type: str, source_text: str) -> str:
        """Normalize generic decisions into explicit rule outcomes."""
        lowered_text = source_text.lower()
        explicit = decision.strip().lower().replace(" ", "_")
        if explicit in {
            "approve_if_criteria_met",
            "require_prior_authorization",
            "require_coding_compliance",
            "require_documentation_review",
            "deny_not_covered",
            "limit_repeat_service",
            "require_medical_necessity_review",
        }:
            return explicit

        if rule_type == "coverage":
            return "approve_if_criteria_met"
        if rule_type == "prior_authorization":
            return "require_prior_authorization"
        if rule_type in {"coding", "billing", "contract"}:
            return "require_coding_compliance"
        if rule_type == "medical_necessity":
            return "require_medical_necessity_review"
        if rule_type == "exclusion" or "not covered" in lowered_text or "excluded" in lowered_text:
            return "deny_not_covered"
        if rule_type == "frequency_limit":
            return "limit_repeat_service"
        return explicit or "require_review"

    def _normalize_action(self, action: str, decision: str, rule_type: str) -> str:
        """Fill or normalize action text from decision semantics."""
        if action.strip():
            return action.strip()

        action_map = {
            "approve_if_criteria_met": "Approve when all policy criteria are satisfied.",
            "require_prior_authorization": "Require prior authorization before approval.",
            "require_coding_compliance": "Require coding and documentation compliance review.",
            "require_documentation_review": "Require supporting documentation review.",
            "deny_not_covered": "Deny when the policy exclusion applies.",
            "limit_repeat_service": "Review repeat service against policy frequency limits.",
            "require_medical_necessity_review": "Require medical necessity validation.",
            "require_review": f"Review the {rule_type.replace('_', ' ')} requirement.",
        }
        return action_map.get(decision, "Review policy requirement.")

    def _ensure_conditions(self, rule: dict[str, Any]) -> list[dict[str, Any]]:
        """Ensure every rule has at least one executable condition."""
        conditions = rule.get("conditions", [])
        if conditions:
            return conditions

        source_text = str(rule.get("source_text", "")).strip()
        rule_type = str(rule.get("rule_type", "coverage"))
        fallback_field = {
            "coverage": "coverage_criteria",
            "prior_authorization": "prior_authorization_requirement",
            "coding": "coding_requirement",
            "billing": "billing_requirement",
            "medical_necessity": "medical_necessity_requirement",
            "exclusion": "exclusion_clause",
            "contract": "contract_requirement",
            "frequency_limit": "frequency_limit",
            "age_limit": "age_requirement",
        }.get(rule_type, "policy_requirement")

        return [
            {
                "field": fallback_field,
                "operator": "exists",
                "value": True,
                "unit": "",
                "logic": "AND",
                "description": source_text or f"{rule_type.replace('_', ' ')} requirement extracted from policy text.",
            }
        ]

    def _title_case_service(self, service: str) -> str:
        """Return a clean title-cased service label."""
        cleaned = " ".join(service.strip().split())
        replacements = {
            "Mri": "MRI",
            "Ct": "CT",
        }
        title_cased = cleaned.title()
        for source, target in replacements.items():
            title_cased = title_cased.replace(source, target)
        return title_cased

    def _coerce_string_list(self, value: Any) -> list[str]:
        """Coerce a value into a string list.

        Args:
            value: Candidate list or scalar value.

        Returns:
            List of strings.
        """
        if isinstance(value, list):
            return [str(item) for item in value]
        if value in (None, ""):
            return []
        return [str(value)]

    def _coerce_page_number(self, value: Any) -> int | None:
        """Coerce page metadata into an integer.

        Args:
            value: Candidate page number.

        Returns:
            Integer page number or ``None``.
        """
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_confidence(self, value: Any) -> float:
        """Coerce confidence to a valid score.

        Args:
            value: Candidate confidence value.

        Returns:
            Float confidence between 0 and 1.
        """
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        return min(1.0, max(0.0, confidence))

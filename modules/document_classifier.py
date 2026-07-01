"""Healthcare document classification.

This module classifies healthcare content into workflow categories using
deterministic keyword/regex scoring and an optional LLM classifier. It does not
load PDFs, summarize documents, run RAG, extract rules, or render UI.

Example:
    ```python
    from modules.pdf_loader import PDFLoader
    from modules.document_classifier import HealthcareDocumentClassifier

    loader = PDFLoader()
    documents = loader.load_pdf("data/sample_policy.pdf")

    classifier = HealthcareDocumentClassifier()
    classification = classifier.classify_documents(documents)

    print(classification)
    ```
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator


DOCUMENT_TYPES = (
    "billing_coding_policy",
    "clinical_practice_guideline",
    "payer_provider_contract",
    "coverage_policy",
    "regulatory_document",
    "unknown",
)


class ClassificationResult(BaseModel):
    """Pydantic schema for document classification results."""

    document_type: str
    confidence: float = 0.0
    method: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)
    reason: str = ""
    source_document: str = ""
    page_number: int | None = None

    @field_validator("document_type")
    @classmethod
    def validate_document_type(cls, value: str) -> str:
        """Validate document type.

        Args:
            value: Candidate document type.

        Returns:
            Valid document type.

        Raises:
            ValueError: If document type is unsupported.
        """
        if value not in DOCUMENT_TYPES:
            raise ValueError("Unsupported document_type.")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Validate confidence score.

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


class HealthcareDocumentClassifier:
    """Classifies healthcare documents into downstream workflow categories."""

    def __init__(
        self,
        llm: Any | None = None,
        llm_model: str = "gpt-4.1",
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the healthcare document classifier.

        Args:
            llm: Optional LangChain-compatible chat model for LLM classification.
            llm_model: OpenAI model name used when initializing an optional LLM.
            temperature: LLM temperature.
            request_timeout: LLM request timeout in seconds.
            logger: Optional logger instance.

        Returns:
            None.
        """
        self.llm = llm
        self.llm_model = llm_model
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.logger = logger or logging.getLogger(__name__)

    def classify_text(self, text: str) -> dict[str, Any]:
        """Classify raw healthcare document text.

        Args:
            text: Raw healthcare document text.

        Returns:
            Classification result dictionary.
        """
        self.logger.info("Classification started.")
        if not isinstance(text, str) or not text.strip():
            self.logger.warning("Empty text provided for classification.")
            return self._unknown_result("Empty or invalid text.")

        keyword_scores = self.calculate_keyword_scores(text)
        signals = self.detect_document_signals(text)
        keyword_result = self._classification_from_scores(keyword_scores, signals)
        llm_result = self.llm_classify(text)
        if llm_result and self.validate_classification(llm_result)["valid"]:
            final_result = self._merge_llm_and_keyword_results(keyword_result, llm_result)
        else:
            final_result = keyword_result

        self.logger.info("Final classification: %s", final_result["document_type"])
        return final_result

    def classify_document(self, document: Document) -> dict[str, Any]:
        """Classify one LangChain Document.

        Args:
            document: LangChain document.

        Returns:
            Classification result with source metadata preserved.
        """
        if not isinstance(document, Document):
            return self._unknown_result("Unsupported document input.")

        result = self.classify_text(document.page_content or "")
        metadata = document.metadata or {}
        result["source_document"] = str(metadata.get("filename") or metadata.get("source") or "")
        result["page_number"] = self._coerce_page_number(metadata.get("page_number", metadata.get("page")))
        result["original_metadata"] = metadata
        return result

    def classify_documents(self, documents: list[Document]) -> dict[str, Any]:
        """Classify multiple document pages and aggregate the result.

        Args:
            documents: LangChain document pages.

        Returns:
            Aggregate document classification with page classifications.
        """
        if not isinstance(documents, list) or not documents:
            return {
                **self._unknown_result("Missing documents."),
                "page_classifications": [],
            }

        page_classifications = [
            self.classify_document(document)
            for document in documents
            if isinstance(document, Document)
        ]
        if not page_classifications:
            return {
                **self._unknown_result("No valid documents."),
                "page_classifications": [],
            }

        aggregate_type, aggregate_confidence = self._aggregate_page_classifications(page_classifications)
        signals = self._unique_signals(page_classifications)
        return {
            "document_type": aggregate_type,
            "confidence": aggregate_confidence,
            "method": "aggregate_keyword_and_llm",
            "page_classifications": page_classifications,
            "signals": signals,
            "reason": self._reason_for_classification(aggregate_type, signals),
        }

    def classify_from_summary(self, summary: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """Classify a document from structured summary fields.

        Args:
            summary: Summary dictionary or summarizer output list.

        Returns:
            Classification result.
        """
        payload = self._extract_summary_payload(summary)
        if not payload:
            return self._unknown_result("Missing summary.")
        text = self._summary_to_text(payload)
        return self.classify_text(text)

    def llm_classify(self, text: str) -> dict[str, Any] | None:
        """Optionally classify text with an LLM.

        Args:
            text: Healthcare document text.

        Returns:
            LLM classification result, or ``None`` when unavailable or failed.
        """
        llm = self._initialize_llm()
        if llm is None:
            return None

        messages = self._build_llm_messages(text)
        try:
            self.logger.info("LLM classification attempted.")
            response = llm.invoke(messages)
            content = self._extract_response_content(response)
            parsed = json.loads(self._strip_code_fences(content))
            if isinstance(parsed, dict):
                result = self._normalize_classification(parsed, method="keyword_and_llm")
                return result
        except (json.JSONDecodeError, RuntimeError, TimeoutError, ValueError) as error:
            self.logger.warning("LLM fallback used after classification failure: %s", error)
        return None

    def calculate_keyword_scores(self, text: str) -> dict[str, float]:
        """Calculate keyword classification scores.

        Args:
            text: Healthcare document text.

        Returns:
            Scores for every supported document type.
        """
        if not isinstance(text, str) or not text.strip():
            return self._empty_scores()

        lowered = text.lower()
        raw_scores: dict[str, float] = {}
        for document_type, keywords in self._keyword_map().items():
            score = 0.0
            for keyword in keywords:
                score += lowered.count(keyword.lower())
            raw_scores[document_type] = score

        signal_bonus = self._signal_score_bonus(text)
        for document_type, bonus in signal_bonus.items():
            raw_scores[document_type] += bonus

        max_score = max(raw_scores.values()) if raw_scores else 0.0
        scores = self._empty_scores()
        if max_score <= 0:
            scores["unknown"] = 1.0
        else:
            for document_type, score in raw_scores.items():
                scores[document_type] = round(min(score / max_score, 1.0), 3)
            scores["unknown"] = 0.0 if max_score >= 2 else 0.4
        self.logger.info("Keyword scores calculated.")
        return scores

    def detect_document_signals(self, text: str) -> list[str]:
        """Detect useful regex and phrase signals.

        Args:
            text: Healthcare document text.

        Returns:
            Detected signal descriptions.
        """
        if not isinstance(text, str):
            return []

        signals: list[str] = []
        patterns = [
            (r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b", "ICD code present"),
            (r"\b\d{5}\b", "CPT code present"),
            (r"\b[A-Z]\d{4}\b", "HCPCS code present"),
            (r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "Dollar amount present"),
            (r"\b\d{1,3}(?:\.\d+)?%\b", "Percentage present"),
            (r"\b(?:42\s+CFR|section\s+1862|HIPAA|CMS)\b", "Regulatory reference present"),
        ]
        for pattern, signal in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                signals.append(signal)

        phrase_signals = [
            ("prior authorization", "Prior authorization mentioned"),
            ("medical necessity", "Medical necessity mentioned"),
            ("not covered", "Coverage/exclusion language present"),
            ("excluded", "Coverage/exclusion language present"),
            ("covered", "Coverage/exclusion language present"),
            ("agreement", "Contract language present"),
            ("contract", "Contract language present"),
            ("fee schedule", "Contract language present"),
            ("compliance", "Regulatory language present"),
            ("regulation", "Regulatory language present"),
        ]
        lowered = text.lower()
        for phrase, signal in phrase_signals:
            if phrase in lowered and signal not in signals:
                signals.append(signal)

        self.logger.info("Signals detected: %s", len(signals))
        return signals

    def validate_classification(self, result: dict[str, Any]) -> dict[str, Any]:
        """Validate a classification result.

        Args:
            result: Candidate classification dictionary.

        Returns:
            Validation result with ``valid`` and ``errors``.
        """
        try:
            ClassificationResult.model_validate(self._normalize_classification(result))
            return {"valid": True, "errors": []}
        except ValidationError as error:
            return {"valid": False, "errors": [item["msg"] for item in error.errors()]}

    def save_classifications(self, results: dict[str, Any] | list[dict[str, Any]], output_path: str | Path) -> bool:
        """Save classifications as JSON.

        Args:
            results: Classification result or list of results.
            output_path: Destination JSON path.

        Returns:
            ``True`` when saved successfully; otherwise ``False``.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            return True
        except (OSError, TypeError) as error:
            self.logger.error("Failed to save classifications to %s: %s", path, error)
            return False

    def load_classifications(self, input_path: str | Path) -> dict[str, Any] | list[dict[str, Any]]:
        """Load classifications from JSON.

        Args:
            input_path: Source JSON path.

        Returns:
            Loaded classification data, or an empty list on failure.
        """
        path = Path(input_path)
        if not path.exists():
            self.logger.error("Classification file does not exist: %s", path)
            return []
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, (dict, list)) else []
        except (json.JSONDecodeError, OSError) as error:
            self.logger.error("Failed to load classifications from %s: %s", path, error)
            return []

    def get_classification_statistics(self, results: list[dict[str, Any]]) -> dict[str, int | float]:
        """Calculate classification statistics.

        Args:
            results: Classification result dictionaries.

        Returns:
            Aggregate classification statistics.
        """
        if not isinstance(results, list) or not results:
            return self._empty_statistics()

        valid_results = [result for result in results if isinstance(result, dict)]
        confidences = [
            float(result.get("confidence", 0.0))
            for result in valid_results
            if isinstance(result.get("confidence", 0.0), (int, float))
        ]
        statistics = self._empty_statistics()
        statistics["total_documents"] = len(valid_results)
        for document_type in DOCUMENT_TYPES:
            statistics[document_type] = sum(
                1 for result in valid_results if result.get("document_type") == document_type
            )
        statistics["average_confidence"] = round(float(np.mean(confidences)), 2) if confidences else 0.0
        return statistics

    def _classification_from_scores(
        self,
        scores: dict[str, float],
        signals: list[str],
    ) -> dict[str, Any]:
        """Build a classification from keyword scores.

        Args:
            scores: Keyword scores.
            signals: Detected document signals.

        Returns:
            Classification dictionary.
        """
        document_type = max(scores, key=scores.get) if scores else "unknown"
        confidence = float(scores.get(document_type, 0.0))
        if document_type == "unknown" or confidence < 0.45:
            document_type = "unknown"
            confidence = max(confidence, 0.25 if signals else 0.0)
        return self._normalize_classification(
            {
                "document_type": document_type,
                "confidence": round(confidence, 2),
                "method": "keyword",
                "scores": scores,
                "signals": signals,
                "reason": self._reason_for_classification(document_type, signals),
            }
        )

    def _merge_llm_and_keyword_results(
        self,
        keyword_result: dict[str, Any],
        llm_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge LLM and keyword classification results.

        Args:
            keyword_result: Deterministic keyword result.
            llm_result: LLM classification result.

        Returns:
            Merged classification result.
        """
        llm_type = llm_result.get("document_type", "unknown")
        keyword_type = keyword_result.get("document_type", "unknown")
        if llm_type == keyword_type or llm_result.get("confidence", 0.0) >= 0.75:
            merged = dict(keyword_result)
            merged.update(
                {
                    "document_type": llm_type,
                    "confidence": round(
                        max(
                            float(keyword_result.get("confidence", 0.0)),
                            float(llm_result.get("confidence", 0.0)),
                        ),
                        2,
                    ),
                    "method": "keyword_and_llm",
                    "signals": self._merge_lists(
                        keyword_result.get("signals", []),
                        llm_result.get("signals", []),
                    ),
                    "reason": llm_result.get("reason") or keyword_result.get("reason", ""),
                }
            )
            return self._normalize_classification(merged)
        return keyword_result

    def _keyword_map(self) -> dict[str, tuple[str, ...]]:
        """Return keyword map by document type.

        Args:
            None.

        Returns:
            Mapping of document types to keywords.
        """
        return {
            "billing_coding_policy": (
                "cpt",
                "icd",
                "hcpcs",
                "claim",
                "billing",
                "coding",
                "reimbursement",
                "modifier",
                "denial",
                "allowable",
            ),
            "clinical_practice_guideline": (
                "guideline",
                "evidence-based",
                "treatment",
                "diagnosis",
                "clinical",
                "patient management",
                "recommended therapy",
                "standard of care",
            ),
            "payer_provider_contract": (
                "agreement",
                "contract",
                "provider",
                "payer",
                "allowed amount",
                "fee schedule",
                "copay",
                "coinsurance",
                "reimbursement rate",
                "payment terms",
            ),
            "coverage_policy": (
                "covered",
                "not covered",
                "excluded",
                "eligibility",
                "prior authorization",
                "medical necessity",
                "benefit",
                "coverage criteria",
            ),
            "regulatory_document": (
                "cms",
                "hipaa",
                "regulation",
                "compliance",
                "federal",
                "state law",
                "cfr",
                "statute",
                "audit requirement",
            ),
        }

    def _signal_score_bonus(self, text: str) -> dict[str, float]:
        """Calculate score bonuses from regex signals.

        Args:
            text: Healthcare document text.

        Returns:
            Document type bonus scores.
        """
        bonuses = {document_type: 0.0 for document_type in DOCUMENT_TYPES if document_type != "unknown"}
        if re.search(r"\b\d{5}\b", text) or re.search(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b", text):
            bonuses["billing_coding_policy"] += 2.0
        if re.search(r"\b[A-Z]\d{4}\b", text):
            bonuses["billing_coding_policy"] += 1.5
        if re.search(r"\$\s?\d", text) or re.search(r"\b\d{1,3}(?:\.\d+)?%\b", text):
            bonuses["payer_provider_contract"] += 2.0
        if re.search(r"\b(?:42\s+CFR|section\s+1862|HIPAA|CMS)\b", text, flags=re.IGNORECASE):
            bonuses["regulatory_document"] += 2.5
        return bonuses

    def _build_llm_messages(self, text: str) -> list[SystemMessage | HumanMessage]:
        """Build optional LLM classification messages.

        Args:
            text: Healthcare document text.

        Returns:
            LangChain messages.
        """
        system_prompt = (
            "You are an expert healthcare document classification engine. "
            "Classify the provided healthcare document text into exactly one category: "
            "billing_coding_policy, clinical_practice_guideline, payer_provider_contract, "
            "coverage_policy, regulatory_document, unknown. Use only the provided text. "
            "Do not invent facts. Return valid JSON only. Include confidence, detected "
            "signals, and a short reason."
        )
        human_prompt = f"Document text:\n{text}"
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    def _initialize_llm(self) -> Any | None:
        """Initialize optional LLM classifier.

        Args:
            None.

        Returns:
            LangChain-compatible chat model, or ``None``.
        """
        if self.llm is not None:
            return self.llm
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.llm_model,
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            return self.llm
        except Exception as error:
            self.logger.warning("LLM classifier unavailable: %s", error)
            return None

    def _extract_response_content(self, response: Any) -> str:
        """Extract response content.

        Args:
            response: LLM response or string.

        Returns:
            Response text.

        Raises:
            ValueError: If response content is missing.
        """
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response did not contain text content.")
        return content.strip()

    def _strip_code_fences(self, content: str) -> str:
        """Strip Markdown code fences.

        Args:
            content: Raw response content.

        Returns:
            Content without wrapping fences.
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

    def _normalize_classification(
        self,
        result: dict[str, Any],
        method: str | None = None,
    ) -> dict[str, Any]:
        """Normalize classification output to required schema.

        Args:
            result: Candidate classification result.
            method: Optional method override.

        Returns:
            Normalized classification dictionary.
        """
        document_type = str(result.get("document_type", "unknown"))
        if document_type not in DOCUMENT_TYPES:
            document_type = "unknown"
        scores = self._empty_scores()
        if isinstance(result.get("scores"), dict):
            for key in scores:
                try:
                    scores[key] = float(result["scores"].get(key, scores[key]))
                except (TypeError, ValueError):
                    scores[key] = 0.0
        return {
            "document_type": document_type,
            "confidence": self._clamp(result.get("confidence", 0.0)),
            "method": method or str(result.get("method", "")),
            "scores": scores,
            "signals": self._coerce_string_list(result.get("signals", [])),
            "reason": str(result.get("reason", "")),
            "source_document": str(result.get("source_document", "")),
            "page_number": self._coerce_page_number(result.get("page_number")),
        }

    def _aggregate_page_classifications(
        self,
        page_classifications: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """Aggregate page-level classifications.

        Args:
            page_classifications: Page classification results.

        Returns:
            Aggregate document type and confidence.
        """
        weighted_scores = {document_type: 0.0 for document_type in DOCUMENT_TYPES}
        for result in page_classifications:
            document_type = result.get("document_type", "unknown")
            confidence = float(result.get("confidence", 0.0))
            weighted_scores[document_type] = weighted_scores.get(document_type, 0.0) + confidence
        aggregate_type = max(weighted_scores, key=weighted_scores.get)
        total_weight = sum(weighted_scores.values()) or 1.0
        aggregate_confidence = round(weighted_scores[aggregate_type] / total_weight, 2)
        return aggregate_type, aggregate_confidence

    def _reason_for_classification(self, document_type: str, signals: list[str]) -> str:
        """Build classification reason.

        Args:
            document_type: Selected document type.
            signals: Detected signals.

        Returns:
            Human-readable reason.
        """
        if document_type == "unknown":
            return "The document did not contain enough reliable healthcare classification signals."
        signal_text = ", ".join(signals[:4]) if signals else "category-specific keywords"
        return f"The document was classified as {document_type} based on {signal_text}."

    def _unique_signals(self, page_classifications: list[dict[str, Any]]) -> list[str]:
        """Collect unique signals from page classifications.

        Args:
            page_classifications: Page classification results.

        Returns:
            Unique signal list.
        """
        signals: list[str] = []
        for result in page_classifications:
            signals = self._merge_lists(signals, result.get("signals", []))
        return signals

    def _extract_summary_payload(self, summary: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """Extract summary payload from supported summary shapes.

        Args:
            summary: Summary dictionary or summarizer output list.

        Returns:
            Summary payload.
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
        """Convert summary fields to text.

        Args:
            payload: Summary payload.

        Returns:
            Text representation.
        """
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"{key}: {'; '.join(str(item) for item in value)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _empty_scores(self) -> dict[str, float]:
        """Return empty score schema.

        Args:
            None.

        Returns:
            Score dictionary.
        """
        return {document_type: 0.0 for document_type in DOCUMENT_TYPES}

    def _empty_statistics(self) -> dict[str, int | float]:
        """Return empty statistics schema.

        Args:
            None.

        Returns:
            Classification statistics dictionary.
        """
        statistics: dict[str, int | float] = {"total_documents": 0}
        for document_type in DOCUMENT_TYPES:
            statistics[document_type] = 0
        statistics["average_confidence"] = 0.0
        return statistics

    def _unknown_result(self, reason: str) -> dict[str, Any]:
        """Build unknown classification result.

        Args:
            reason: Reason for unknown classification.

        Returns:
            Unknown classification result.
        """
        scores = self._empty_scores()
        scores["unknown"] = 1.0
        return {
            "document_type": "unknown",
            "confidence": 0.0,
            "method": "fallback",
            "scores": scores,
            "signals": [],
            "reason": reason,
            "source_document": "",
            "page_number": None,
        }

    def _merge_lists(self, left: list[Any], right: list[Any]) -> list[str]:
        """Merge lists with de-duplication.

        Args:
            left: First list.
            right: Second list.

        Returns:
            Merged string list.
        """
        merged: list[str] = []
        for item in list(left or []) + list(right or []):
            value = str(item)
            if value not in merged:
                merged.append(value)
        return merged

    def _coerce_string_list(self, value: Any) -> list[str]:
        """Coerce value to string list.

        Args:
            value: Candidate value.

        Returns:
            List of strings.
        """
        if isinstance(value, list):
            return [str(item) for item in value]
        if value in (None, ""):
            return []
        return [str(value)]

    def _coerce_page_number(self, value: Any) -> int | None:
        """Coerce page number.

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

    def _clamp(self, value: Any) -> float:
        """Clamp numeric value to [0, 1].

        Args:
            value: Candidate numeric value.

        Returns:
            Clamped float.
        """
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            return 0.0


"""Healthcare document summarization with LangChain and Groq models.

This module accepts LangChain ``Document`` objects produced by
``modules.pdf_loader.PDFLoader`` and generates structured JSON summaries for
healthcare policies, clinical guidelines, contracts, coverage policies, and
regulatory documents.

Example:
    ```python
    from modules.pdf_loader import PDFLoader
    from modules.summarizer import HealthcareSummarizer

    loader = PDFLoader()
    docs = loader.load_multiple_pdfs("data/")

    summarizer = HealthcareSummarizer()
    summaries = summarizer.summarize_documents(docs)

    print(summaries)
    ```
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - dependency is installed in the app venv
    ChatGroq = None  # type: ignore[assignment]


class ChatModel(Protocol):
    """Protocol for LangChain-compatible chat models."""

    def invoke(self, messages: list[SystemMessage | HumanMessage]) -> Any:
        """Invoke a chat model.

        Args:
            messages: System and human messages for the model.

        Returns:
            A LangChain chat model response or response-like object.
        """
        ...


class HealthcareSummarizer:
    """Generates structured summaries for healthcare content documents."""

    def __init__(
        self,
        llm: ChatModel | None = None,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_retries: int = 2,
        request_timeout: float = 60.0,
        max_workers: int = 4,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the healthcare summarizer.

        Args:
            llm: Optional LangChain-compatible chat model. If omitted,
                ``ChatGroq`` is created lazily from the provided settings.
            model_name: Groq model name used when constructing ``ChatGroq``.
            temperature: Model temperature for deterministic summarization.
            max_retries: Number of retries for failed API or JSON attempts.
            request_timeout: Request timeout in seconds for Groq calls.
            max_workers: Maximum worker threads for multi-document summaries.
            logger: Optional logger instance. If omitted, a module logger is
                used.

        Returns:
            None.

        Raises:
            ValueError: If retry or worker configuration is invalid.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")
        if max_workers < 1:
            raise ValueError("max_workers must be greater than or equal to 1.")

        if llm is None:
            if ChatGroq is None:
                raise ImportError(
                    "langchain-groq is required for Groq-powered summarization."
                )

            groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not configured.")

            llm = ChatGroq(
                model=model_name,
                temperature=temperature,
                timeout=request_timeout,
                api_key=groq_api_key,
            )

        self.llm = llm
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)
        self._last_processing_time_seconds = 0.0

    def summarize_documents(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Summarize multiple LangChain documents.

        Args:
            documents: List of LangChain ``Document`` objects to summarize.

        Returns:
            A list of dictionaries containing source document names and
            structured summaries.

        Raises:
            TypeError: If ``documents`` is not a list of LangChain documents.
        """
        self._validate_documents(documents)
        if not documents:
            self.logger.warning("No documents provided for summarization.")
            self._last_processing_time_seconds = 0.0
            return []

        start_time = time.perf_counter()
        self.logger.info("Document batch loaded for summarization: %s", len(documents))

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self.summarize_document, document): document
                for document in documents
            }

            for future in as_completed(future_map):
                document = future_map[future]
                document_name = self._get_document_name(document)
                try:
                    summary = future.result()
                except Exception as error:
                    self.logger.error(
                        "Unexpected summarization failure for %s: %s",
                        document_name,
                        error,
                    )
                    summary = self._empty_summary()
                    summary["_error"] = str(error)

                results.append({"document": document_name, "summary": summary})

        self._last_processing_time_seconds = time.perf_counter() - start_time
        self.logger.info(
            "Generated summaries for %s documents in %.2f seconds.",
            len(results),
            self._last_processing_time_seconds,
        )
        return results

    def summarize_document(self, document: Document) -> dict[str, Any]:
        """Summarize a single LangChain document.

        Args:
            document: LangChain ``Document`` containing healthcare content.

        Returns:
            Structured summary dictionary with the required summary fields.

        Raises:
            TypeError: If ``document`` is not a LangChain ``Document``.
        """
        if not isinstance(document, Document):
            raise TypeError("document must be a LangChain Document.")

        document_name = self._get_document_name(document)
        self.logger.info("Document loaded for summarization: %s", document_name)

        text = document.page_content or ""
        if not text.strip():
            self.logger.warning("Empty document skipped: %s", document_name)
            return self._empty_summary()

        return self.generate_structured_summary(text)

    def generate_structured_summary(self, text: str) -> dict[str, Any]:
        """Generate a structured JSON summary from healthcare document text.

        Args:
            text: Healthcare document text to summarize.

        Returns:
            A valid JSON-compatible dictionary containing all required summary
            fields.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text.strip():
            self.logger.warning("Empty text provided for structured summary.")
            return self._empty_summary()

        messages = self._build_messages(text)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 2):
            request_start = time.perf_counter()
            try:
                self.logger.info("Submitting summarization API request. Attempt: %s", attempt)
                response = self.llm.invoke(messages)
                response_time = time.perf_counter() - request_start
                self.logger.info("Summarization API response time: %.2f seconds", response_time)

                content = self._extract_response_content(response)
                parsed_summary = self._parse_summary_json(content)
                normalized_summary = self._normalize_summary(parsed_summary)
                self.logger.info("Summary generated successfully.")
                return normalized_summary
            except Exception as error:
                last_error = error
                self.logger.error(
                    "Summary generation attempt %s failed: %s",
                    attempt,
                    error,
                )

        self.logger.error("Summary generation failed after retries: %s", last_error)
        summary = self._empty_summary()
        summary["_error"] = str(last_error) if last_error else "Unknown Groq summarization failure."
        return summary

    def export_summary(self, summary: dict[str, Any] | list[dict[str, Any]], output_path: str | Path) -> None:
        """Export a summary or list of summaries to JSON or Markdown.

        Args:
            summary: Summary dictionary or list of summary result dictionaries.
            output_path: Destination path. ``.md`` writes Markdown; all other
                extensions write JSON.

        Returns:
            None.

        Raises:
            OSError: If the output path cannot be written.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".md":
            path.write_text(self._to_markdown(summary), encoding="utf-8")
        else:
            path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        self.logger.info("Summary exported: %s", path)

    def get_summary_statistics(
        self,
        summary: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, int | str]:
        """Calculate summary statistics.

        Args:
            summary: Summary dictionary or list of summary result dictionaries.

        Returns:
            Dictionary with documents processed, average summary length, and
            processing time.
        """
        summaries = self._coerce_summary_list(summary)
        summary_lengths = [
            len(json.dumps(item.get("summary", item)).split())
            for item in summaries
        ]
        average_length = (
            round(sum(summary_lengths) / len(summary_lengths))
            if summary_lengths
            else 0
        )

        return {
            "documents_processed": len(summaries),
            "average_summary_length": average_length,
            "processing_time": f"{self._last_processing_time_seconds:.2f}s",
        }

    def _build_messages(self, text: str) -> list[SystemMessage | HumanMessage]:
        """Build structured prompt messages.

        Args:
            text: Healthcare document text.

        Returns:
            LangChain system and human messages.
        """
        system_prompt = (
            "You are a healthcare policy analyst. "
            "Summarize healthcare documents accurately. "
            "Preserve all clinical and reimbursement requirements. "
            "Do not invent information. "
            "Do not copy long raw passages from the document. "
            "Write concise, professional, analyst-ready summaries. "
            "Synthesize the content into polished business language rather than "
            "repeating source sentences verbatim. "
            "Return valid JSON only."
        )
        human_prompt = (
            "Generate a concise structured JSON summary for the healthcare "
            "document text below. Use this exact schema and include empty "
            "strings or empty arrays when information is not present. "
            "The executive_summary must be a strong 3-5 sentence overview written "
            "for a healthcare analyst or interviewer. "
            "Each array item should be a short, polished bullet-style sentence. "
            "Only populate eligibility_criteria when the document explicitly "
            "states member, patient, diagnosis, age, benefit, or clinical "
            "eligibility requirements. Do not place prior authorization rules "
            "or site-of-service instructions inside eligibility_criteria. "
            "Avoid OCR headers, page numbers, footers, duplicated phrases, and "
            "raw table dumps. Preserve important CPT, HCPCS, ICD-10, dollar, "
            "date, prior authorization, exclusion, and medical necessity details. "
            "Do not return copied clauses unless a direct quote is necessary. "
            "Prefer synthesized, plain, professional language:\n\n"
            "{\n"
            '  "document_type":"",\n'
            '  "purpose":"",\n'
            '  "covered_services":[],\n'
            '  "excluded_services":[],\n'
            '  "eligibility_criteria":[],\n'
            '  "medical_necessity":[],\n'
            '  "prior_authorization":"",\n'
            '  "coding_requirements":[],\n'
            '  "key_dates":[],\n'
            '  "important_changes":[],\n'
            '  "executive_summary":""\n'
            "}\n\n"
            f"Document text:\n{text}"
        )
        return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from a LangChain model response.

        Args:
            response: LangChain chat model response or response-like object.

        Returns:
            Response content as a string.

        Raises:
            ValueError: If content cannot be extracted.
        """
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response did not contain text content.")
        return content.strip()

    def _parse_summary_json(self, content: str) -> dict[str, Any]:
        """Parse model content into a JSON dictionary.

        Args:
            content: Raw model response content.

        Returns:
            Parsed JSON dictionary.

        Raises:
            json.JSONDecodeError: If JSON parsing fails.
            ValueError: If parsed JSON is not an object.
        """
        cleaned_content = self._strip_code_fences(content)
        parsed = json.loads(cleaned_content)
        if not isinstance(parsed, dict):
            raise ValueError("Summary response must be a JSON object.")
        return parsed

    def _strip_code_fences(self, content: str) -> str:
        """Remove Markdown code fences from model output.

        Args:
            content: Raw model response.

        Returns:
            Response content without wrapping code fences.
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

    def _normalize_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Normalize parsed summary output to the required schema.

        Args:
            summary: Parsed summary dictionary.

        Returns:
            Summary dictionary containing all required keys with expected basic
            value types.
        """
        normalized = self._empty_summary()
        for key, default_value in normalized.items():
            value = summary.get(key, default_value)
            if isinstance(default_value, list):
                normalized[key] = value if isinstance(value, list) else [str(value)]
            else:
                normalized[key] = value if isinstance(value, str) else str(value)

        if not normalized["executive_summary"].strip():
            normalized["executive_summary"] = self._build_fallback_executive_summary(
                normalized
            )
        return normalized

    def _build_fallback_executive_summary(
        self,
        summary: dict[str, Any],
    ) -> str:
        """Build an executive summary when the model leaves it blank.

        Args:
            summary: Normalized summary dictionary.

        Returns:
            A concise synthesized executive summary.
        """
        sentences: list[str] = []

        document_type = str(summary.get("document_type", "")).strip()
        purpose = str(summary.get("purpose", "")).strip()
        prior_authorization = str(summary.get("prior_authorization", "")).strip()

        covered_services = self._clean_summary_list(summary.get("covered_services", []))
        exclusions = self._clean_summary_list(summary.get("excluded_services", []))
        medical_necessity = self._clean_summary_list(summary.get("medical_necessity", []))
        coding_requirements = self._clean_summary_list(
            summary.get("coding_requirements", [])
        )
        key_dates = self._clean_summary_list(summary.get("key_dates", []))

        if document_type and purpose:
            sentences.append(
                f"This {document_type.lower()} outlines {self._ensure_sentence_fragment(purpose)}"
            )
        elif purpose:
            sentences.append(self._ensure_sentence_fragment(purpose).capitalize())
        elif document_type:
            sentences.append(
                f"This document is summarized as a {document_type.lower()}."
            )

        if covered_services:
            sentences.append(
                f"Coverage focuses on {self._join_examples(covered_services, limit=2)}."
            )

        if medical_necessity:
            sentences.append(
                f"Medical necessity criteria emphasize {self._join_examples(medical_necessity, limit=2)}."
            )

        if prior_authorization:
            sentences.append(
                f"Authorization guidance indicates {self._ensure_sentence_fragment(prior_authorization)}"
            )

        if exclusions:
            sentences.append(
                f"Key exclusions include {self._join_examples(exclusions, limit=2)}."
            )

        if coding_requirements:
            sentences.append(
                f"Coding requirements reference {self._join_examples(coding_requirements, limit=2)}."
            )

        if key_dates:
            sentences.append(
                f"Important policy dates include {self._join_examples(key_dates, limit=2)}."
            )

        summary_text = " ".join(
            self._ensure_sentence(item) for item in sentences[:5] if item.strip()
        ).strip()
        return summary_text

    def _clean_summary_list(self, values: Any) -> list[str]:
        """Return cleaned list items for summary synthesis."""
        if not isinstance(values, list):
            values = [values] if values else []

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = " ".join(str(value).strip().split())
            text = text.strip('" ').strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    def _join_examples(self, items: list[str], limit: int = 2) -> str:
        """Join a short list of examples into readable prose."""
        selected = items[:limit]
        if not selected:
            return ""
        if len(selected) == 1:
            return selected[0]
        return f"{selected[0]} and {selected[1]}"

    def _ensure_sentence_fragment(self, text: str) -> str:
        """Return text suitable for embedding inside a sentence."""
        return text.strip().rstrip(".")

    def _ensure_sentence(self, text: str) -> str:
        """Ensure a text fragment ends as a sentence."""
        stripped = text.strip()
        if not stripped:
            return ""
        if stripped[-1] not in ".!?":
            return f"{stripped}."
        return stripped

    def _empty_summary(self) -> dict[str, Any]:
        """Create an empty summary using the required schema.

        Args:
            None.

        Returns:
            Empty summary dictionary.
        """
        return {
            "document_type": "",
            "purpose": "",
            "covered_services": [],
            "excluded_services": [],
            "eligibility_criteria": [],
            "medical_necessity": [],
            "prior_authorization": "",
            "coding_requirements": [],
            "key_dates": [],
            "important_changes": [],
            "executive_summary": "",
        }

    def _get_document_name(self, document: Document) -> str:
        """Return the display name for a LangChain document.

        Args:
            document: LangChain document.

        Returns:
            Filename, source, or fallback document label.
        """
        filename = document.metadata.get("filename")
        if filename:
            return str(filename)

        source = document.metadata.get("source")
        if source:
            return Path(str(source)).name

        return "unknown_document"

    def _validate_documents(self, documents: list[Document]) -> None:
        """Validate a list of LangChain documents.

        Args:
            documents: Candidate list of LangChain documents.

        Returns:
            None.

        Raises:
            TypeError: If the input is not a list of LangChain documents.
        """
        if not isinstance(documents, list):
            raise TypeError("documents must be a list of LangChain Document objects.")

        invalid_items = [
            type(document).__name__
            for document in documents
            if not isinstance(document, Document)
        ]
        if invalid_items:
            raise TypeError(f"All documents must be LangChain Document objects: {invalid_items}")

    def _coerce_summary_list(
        self,
        summary: dict[str, Any] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Coerce summary input into a list for statistics and export helpers.

        Args:
            summary: Summary dictionary or list of summary dictionaries.

        Returns:
            List of summary dictionaries.
        """
        if isinstance(summary, list):
            return summary
        return [summary]

    def _to_markdown(self, summary: dict[str, Any] | list[dict[str, Any]]) -> str:
        """Convert summary output to Markdown.

        Args:
            summary: Summary dictionary or list of summary result dictionaries.

        Returns:
            Markdown representation of the summary output.
        """
        sections: list[str] = ["# Healthcare Document Summaries", ""]
        for index, item in enumerate(self._coerce_summary_list(summary), start=1):
            document_name = item.get("document", f"Document {index}")
            payload = item.get("summary", item)
            sections.extend([f"## {document_name}", ""])
            if isinstance(payload, dict):
                for key, value in payload.items():
                    title = key.replace("_", " ").title()
                    sections.append(f"### {title}")
                    if isinstance(value, list):
                        if value:
                            sections.extend(f"- {entry}" for entry in value)
                        else:
                            sections.append("- Not specified")
                    else:
                        sections.append(value or "Not specified")
                    sections.append("")
            else:
                sections.extend([str(payload), ""])

        return "\n".join(sections).strip() + "\n"

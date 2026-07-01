"""Centralized prompt management for healthcare content intelligence.

This module loads, validates, formats, versions, and exports prompt templates
used by LLM-facing modules. It does not call OpenAI APIs, execute LangChain
chains, run RAG, or perform summarization.

Example:
    ```python
    from modules.prompt_manager import PromptManager

    manager = PromptManager()

    prompt = manager.get_prompt(
        prompt_name="policy_qa",
        variables={
            "context": "MRI is covered after six weeks of therapy.",
            "question": "When is MRI covered?"
        }
    )

    print(prompt)
    ```
"""

from __future__ import annotations

import json
import logging
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptTemplate:
    """Prompt template metadata and body."""

    prompt_name: str
    template: str
    version: str = "1.0"
    description: str = ""
    required_variables: list[str] = field(default_factory=list)


class PromptManager:
    """Manages prompt templates and prompt metadata."""

    def __init__(
        self,
        prompts_directory: str | Path = "prompts",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the prompt manager.

        Args:
            prompts_directory: Directory that may contain prompt ``.txt`` files.
            logger: Optional logger instance.

        Returns:
            None.
        """
        self.prompts_directory = Path(prompts_directory)
        self.logger = logger or logging.getLogger(__name__)
        self.prompts: dict[str, PromptTemplate] = {}
        self._load_default_prompts()
        self.load_prompts_from_directory(self.prompts_directory)

    def load_prompt(self, prompt_name: str) -> dict[str, Any]:
        """Load a prompt template by name.

        Args:
            prompt_name: Supported prompt name.

        Returns:
            Prompt metadata dictionary, or structured error dictionary if the
            prompt is unavailable.
        """
        normalized_name = self._normalize_prompt_name(prompt_name)
        if not normalized_name:
            return self._error("Invalid prompt name.")

        if normalized_name in self.prompts:
            self.logger.info("Prompt loaded: %s", normalized_name)
            return asdict(self.prompts[normalized_name])

        file_path = self.prompts_directory / self._filename_for_prompt(normalized_name)
        if file_path.exists():
            loaded = self._load_prompt_file(normalized_name, file_path)
            if loaded:
                return asdict(loaded)

        self.logger.warning("Prompt not found: %s", normalized_name)
        return self._error(f"Prompt not found: {normalized_name}.")

    def get_prompt(
        self,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        """Return a prompt template, optionally formatted with variables.

        Args:
            prompt_name: Prompt name.
            variables: Optional variable mapping for template placeholders.

        Returns:
            Formatted prompt string, raw template string, or structured error
            dictionary if prompt loading or formatting fails.
        """
        prompt = self.load_prompt(prompt_name)
        if "error" in prompt:
            return prompt

        template = str(prompt["template"])
        if variables is None:
            return template
        return self.format_prompt(template, variables)

    def format_prompt(
        self,
        template: str,
        variables: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        """Safely format a prompt template with variables.

        Args:
            template: Prompt template containing ``{placeholder}`` values.
            variables: Values used to replace placeholders.

        Returns:
            Formatted prompt string, or structured error dictionary when
            required variables are missing.
        """
        if not isinstance(template, str) or not template.strip():
            return self._error("Template must be a non-empty string.")

        validation = self.validate_variables(template, variables)
        if not validation["valid"]:
            self.logger.warning("Prompt formatting failed: missing variables.")
            return {
                "error": "Missing required variables.",
                "missing_variables": validation["missing_variables"],
            }

        safe_variables = {key: str(value) for key, value in (variables or {}).items()}
        try:
            formatted = template.format_map(_SafeFormatDict(safe_variables))
            self.logger.info("Prompt formatted.")
            return formatted
        except (KeyError, ValueError) as error:
            self.logger.error("Prompt formatting error: %s", error)
            return self._error(f"Prompt formatting error: {error}.")

    def validate_variables(
        self,
        template: str,
        variables: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate that all required template variables are present.

        Args:
            template: Prompt template.
            variables: Candidate variable mapping.

        Returns:
            Validation dictionary with ``valid`` and ``missing_variables``.
        """
        required_variables = self._extract_required_variables(template)
        provided_variables = set((variables or {}).keys()) if isinstance(variables, dict) else set()
        missing_variables = sorted(variable for variable in required_variables if variable not in provided_variables)
        result = {"valid": not missing_variables, "missing_variables": missing_variables}
        self.logger.info("Prompt validation result: %s", result["valid"])
        return result

    def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompt names and metadata.

        Args:
            None.

        Returns:
            List of prompt metadata dictionaries.
        """
        return [
            self.get_prompt_metadata(prompt_name)
            for prompt_name in sorted(self.prompts)
        ]

    def add_prompt(
        self,
        prompt_name: str,
        template: str,
        version: str = "1.0",
        description: str = "",
    ) -> dict[str, Any]:
        """Add a new prompt to memory.

        Args:
            prompt_name: Prompt name.
            template: Prompt template text.
            version: Prompt version.
            description: Prompt description.

        Returns:
            Added prompt metadata, or structured error dictionary.
        """
        normalized_name = self._normalize_prompt_name(prompt_name)
        if not normalized_name:
            return self._error("Invalid prompt name.")
        if not isinstance(template, str) or not template.strip():
            return self._error("Prompt template cannot be empty.")

        prompt = PromptTemplate(
            prompt_name=normalized_name,
            template=template,
            version=version,
            description=description,
            required_variables=self._extract_required_variables(template),
        )
        self.prompts[normalized_name] = prompt
        self.logger.info("Prompt added: %s", normalized_name)
        return asdict(prompt)

    def update_prompt(
        self,
        prompt_name: str,
        template: str,
        version: str,
    ) -> dict[str, Any]:
        """Update an existing prompt template and version.

        Args:
            prompt_name: Prompt name.
            template: Updated prompt template.
            version: Updated prompt version.

        Returns:
            Updated prompt metadata, or structured error dictionary.
        """
        normalized_name = self._normalize_prompt_name(prompt_name)
        if normalized_name not in self.prompts:
            return self._error(f"Prompt not found: {normalized_name}.")
        if not isinstance(template, str) or not template.strip():
            return self._error("Prompt template cannot be empty.")

        current = self.prompts[normalized_name]
        current.template = template
        current.version = version
        current.required_variables = self._extract_required_variables(template)
        self.logger.info("Prompt updated: %s", normalized_name)
        return asdict(current)

    def delete_prompt(self, prompt_name: str) -> dict[str, Any]:
        """Delete a prompt from memory.

        Args:
            prompt_name: Prompt name.

        Returns:
            Deletion result dictionary.
        """
        normalized_name = self._normalize_prompt_name(prompt_name)
        if normalized_name not in self.prompts:
            return self._error(f"Prompt not found: {normalized_name}.")
        del self.prompts[normalized_name]
        self.logger.info("Prompt deleted: %s", normalized_name)
        return {"deleted": True, "prompt_name": normalized_name}

    def get_prompt_metadata(self, prompt_name: str) -> dict[str, Any]:
        """Return prompt metadata without the full template body.

        Args:
            prompt_name: Prompt name.

        Returns:
            Prompt metadata dictionary, or structured error dictionary.
        """
        normalized_name = self._normalize_prompt_name(prompt_name)
        if normalized_name not in self.prompts:
            return self._error(f"Prompt not found: {normalized_name}.")

        prompt = self.prompts[normalized_name]
        return {
            "prompt_name": prompt.prompt_name,
            "version": prompt.version,
            "description": prompt.description,
            "required_variables": prompt.required_variables,
        }

    def export_prompts(self, output_path: str | Path) -> bool:
        """Export prompt metadata and templates to JSON.

        Args:
            output_path: Destination JSON file.

        Returns:
            ``True`` when export succeeds; otherwise ``False``.
        """
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {name: asdict(prompt) for name, prompt in sorted(self.prompts.items())}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.logger.info("Prompt export completed: %s", path)
            return True
        except (OSError, TypeError) as error:
            self.logger.error("Prompt export failed: %s", error)
            return False

    def load_prompts_from_directory(self, directory: str | Path) -> dict[str, Any]:
        """Load all ``.txt`` prompt files from a directory.

        Args:
            directory: Directory containing prompt text files.

        Returns:
            Summary dictionary containing loaded prompts and errors.
        """
        path = Path(directory)
        if not path.exists():
            self.logger.warning("Prompt directory not found: %s", path)
            return {"loaded": [], "errors": [f"Directory not found: {path}."]}
        if not path.is_dir():
            self.logger.warning("Prompt path is not a directory: %s", path)
            return {"loaded": [], "errors": [f"Not a directory: {path}."]}

        loaded: list[str] = []
        errors: list[str] = []
        for file_path in sorted(path.glob("*.txt")):
            prompt_name = self._prompt_name_from_filename(file_path.stem)
            prompt = self._load_prompt_file(prompt_name, file_path)
            if prompt:
                loaded.append(prompt.prompt_name)
            else:
                errors.append(f"Failed to load prompt file: {file_path}.")
        return {"loaded": loaded, "errors": errors}

    def _load_default_prompts(self) -> None:
        """Load safe built-in default prompts.

        Args:
            None.

        Returns:
            None.
        """
        for prompt_name, template, description in self._default_prompt_definitions():
            self.prompts[prompt_name] = PromptTemplate(
                prompt_name=prompt_name,
                template=template,
                version="1.0",
                description=description,
                required_variables=self._extract_required_variables(template),
            )

    def _default_prompt_definitions(self) -> list[tuple[str, str, str]]:
        """Return default prompt definitions.

        Args:
            None.

        Returns:
            Prompt name, template, and description tuples.
        """
        return [
            (
                "summarization",
                (
                    "You are an expert healthcare policy analyst. Summarize the "
                    "following healthcare document using only the provided text. "
                    "Do not invent facts, do not include PHI, explain uncertainty, "
                    "and return structured JSON.\n\nDocument text:\n{document_text}"
                ),
                "Healthcare document summarization prompt",
            ),
            (
                "rule_extraction",
                (
                    "You are an expert healthcare policy analyst and rule extraction "
                    "engine. Convert the following policy text into explicit JSON "
                    "business rules. Preserve conditions, exceptions, thresholds, "
                    "citations, and missing information. Do not expose chain-of-thought.\n\n"
                    "Policy text:\n{policy_text}"
                ),
                "Healthcare rule extraction prompt",
            ),
            (
                "feature_extraction",
                (
                    "You are an expert healthcare feature extraction engine. Extract "
                    "ICD codes, CPT codes, HCPCS codes, diagnoses, services, therapy "
                    "duration, prior authorization, contract terms, dates, and coverage "
                    "features from the provided context. Return valid JSON only.\n\n"
                    "Healthcare text:\n{healthcare_text}"
                ),
                "Healthcare feature extraction prompt",
            ),
            (
                "policy_qa",
                (
                    "You are an expert healthcare policy analyst. Answer the question "
                    "using only the retrieved context. If the answer is missing, say "
                    "the information is unavailable. Cite source references and do not "
                    "invent facts.\n\nContext:\n{context}\n\nQuestion:\n{question}"
                ),
                "Healthcare policy question answering prompt",
            ),
            (
                "version_comparison",
                (
                    "You are an expert healthcare policy analyst. Compare the old and "
                    "new policy versions. Ignore formatting changes. Identify meaningful "
                    "clinical, reimbursement, coding, contract, medical necessity, and "
                    "authorization changes. Return structured JSON.\n\nOld version:\n"
                    "{old_text}\n\nNew version:\n{new_text}"
                ),
                "Healthcare policy version comparison prompt",
            ),
            (
                "explainability",
                (
                    "You are an expert healthcare payment integrity analyst. Generate a "
                    "plain-language explanation of the claim decision using only the "
                    "provided decision, rules, ML signals, features, and citations. Do "
                    "not change the decision or invent facts.\n\nDecision context:\n"
                    "{decision_context}"
                ),
                "Claim decision explainability prompt",
            ),
        ]

    def _load_prompt_file(self, prompt_name: str, file_path: Path) -> PromptTemplate | None:
        """Load one prompt file.

        Args:
            prompt_name: Prompt name.
            file_path: Prompt text file path.

        Returns:
            PromptTemplate when loaded successfully; otherwise ``None``.
        """
        try:
            template = file_path.read_text(encoding="utf-8")
            if not template.strip():
                self.logger.warning("Empty prompt file skipped: %s", file_path)
                return None
            prompt = PromptTemplate(
                prompt_name=prompt_name,
                template=template,
                version="1.0",
                description=f"Prompt loaded from {file_path.name}",
                required_variables=self._extract_required_variables(template),
            )
            self.prompts[prompt_name] = prompt
            self.logger.info("Prompt loaded from file: %s", file_path)
            return prompt
        except OSError as error:
            self.logger.error("Failed to load prompt file %s: %s", file_path, error)
            return None

    def _extract_required_variables(self, template: str) -> list[str]:
        """Extract replacement variables from a template.

        Args:
            template: Prompt template.

        Returns:
            Sorted placeholder names.
        """
        if not isinstance(template, str):
            return []
        variables: set[str] = set()
        formatter = string.Formatter()
        for _, field_name, _, _ in formatter.parse(template):
            if field_name:
                variables.add(field_name.split(".")[0].split("[")[0])
        return sorted(variables)

    def _filename_for_prompt(self, prompt_name: str) -> str:
        """Map prompt name to expected prompt file name.

        Args:
            prompt_name: Prompt name.

        Returns:
            Prompt filename.
        """
        mapping = {
            "summarization": "summarize_prompt.txt",
            "rule_extraction": "rule_prompt.txt",
            "feature_extraction": "feature_prompt.txt",
            "policy_qa": "qa_prompt.txt",
            "version_comparison": "compare_prompt.txt",
            "explainability": "explainability_prompt.txt",
        }
        return mapping.get(prompt_name, f"{prompt_name}.txt")

    def _prompt_name_from_filename(self, stem: str) -> str:
        """Map prompt file stem to prompt name.

        Args:
            stem: File stem.

        Returns:
            Prompt name.
        """
        mapping = {
            "summarize_prompt": "summarization",
            "rule_prompt": "rule_extraction",
            "feature_prompt": "feature_extraction",
            "qa_prompt": "policy_qa",
            "compare_prompt": "version_comparison",
            "explainability_prompt": "explainability",
        }
        return mapping.get(stem, self._normalize_prompt_name(stem))

    def _normalize_prompt_name(self, prompt_name: str) -> str:
        """Normalize prompt names.

        Args:
            prompt_name: Candidate prompt name.

        Returns:
            Normalized prompt name or empty string.
        """
        if not isinstance(prompt_name, str):
            return ""
        return prompt_name.strip().lower().replace("-", "_").replace(" ", "_")

    def _error(self, message: str) -> dict[str, Any]:
        """Build a structured error dictionary.

        Args:
            message: Error message.

        Returns:
            Error dictionary.
        """
        self.logger.error(message)
        return {"error": message}


class _SafeFormatDict(dict[str, str]):
    """Dictionary that leaves unknown format variables intact."""

    def __missing__(self, key: str) -> str:
        """Return placeholder text for missing keys.

        Args:
            key: Missing key.

        Returns:
            Original placeholder text.
        """
        return "{" + key + "}"


"""Unit tests for prompt management."""

from __future__ import annotations

import json
from pathlib import Path

from modules.prompt_manager import PromptManager


def test_load_default_prompt(tmp_path: Path) -> None:
    """Verify built-in default prompts are available."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    prompt = manager.load_prompt("summarization")

    assert prompt["prompt_name"] == "summarization"
    assert "document_text" in prompt["required_variables"]


def test_load_prompt_from_file(tmp_path: Path) -> None:
    """Verify prompt templates can be loaded from files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "qa_prompt.txt").write_text(
        "Answer {question} using {context}.",
        encoding="utf-8",
    )

    manager = PromptManager(prompts_directory=prompts_dir)
    prompt = manager.load_prompt("policy_qa")

    assert prompt["template"] == "Answer {question} using {context}."
    assert prompt["required_variables"] == ["context", "question"]


def test_format_prompt_with_variables(tmp_path: Path) -> None:
    """Verify templates format with provided variables."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    formatted = manager.format_prompt(
        "Answer {question} using {context}.",
        {"question": "When is MRI covered?", "context": "After six weeks."},
    )

    assert formatted == "Answer When is MRI covered? using After six weeks.."


def test_detect_missing_variables(tmp_path: Path) -> None:
    """Verify missing template variables are reported."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    validation = manager.validate_variables(
        "Answer {question} using {context}.",
        {"question": "When?"},
    )

    assert validation == {"valid": False, "missing_variables": ["context"]}


def test_get_prompt_missing_variables_returns_error(tmp_path: Path) -> None:
    """Verify get_prompt returns structured error for missing variables."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    result = manager.get_prompt("policy_qa", {"question": "When is MRI covered?"})

    assert isinstance(result, dict)
    assert result["error"] == "Missing required variables."
    assert result["missing_variables"] == ["context"]


def test_list_prompts(tmp_path: Path) -> None:
    """Verify prompt listing returns metadata."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    prompts = manager.list_prompts()

    assert len(prompts) >= 6
    assert "template" not in prompts[0]


def test_add_prompt(tmp_path: Path) -> None:
    """Verify adding a prompt stores it in memory."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    prompt = manager.add_prompt(
        "custom_prompt",
        "Use {context}.",
        version="1.0",
        description="Custom prompt",
    )

    assert prompt["prompt_name"] == "custom_prompt"
    assert prompt["required_variables"] == ["context"]


def test_update_prompt(tmp_path: Path) -> None:
    """Verify updating a prompt changes template and version."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")
    manager.add_prompt("custom", "Old {value}.")

    updated = manager.update_prompt("custom", "New {value} {context}.", "2.0")

    assert updated["version"] == "2.0"
    assert updated["required_variables"] == ["context", "value"]


def test_delete_prompt(tmp_path: Path) -> None:
    """Verify prompts can be deleted from memory."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")
    manager.add_prompt("custom", "Prompt.")

    result = manager.delete_prompt("custom")

    assert result == {"deleted": True, "prompt_name": "custom"}
    assert "error" in manager.load_prompt("custom")


def test_get_prompt_metadata(tmp_path: Path) -> None:
    """Verify metadata can be retrieved without template body."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    metadata = manager.get_prompt_metadata("feature_extraction")

    assert metadata["prompt_name"] == "feature_extraction"
    assert metadata["version"] == "1.0"
    assert "template" not in metadata


def test_export_prompts(tmp_path: Path) -> None:
    """Verify prompt export writes JSON."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")
    output_path = tmp_path / "prompts.json"

    assert manager.export_prompts(output_path) is True
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert "summarization" in exported
    assert "template" in exported["summarization"]


def test_handle_missing_prompt_file(tmp_path: Path) -> None:
    """Verify missing prompt files fall back to defaults when available."""
    manager = PromptManager(prompts_directory=tmp_path / "prompts")

    prompt = manager.load_prompt("rule_extraction")

    assert prompt["prompt_name"] == "rule_extraction"


def test_handle_invalid_prompt_name(tmp_path: Path) -> None:
    """Verify invalid prompt names return structured errors."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    result = manager.load_prompt("")

    assert "error" in result


def test_load_prompts_from_directory_missing(tmp_path: Path) -> None:
    """Verify missing directories return structured errors."""
    manager = PromptManager(prompts_directory=tmp_path / "missing")

    result = manager.load_prompts_from_directory(tmp_path / "does_not_exist")

    assert result["loaded"] == []
    assert result["errors"]


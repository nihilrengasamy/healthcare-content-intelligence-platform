"""Unit tests for policy version comparison."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from modules.compare_versions import PolicyVersionComparator


def test_compare_documents_no_differences() -> None:
    """Verify identical documents produce no content changes."""
    comparator = PolicyVersionComparator()
    old_document = Document(
        page_content="MRI covered after 8 weeks.",
        metadata={"filename": "policy.pdf", "version": "2025"},
    )
    new_document = Document(
        page_content="MRI covered after 8 weeks.",
        metadata={"filename": "policy.pdf", "version": "2026"},
    )

    report = comparator.compare_documents(old_document, new_document)

    assert report["added_sections"] == []
    assert report["removed_sections"] == []
    assert report["modified_sections"] == []
    assert report["overall_summary"] == "No meaningful content changes were detected."


def test_compare_documents_small_difference() -> None:
    """Verify small wording changes are detected."""
    comparator = PolicyVersionComparator()
    old_document = Document(page_content="Service is covered for adults.")
    new_document = Document(page_content="Service is covered for adult members.")

    report = comparator.compare_documents(old_document, new_document)

    assert report["added_sections"] == ["Service is covered for adult members."]
    assert report["removed_sections"] == ["Service is covered for adults."]
    assert report["modified_sections"]


def test_major_reimbursement_threshold_change_is_high_severity() -> None:
    """Verify medical necessity timing changes are identified."""
    comparator = PolicyVersionComparator()
    old_document = Document(page_content="MRI covered after 8 weeks of conservative therapy.")
    new_document = Document(page_content="MRI covered after 6 weeks of conservative therapy.")

    report = comparator.compare_documents(old_document, new_document)

    assert report["critical_changes"]
    assert report["critical_changes"][0]["severity"] == "High"
    assert report["critical_changes"][0]["category"] == "Medical Necessity"
    assert report["critical_changes"][0]["old"] == "8 week"
    assert report["critical_changes"][0]["new"] == "6 week"


def test_prior_authorization_change_is_critical() -> None:
    """Verify new prior authorization requirements are critical."""
    comparator = PolicyVersionComparator()
    old_document = Document(page_content="Prior authorization not required for MRI.")
    new_document = Document(page_content="Prior authorization required for MRI.")

    report = comparator.compare_documents(old_document, new_document)

    assert report["critical_changes"]
    assert report["critical_changes"][0]["severity"] == "Critical"
    assert report["critical_changes"][0]["category"] == "Prior Authorization"


def test_coding_changes_are_categorized() -> None:
    """Verify coding changes are categorized."""
    comparator = PolicyVersionComparator()
    old_document = Document(page_content="Use CPT 12345 for the procedure.")
    new_document = Document(page_content="Use CPT 67890 with modifier 59 for the procedure.")

    report = comparator.compare_documents(old_document, new_document)

    assert report["coding_changes"]
    assert any(change["category"] == "Coding" for change in report["critical_changes"])


def test_clinical_guideline_changes_are_categorized() -> None:
    """Verify clinical guideline changes are categorized."""
    comparator = PolicyVersionComparator()
    old_document = Document(page_content="Treatment is recommended for mild symptoms.")
    new_document = Document(page_content="Treatment is recommended for moderate symptoms.")

    report = comparator.compare_documents(old_document, new_document)

    assert report["clinical_changes"]


def test_contract_changes_are_categorized() -> None:
    """Verify payer-provider contract changes are categorized."""
    comparator = PolicyVersionComparator()
    old_document = Document(page_content="Provider payment rate is 100 dollars.")
    new_document = Document(page_content="Provider payment rate is 120 dollars.")

    report = comparator.compare_documents(old_document, new_document)

    assert report["contract_changes"]
    assert any(change["category"] == "Contract" for change in report["critical_changes"])


def test_compare_summaries_detects_field_differences() -> None:
    """Verify structured summaries can be compared."""
    comparator = PolicyVersionComparator()
    old_summary = {
        "document": "policy.pdf",
        "summary": {
            "document_type": "Coverage Policy",
            "prior_authorization": "Not required",
            "coding_requirements": ["Use CPT 12345"],
        },
    }
    new_summary = {
        "document": "policy.pdf",
        "summary": {
            "document_type": "Coverage Policy",
            "prior_authorization": "Required",
            "coding_requirements": ["Use CPT 67890"],
        },
    }

    report = comparator.compare_summaries(old_summary, new_summary)

    assert report["document_name"] == "policy.pdf"
    assert report["summary_differences"]
    assert {diff["field"] for diff in report["summary_differences"]} == {
        "coding_requirements",
        "prior_authorization",
    }


def test_json_export(tmp_path: Path) -> None:
    """Verify change reports export as JSON."""
    comparator = PolicyVersionComparator()
    report = comparator.generate_change_report("Old CPT 12345", "New CPT 67890")
    output_path = tmp_path / "change_report.json"

    comparator.export_change_report(report, output_path)

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert "added_sections" in exported
    assert "overall_summary" in exported


def test_markdown_and_html_export(tmp_path: Path) -> None:
    """Verify Markdown and HTML exports are supported."""
    comparator = PolicyVersionComparator()
    report = comparator.generate_change_report("Old payment rate", "New payment rate")
    markdown_path = tmp_path / "change_report.md"
    html_path = tmp_path / "change_report.html"

    comparator.export_change_report(report, markdown_path)
    comparator.export_change_report(report, html_path)

    assert "# Healthcare Policy Change Report" in markdown_path.read_text(encoding="utf-8")
    assert "<h1>Healthcare Policy Change Report</h1>" in html_path.read_text(encoding="utf-8")


def test_get_change_statistics() -> None:
    """Verify change statistics are calculated."""
    comparator = PolicyVersionComparator()
    report = comparator.compare_documents(
        Document(page_content="MRI covered after 8 weeks."),
        Document(page_content="MRI covered after 6 weeks."),
    )

    statistics = comparator.get_change_statistics(report)

    assert statistics["added"] == 1
    assert statistics["removed"] == 1
    assert statistics["modified"] == 1
    assert statistics["critical"] == 1
    assert statistics["processing_time"].endswith("s")


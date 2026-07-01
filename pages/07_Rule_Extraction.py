"""Rule extraction page."""

from __future__ import annotations

import importlib
import json
from typing import Any

from utils import pandas_compat as pd
import streamlit as st

from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.sidebar import render_sidebar
from components.table_view import render_table
from utils.session_manager import SessionManager
from utils.theme import apply_theme


def _all_documents(payloads: list[dict[str, Any]]) -> list[Any]:
    """Return all uploaded page documents."""
    documents: list[Any] = []
    for payload in payloads:
        documents.extend(payload.get("documents", []))
    return documents


def _format_rule_type(rule_type: str) -> str:
    """Return a presentation-friendly rule type label."""
    label_map = {
        "coverage": "Coverage",
        "medical_necessity": "Medical Necessity",
        "prior_authorization": "Prior Authorization",
        "billing": "Billing",
        "coding": "Coding",
        "contract": "Contract",
        "exclusion": "Exclusion",
        "frequency_limit": "Frequency Limit",
        "age_limit": "Age Limit",
    }
    return label_map.get(rule_type, rule_type.replace("_", " ").title())


def _format_decision(decision: str) -> str:
    """Return a presentation-friendly decision label."""
    decision_map = {
        "approve_if_criteria_met": "Approve If Criteria Met",
        "require_prior_authorization": "Require Prior Authorization",
        "require_coding_compliance": "Require Coding Compliance",
        "require_documentation_review": "Require Documentation Review",
        "require_medical_necessity_review": "Require Medical Necessity Review",
        "deny_not_covered": "Deny - Not Covered",
        "limit_repeat_service": "Limit Repeat Service",
        "require_review": "Require Review",
        "approve": "Approve If Criteria Met",
        "review": "Require Review",
        "deny": "Deny - Not Covered",
    }
    normalized = decision.strip().lower().replace(" ", "_")
    return decision_map.get(normalized, decision.replace("_", " ").title())


def _sanitize_rules(rules: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Normalize displayed rules so the page never shows stale weak structures."""
    try:
        import modules.rule_extractor as rule_extractor_module

        rule_extractor_module = importlib.reload(rule_extractor_module)
        extractor = rule_extractor_module.HealthcareRuleExtractor()
        normalized_rules = extractor._post_process_rules(rules)
        normalized_rules = extractor._assign_rule_ids(normalized_rules, force=True)
        validation = extractor.validate_rules(normalized_rules)
        return normalized_rules, validation
    except Exception:
        return rules, {"valid_rules": rules, "invalid_rules": []}


def _rules_frame(rules: list[dict[str, Any]]) -> pd.DataFrame:
    """Build rules DataFrame."""
    rows = []
    for rule in rules:
        conditions = rule.get("conditions", [])
        rows.append(
            {
                "Rule ID": rule.get("rule_id", ""),
                "Type": _format_rule_type(str(rule.get("rule_type", ""))),
                "Service": rule.get("service", "") or "Policy Requirement",
                "Decision": _format_decision(str(rule.get("decision", ""))),
                "Conditions": len(conditions),
                "Primary Condition": (
                    str(conditions[0].get("description", ""))[:80] if conditions else ""
                ),
                "Confidence": f"{float(rule.get('confidence', 0.0)):.2f}",
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Rule Extraction", page_icon="RULES", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Rule Extraction")
render_header(
    "Rule Extraction",
    "Extract executable JSON business rules from uploaded healthcare policy content.",
)

payloads = st.session_state.get("uploaded_documents", [])
if not payloads:
    st.warning("Upload and extract PDFs before rule extraction.")
else:
    try:
        from modules.prompt_manager import PromptManager

        with st.expander("Rule Extraction Prompt Template", expanded=False):
            st.code(PromptManager().get_prompt("rule_extraction"))
    except Exception as error:
        st.info(f"Prompt manager unavailable: {error}")

    if st.button("Extract Rules", type="primary"):
        try:
            import modules.rule_extractor as rule_extractor_module

            rule_extractor_module = importlib.reload(rule_extractor_module)
            HealthcareRuleExtractor = rule_extractor_module.HealthcareRuleExtractor

            with st.spinner("Extracting structured rules"):
                extractor = HealthcareRuleExtractor()
                rules = extractor.extract_rules_from_documents(_all_documents(payloads))
                validation = extractor.validate_rules(rules)
            st.session_state["rules"] = rules
            st.session_state["rules_validation"] = validation
            st.success(f"Extracted {len(rules)} rule(s).")
        except Exception as error:
            st.error(f"Rule extraction failed: {error}")

rules = st.session_state.get("rules", [])
rules_validation = st.session_state.get("rules_validation", {"valid_rules": [], "invalid_rules": []})
if rules:
    sanitized_rules, sanitized_validation = _sanitize_rules(rules)
    if sanitized_rules != rules:
        st.session_state["rules"] = sanitized_rules
        st.session_state["rules_validation"] = sanitized_validation
    rules = st.session_state.get("rules", sanitized_rules)
    rules_validation = st.session_state.get("rules_validation", sanitized_validation)
    rule_types = {_format_rule_type(str(rule.get("rule_type", "unknown"))) for rule in rules}
    avg_confidence = sum(float(rule.get("confidence", 0.0)) for rule in rules) / len(rules)
    valid_rule_count = len(rules_validation.get("valid_rules", [])) or len(rules)
    render_metric_row(
        [
            {"title": "Rules", "value": str(len(rules)), "delta": "Extracted", "icon": "RULE", "color": "#1455a0"},
            {"title": "Rule Types", "value": str(len(rule_types)), "delta": ", ".join(sorted(rule_types)), "icon": "TYPE", "color": "#047857"},
            {"title": "Valid Rules", "value": str(valid_rule_count), "delta": "Schema-checked", "icon": "CHK", "color": "#b45309"},
            {"title": "Avg Confidence", "value": f"{avg_confidence:.2f}", "delta": "Normalized", "icon": "CONF", "color": "#7c3aed"},
        ]
    )
    render_table(_rules_frame(rules), title="Extracted Rules", search=True)
    st.download_button(
        "Download Rules JSON",
        json.dumps(rules, indent=2),
        file_name="rules.json",
        mime="application/json",
    )
    render_json_viewer(rules, "Rules JSON", expanded=False)
else:
    st.info("Extracted rules will appear here.")

render_footer()



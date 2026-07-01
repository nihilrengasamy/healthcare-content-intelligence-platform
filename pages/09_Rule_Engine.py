"""Rule engine integration page."""

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


def _primary_features() -> dict[str, Any]:
    """Return the primary feature record from session state."""
    features = st.session_state.get("features", [])
    uploaded_documents = st.session_state.get("uploaded_documents", [])
    combined_source_text = "\n".join(
        getattr(document, "page_content", "")
        for payload in uploaded_documents
        for document in payload.get("documents", [])
        if getattr(document, "page_content", "")
    )
    if isinstance(features, list) and features:
        feature_record = dict(features[0])
    elif isinstance(features, dict):
        feature_record = dict(features)
    else:
        feature_record = {}

    if feature_record and combined_source_text:
        feature_record["source_text"] = combined_source_text
    return feature_record


def _results_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create rule evaluation summary table."""
    decision_labels = {
        "approve_if_criteria_met": "Approve If Criteria Met",
        "require_prior_authorization": "Require Prior Authorization",
        "require_coding_compliance": "Require Coding Compliance",
        "require_medical_necessity_review": "Require Medical Necessity Review",
        "deny_not_covered": "Deny - Not Covered",
        "limit_repeat_service": "Limit Repeat Service",
        "require_review": "Require Review",
        "no_match": "No Match",
    }
    return pd.DataFrame(
        [
            {
                "Rule ID": result.get("rule_id", ""),
                "Type": result.get("rule_type", ""),
                "Matched": result.get("matched", False),
                "Decision": decision_labels.get(
                    str(result.get("decision", "")).lower(),
                    str(result.get("decision", "")).replace("_", " ").title(),
                ),
                "Confidence": f"{float(result.get('confidence', 0.0)):.2f}",
                "Reason": result.get("reason", ""),
            }
            for result in results
        ]
    )


def _conditions_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create condition-level evaluation table."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for condition in result.get("conditions_evaluated", []):
            rows.append(
                {
                    "Rule ID": result.get("rule_id", ""),
                    "Field": condition.get("field", ""),
                    "Resolved Field": condition.get("resolved_field", condition.get("field", "")),
                    "Expected": condition.get("expected", ""),
                    "Actual": condition.get("actual"),
                    "Matched": condition.get("matched", False),
                    "Reason": condition.get("reason", ""),
                }
            )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Rule Engine", page_icon="RULE", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Rule Engine")
render_header(
    "Rule Engine",
    "Execute extracted JSON healthcare rules against structured features.",
)

rules = st.session_state.get("rules", [])
features = _primary_features()

if not rules:
    st.warning("Extract rules before running the rule engine.")
elif not features:
    st.warning("Extract features before running the rule engine.")
else:
    if st.button("Evaluate Rules", type="primary"):
        try:
            import modules.rule_engine as rule_engine_module

            rule_engine_module = importlib.reload(rule_engine_module)
            engine = rule_engine_module.HealthcareRuleEngine()
            with st.spinner("Evaluating rules against features"):
                results = engine.evaluate_rules(rules, features)
                statistics = engine.get_rule_engine_statistics(results)
            st.session_state["rule_results"] = results
            st.session_state["analytics"]["rule_engine_statistics"] = statistics
            st.success("Rule evaluation completed.")
        except Exception as error:
            st.error(f"Rule engine failed: {error}")

results = st.session_state.get("rule_results", [])
if results:
    matched = sum(1 for result in results if result.get("matched") is True)
    failed = len(results) - matched
    approve = sum(
        1
        for result in results
        if result.get("matched") is True
        and str(result.get("decision", "")).lower() in {"approve", "approved", "approve_if_criteria_met"}
    )
    deny = sum(
        1
        for result in results
        if result.get("matched") is True
        and str(result.get("decision", "")).lower() in {"deny", "denied", "deny_not_covered"}
    )
    review = sum(
        1
        for result in results
        if str(result.get("decision", "")).lower()
        in {
            "review",
            "manual_review",
            "require_review",
            "require_prior_authorization",
            "require_coding_compliance",
            "require_medical_necessity_review",
            "limit_repeat_service",
        }
    )
    render_metric_row(
        [
            {"title": "Rules Evaluated", "value": str(len(results)), "delta": "Total", "icon": "RULE", "color": "#1455a0"},
            {"title": "Matched", "value": str(matched), "delta": "Rules", "icon": "MATCH", "color": "#047857"},
            {"title": "Failed", "value": str(failed), "delta": "Rules", "icon": "FAIL", "color": "#b45309"},
            {"title": "Deny", "value": str(deny), "delta": "Overrides", "icon": "DENY", "color": "#b91c1c"},
        ]
    )
    st.caption(f"Approve rules: {approve} | Manual review rules: {review}")
    render_table(_results_frame(results), title="Rule Evaluation Results", search=True)
    conditions = _conditions_frame(results)
    if not conditions.empty:
        render_table(conditions, title="Condition Evaluation", search=True)
    st.download_button(
        "Download Rule Results JSON",
        json.dumps(results, indent=2),
        file_name="rule_results.json",
        mime="application/json",
    )
    render_json_viewer(results, "Rule Results JSON", expanded=False)
else:
    st.info("Rule evaluation results will appear here.")

render_footer()



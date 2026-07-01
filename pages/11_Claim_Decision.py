"""Claim decision integration page."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.sidebar import render_sidebar
from utils.session_manager import SessionManager
from utils.theme import apply_theme


def _primary_features() -> dict[str, Any]:
    """Return the primary feature record."""
    features = st.session_state.get("features", [])
    if isinstance(features, list) and features:
        return dict(features[0])
    if isinstance(features, dict):
        return dict(features)
    return {}


def _decision_color(decision: str) -> str:
    """Return accent color for a decision."""
    return {
        "approve": "#047857",
        "deny": "#b91c1c",
        "manual_review": "#b45309",
    }.get(decision, "#1455a0")


st.set_page_config(page_title="Claim Decision", page_icon="DECISION", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Claim Decision")
render_header(
    "Claim Decision",
    "Combine rule outcomes, ML signals, and extracted features into a final claim recommendation.",
)

rule_results = st.session_state.get("rule_results", [])
ml_prediction = st.session_state.get("ml_prediction")
features = _primary_features()

if not rule_results:
    st.warning("Run the Rule Engine before claim decision.")
elif not ml_prediction:
    st.warning("Run ML Prediction before claim decision.")
elif not features:
    st.warning("Extract features before claim decision.")
else:
    with st.expander("Decision Input Adjustments", expanded=False):
        documentation_complete = st.checkbox("Documentation complete", value=bool(features.get("documentation_complete", True)))
        prior_auth_required = st.checkbox("Prior authorization required", value=bool(features.get("prior_authorization_required", False)))
        prior_auth_obtained = st.checkbox("Prior authorization obtained", value=bool(features.get("prior_authorization_obtained", prior_auth_required)))
        features["documentation_complete"] = documentation_complete
        features["prior_authorization_required"] = prior_auth_required
        features["prior_authorization_obtained"] = prior_auth_obtained

    if st.button("Generate Claim Decision", type="primary"):
        try:
            from modules.claim_decision import ClaimDecisionEngine

            with st.spinner("Combining rule, ML, and feature signals"):
                decision = ClaimDecisionEngine().make_decision(rule_results, ml_prediction, features)
            st.session_state["claim_decision"] = decision
            st.success("Claim decision generated.")
        except Exception as error:
            st.error(f"Claim decision failed: {error}")

decision = st.session_state.get("claim_decision")
if decision:
    claim_decision = decision.get("claim_decision", "manual_review")
    rule_summary = decision.get("rule_summary", {})
    matched_rules = int(rule_summary.get("matched_rules", 0) or 0)
    failed_rules = int(rule_summary.get("failed_rules", 0) or 0)
    top_reason = next(iter(decision.get("reasons", [])), "")
    st.markdown(
        f"""
        <div class="hcip-card" style="border-top: 5px solid {_decision_color(claim_decision)};">
            <h2>{claim_decision.replace('_', ' ').title()}</h2>
            <p class="hcip-muted">{decision.get('recommendation', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if top_reason:
        if claim_decision == "deny" and matched_rules > 0:
            st.warning(
                f"Some rules matched ({matched_rules}), but the final denial was driven by higher-priority risk or completeness checks. "
                f"Primary rationale: {top_reason}"
            )
        else:
            st.info(f"Primary rationale: {top_reason}")
    render_metric_row(
        [
            {"title": "Decision Score", "value": f"{decision.get('decision_score', 0):.2f}", "delta": "Hybrid", "icon": "SCORE", "color": "#1455a0"},
            {"title": "Confidence", "value": f"{decision.get('confidence', 0):.2f}", "delta": "Decision", "icon": "CONF", "color": "#047857"},
            {"title": "Risk Flags", "value": str(len(decision.get("risk_flags", []))), "delta": "Detected", "icon": "RISK", "color": "#b91c1c"},
            {"title": "Conflicts", "value": str(len(decision.get("conflicts", []))), "delta": "Signals", "icon": "CONFLICT", "color": "#b45309"},
        ]
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Reasons")
        for reason in decision.get("reasons", []):
            st.write(f"- {reason}")
        st.markdown("#### Rule Summary")
        st.json(rule_summary)
    with col2:
        st.markdown("#### Risk Flags")
        if decision.get("risk_flags"):
            for flag in decision["risk_flags"]:
                st.warning(flag)
        else:
            st.success("No risk flags.")
        st.markdown("#### ML Summary")
        st.json(decision.get("ml_summary", {}))
    if decision.get("conflicts"):
        with st.expander("Conflicts", expanded=True):
            for conflict in decision["conflicts"]:
                st.write(f"- {conflict}")
    st.download_button(
        "Download Claim Decision JSON",
        json.dumps(decision, indent=2),
        file_name="claim_decision.json",
        mime="application/json",
    )
    render_json_viewer(decision, "Claim Decision JSON", expanded=False)
else:
    st.info("Claim decision output will appear here.")

render_footer()

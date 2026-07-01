"""Explainability integration page."""

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
        return features[0]
    if isinstance(features, dict):
        return features
    return {}


def _citations_from_rag() -> list[dict[str, Any]]:
    """Return citations from RAG response."""
    rag_response = st.session_state.get("rag_response") or {}
    return rag_response.get("citations", []) if isinstance(rag_response, dict) else []


st.set_page_config(page_title="Explainability", page_icon="EXPLAIN", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Explainability")
render_header(
    "Explainability",
    "Generate auditable plain-language explanations for claim decisions.",
)

decision = st.session_state.get("claim_decision")
rule_results = st.session_state.get("rule_results", [])
ml_prediction = st.session_state.get("ml_prediction")
features = _primary_features()
citations = _citations_from_rag()

if not decision:
    st.warning("Generate a claim decision before explainability.")
elif not ml_prediction:
    st.warning("ML prediction is required for explainability.")
else:
    if st.button("Generate Explanation", type="primary"):
        try:
            from modules.explainability import ClaimExplainabilityEngine

            with st.spinner("Generating explanation and audit trail"):
                explanation = ClaimExplainabilityEngine().generate_explanation(
                    decision,
                    rule_results,
                    ml_prediction,
                    features,
                    citations=citations,
                )
            st.session_state["explanation"] = explanation
            st.success("Explanation generated.")
        except Exception as error:
            st.error(f"Explainability failed: {error}")

explanation = st.session_state.get("explanation")
if explanation:
    executive_explanation = explanation.get(
        "executive_explanation",
        explanation.get("plain_language_summary", ""),
    )
    top_reasons = explanation.get("top_supporting_reasons", [])
    decision_drivers = explanation.get("decision_drivers", {})
    confidence_rationale = explanation.get("confidence_rationale", "")
    rule_reasoning = explanation.get("rule_based_reasoning", [])
    risk_items = explanation.get("risk_explanation", [])
    citations_list = explanation.get("supporting_citations", [])

    render_metric_row(
        [
            {"title": "Final Decision", "value": explanation.get("final_decision", ""), "delta": "Explained", "icon": "DEC", "color": "#1455a0"},
            {"title": "Confidence", "value": f"{explanation.get('decision_confidence', 0):.2f}", "delta": "Explanation", "icon": "CONF", "color": "#047857"},
            {"title": "Citations", "value": str(len(citations_list)), "delta": "Sources", "icon": "SRC", "color": "#7c3aed"},
            {"title": "Risks", "value": str(len(risk_items)), "delta": "Flags", "icon": "RISK", "color": "#b45309"},
        ]
    )
    st.markdown("#### Executive Explanation")
    st.info(executive_explanation)

    if confidence_rationale:
        st.caption(confidence_rationale)

    if top_reasons:
        st.markdown("#### Top Supporting Reasons")
        for reason in top_reasons[:3]:
            st.write(f"- {reason}")

    st.markdown("#### Recommended Next Action")
    st.success(explanation.get("recommended_next_action", ""))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Why This Decision Was Made")
        for item in decision_drivers.get("why_decided", []):
            st.write(f"- {item}")

        st.markdown("#### Evidence That Supported It")
        for item in decision_drivers.get("evidence_supported", []):
            st.write(f"- {item}")

    with col2:
        st.markdown("#### What Is Missing")
        for item in decision_drivers.get("missing_information", []):
            st.write(f"- {item}")

        st.markdown("#### What To Do Next")
        for item in decision_drivers.get("next_action_guidance", []):
            st.write(f"- {item}")

    with st.expander("Rule Reasoning", expanded=False):
        if rule_reasoning:
            for item in rule_reasoning[:5]:
                status = "Matched" if item.get("matched") else "Not matched"
                st.markdown(
                    f"""
                    <div class="hcip-card">
                        <strong>{item.get('rule_id', 'Rule')}</strong> | {status} | {str(item.get('decision', '')).replace('_', ' ').title()}<br/>
                        <span class="hcip-muted">{item.get('impact', '')}</span><br/>
                        <span>{item.get('reason', '')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for condition in item.get("conditions", [])[:3]:
                    st.write(f"- {condition.get('explanation', '')}")
        else:
            st.info("No rule reasoning available.")

    with st.expander("ML Reasoning", expanded=False):
        ml_reasoning = explanation.get("ml_reasoning", {})
        st.write(ml_reasoning.get("approval_probability_explanation", ""))
        st.write(ml_reasoning.get("fraud_risk_explanation", ""))
        st.write(ml_reasoning.get("medical_necessity_explanation", ""))
        st.write(ml_reasoning.get("ml_contribution", ""))

    with st.expander("Risk Flags", expanded=False):
        if risk_items:
            for risk in risk_items:
                st.warning(f"{risk.get('risk_flag', '')}: {risk.get('explanation', '')}")
        else:
            st.success("No significant risk flags were detected.")

    with st.expander("Supporting Citations", expanded=False):
        if citations_list:
            for citation in citations_list[:3]:
                st.markdown(
                    f"- **{citation.get('source', 'Source')}** | Page {citation.get('page', '')}: {citation.get('text', '')}"
                )
        else:
            st.info("No supporting citations available.")

    with st.expander("Audit Trail", expanded=False):
        st.json(explanation.get("audit_trail", {}))

    st.download_button(
        "Download Explanation JSON",
        json.dumps(explanation, indent=2),
        file_name="explanation.json",
        mime="application/json",
    )
    st.download_button(
        "Download Explanation Markdown",
        (
            f"# Claim Explanation\n\n"
            f"## Executive Explanation\n{executive_explanation}\n\n"
            f"## Top Supporting Reasons\n"
            + "\n".join(f"- {reason}" for reason in top_reasons[:3])
            + f"\n\n## Next Action\n{explanation.get('recommended_next_action', '')}\n"
        ),
        file_name="explanation.md",
        mime="text/markdown",
    )
    with st.expander("Raw Explanation JSON", expanded=False):
        render_json_viewer(explanation, "Explanation JSON", expanded=False)
else:
    st.info("Explanation output will appear here.")

render_footer()

"""Executive analytics dashboard."""

from __future__ import annotations

import json
from typing import Any

from utils import pandas_compat as pd
import plotly.graph_objects as go
import streamlit as st

from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.sidebar import render_sidebar
from components.status_badges import render_status_badge
from components.table_view import render_table
from utils.session_manager import SessionManager
from utils.theme import apply_theme


def _payloads() -> list[dict[str, Any]]:
    """Return uploaded document payloads."""
    value = st.session_state.get("uploaded_documents", [])
    return value if isinstance(value, list) else []


def _features() -> list[dict[str, Any]]:
    """Return feature records."""
    value = st.session_state.get("features", [])
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, dict) else []


def _rules() -> list[dict[str, Any]]:
    """Return extracted rules."""
    value = st.session_state.get("rules", [])
    return value if isinstance(value, list) else []


def _summary_count() -> int:
    """Return summary count from session state."""
    summary = st.session_state.get("summary", [])
    if isinstance(summary, list):
        return len(summary)
    return 1 if isinstance(summary, dict) and summary else 0


def _module_completion_status() -> dict[str, bool]:
    """Return end-to-end module completion flags."""
    rag_response = st.session_state.get("rag_response") or {}
    decision = st.session_state.get("claim_decision") or {}
    explanation = st.session_state.get("explanation") or {}
    evaluation = st.session_state.get("evaluation") or {}
    classification = st.session_state.get("document_classification") or {}

    return {
        "Upload": bool(_payloads()),
        "Classification": bool(classification),
        "Summarization": _summary_count() > 0,
        "Policy Chat": bool(isinstance(rag_response, dict) and rag_response.get("answer")),
        "Rule Extraction": bool(_rules()),
        "Feature Extraction": bool(_features()),
        "Rule Engine": bool(st.session_state.get("rule_results")),
        "ML Prediction": bool(st.session_state.get("ml_prediction")),
        "Claim Decision": bool(decision),
        "Explainability": bool(explanation),
        "Evaluation": bool(evaluation),
    }


def _average(values: list[Any]) -> float:
    """Return average of numeric values."""
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(numeric) / len(numeric), 2) if numeric else 0.0


def _count_pages(payloads: list[dict[str, Any]]) -> int:
    """Return total page count from uploaded documents."""
    return sum(int(payload.get("metadata", {}).get("pages", 0) or 0) for payload in payloads)


def _active_document_type(classification: dict[str, Any]) -> str:
    """Return classified document type."""
    return str(classification.get("document_type", "unknown")).replace("_", " ").title()


def _decision_distribution(decision: dict[str, Any]) -> list[dict[str, Any]]:
    """Build claim decision distribution rows."""
    claim_decision = str(decision.get("claim_decision", "none"))
    return [
        {"Decision": "Approve", "Count": 1 if claim_decision == "approve" else 0},
        {"Decision": "Deny", "Count": 1 if claim_decision == "deny" else 0},
        {"Decision": "Manual Review", "Count": 1 if claim_decision == "manual_review" else 0},
    ]


def _distribution_frame(values: list[str], label: str) -> pd.DataFrame:
    """Build a simple distribution frame."""
    meaningful = [value for value in values if value and value != "unknown"]
    if not meaningful:
        return pd.DataFrame([])
    series = pd.Series(meaningful).value_counts().reset_index()
    series.columns = [label, "Count"]
    return series


def _feature_distribution(features: list[dict[str, Any]]) -> pd.DataFrame:
    """Build compact feature presence distribution."""
    rows = [
        {"Feature": "ICD Codes", "Count": sum(1 for item in features if item.get("icd_codes"))},
        {"Feature": "CPT Codes", "Count": sum(1 for item in features if item.get("cpt_codes"))},
        {"Feature": "Prior Auth", "Count": sum(1 for item in features if item.get("prior_authorization_required") is True)},
        {"Feature": "Contract Terms", "Count": sum(1 for item in features if item.get("contract_terms"))},
    ]
    rows = [row for row in rows if row["Count"] > 0]
    return pd.DataFrame(rows)


def _confidence_distribution(
    decision: dict[str, Any],
    ml_prediction: dict[str, Any],
    explanation: dict[str, Any],
    evaluation: dict[str, Any],
) -> pd.DataFrame:
    """Build compact confidence score frame."""
    rows = [
        {"Metric": "Decision Confidence", "Value": decision.get("confidence", 0.0)},
        {"Metric": "ML Confidence", "Value": ml_prediction.get("model_confidence", 0.0)},
        {"Metric": "Explanation Confidence", "Value": explanation.get("decision_confidence", 0.0)},
        {"Metric": "Evaluation Score", "Value": evaluation.get("overall_score", 0.0)},
    ]
    rows = [
        {"Metric": row["Metric"], "Value": float(row["Value"])}
        for row in rows
        if isinstance(row.get("Value"), (int, float))
    ]
    return pd.DataFrame(rows)


def _analytics_data() -> dict[str, Any]:
    """Build executive analytics data from session state."""
    payloads = _payloads()
    rules = _rules()
    features = _features()
    rule_results = st.session_state.get("rule_results", []) or []
    decision = st.session_state.get("claim_decision") or {}
    evaluation = (st.session_state.get("evaluation") or {}).get("pipeline", {})
    classification = st.session_state.get("document_classification") or {}
    ml_prediction = st.session_state.get("ml_prediction") or {}
    explanation = st.session_state.get("explanation") or {}
    rag_response = st.session_state.get("rag_response") or {}
    module_completion = _module_completion_status()
    completed_modules = sum(1 for completed in module_completion.values() if completed)

    return {
        "documents_uploaded": len(payloads),
        "pages_processed": _count_pages(payloads),
        "document_type": _active_document_type(classification),
        "policies_processed": _summary_count(),
        "rules_extracted": len(rules),
        "features_extracted": len(features),
        "questions_answered": 1 if isinstance(rag_response, dict) and rag_response.get("answer") else 0,
        "claims_evaluated": 1 if decision else 0,
        "claim_decision": decision.get("claim_decision", "none"),
        "average_confidence": _average(
            [
                decision.get("confidence"),
                ml_prediction.get("model_confidence"),
                explanation.get("decision_confidence"),
            ]
        ),
        "average_evaluation_score": float(evaluation.get("overall_score", 0.0) or 0.0),
        "evaluation_coverage": float(evaluation.get("evaluation_coverage", 0.0) or 0.0),
        "module_completion_rate": round(completed_modules / len(module_completion), 2) if module_completion else 0.0,
        "completed_modules": completed_modules,
        "total_modules": len(module_completion),
        "rule_types": [str(rule.get("rule_type", "unknown")).replace("_", " ").title() for rule in rules if isinstance(rule, dict)],
        "document_types": [classification.get("document_type", "unknown")] if classification else [],
        "feature_counts": {
            "ICD Codes": sum(1 for item in features if item.get("icd_codes")),
            "CPT Codes": sum(1 for item in features if item.get("cpt_codes")),
            "Prior Auth": sum(1 for item in features if item.get("prior_authorization_required") is True),
            "Contract Terms": sum(1 for item in features if item.get("contract_terms")),
        },
        "rule_results": rule_results,
        "evaluation": evaluation,
        "module_completion": module_completion,
    }


def _render_donut(rows: list[dict[str, Any]], label_key: str, value_key: str, title: str) -> None:
    """Render a donut chart with an empty state."""
    meaningful = [row for row in rows if float(row.get(value_key, 0) or 0) > 0]
    st.markdown(f"#### {title}")
    if not meaningful:
        st.info(f"No data available for {title.lower()} yet.")
        return
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[row.get(label_key, "") for row in meaningful],
                values=[row.get(value_key, 0) for row in meaningful],
                hole=0.60,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 16, "b": 0}, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def _render_horizontal_bar(rows: list[dict[str, Any]], label_key: str, value_key: str, title: str, color: str) -> None:
    """Render a compact horizontal bar chart with empty-state handling."""
    st.markdown(f"#### {title}")
    meaningful = [row for row in rows if float(row.get(value_key, 0) or 0) > 0]
    if not meaningful:
        st.info(f"No data available for {title.lower()} yet.")
        return
    fig = go.Figure(
        data=[
            go.Bar(
                x=[row.get(value_key, 0) for row in meaningful],
                y=[row.get(label_key, "") for row in meaningful],
                orientation="h",
                marker={"color": color},
                text=[row.get(value_key, 0) for row in meaningful],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 16, "b": 0},
        xaxis_title="Count",
        yaxis_title="",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_score_bars(rows: list[dict[str, Any]], title: str) -> None:
    """Render compact score bars for quality metrics."""
    st.markdown(f"#### {title}")
    meaningful = [row for row in rows if isinstance(row.get("Value"), (int, float))]
    if not meaningful:
        st.info(f"No data available for {title.lower()} yet.")
        return
    fig = go.Figure(
        data=[
            go.Bar(
                x=[row.get("Metric", "") for row in meaningful],
                y=[row.get("Value", 0.0) for row in meaningful],
                marker={"color": "#1455a0"},
                text=[f"{float(row.get('Value', 0.0)):.2f}" for row in meaningful],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 16, "b": 0},
        yaxis={"range": [0, 1], "title": "Score"},
        xaxis_title="",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _executive_summary(data: dict[str, Any]) -> tuple[str, str]:
    """Return dashboard summary headline and body."""
    decision = str(data.get("claim_decision", "none")).replace("_", " ").title()
    completion_rate = float(data.get("module_completion_rate", 0.0))
    evaluation_score = float(data.get("average_evaluation_score", 0.0))
    if completion_rate >= 0.85 and evaluation_score >= 0.85:
        return (
            "The platform is showing strong end-to-end readiness.",
            f"Most workflow modules are complete, the current decision is {decision}, and quality signals are consistently strong across extraction, decisioning, and evaluation.",
        )
    if completion_rate >= 0.60:
        return (
            "The platform workflow is substantially complete and interview-ready.",
            f"The pipeline has progressed through {data.get('completed_modules', 0)} of {data.get('total_modules', 0)} major stages, with the current claim outcome recorded as {decision}.",
        )
    return (
        "The dashboard is active, but more workflow stages should be completed for a full executive view.",
        "Upload, classify, summarize, extract, and evaluate more of the pipeline to unlock richer business analytics.",
    )


st.set_page_config(page_title="Analytics", page_icon="ANALYTICS", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Analytics")
render_header(
    "Analytics Dashboard",
    "Executive view of document intelligence, extraction quality, decision support, and platform readiness.",
)

data = _analytics_data()
st.session_state["analytics"]["dashboard"] = data

decision_rows = _decision_distribution(st.session_state.get("claim_decision") or {})
document_type_frame = _distribution_frame(
    [str(item).replace("_", " ").title() for item in data.get("document_types", [])],
    "Document Type",
)
rule_type_frame = _distribution_frame(data.get("rule_types", []), "Rule Type")
feature_frame = _feature_distribution(_features())
score_frame = _confidence_distribution(
    st.session_state.get("claim_decision") or {},
    st.session_state.get("ml_prediction") or {},
    st.session_state.get("explanation") or {},
    (st.session_state.get("evaluation") or {}).get("pipeline", {}),
)

headline, body = _executive_summary(data)
st.markdown("#### Executive Summary")
st.info(f"{headline} {body}")

render_metric_row(
    [
        {
            "title": "Documents Uploaded",
            "value": str(data["documents_uploaded"]),
            "delta": f"{data['pages_processed']} pages processed",
            "icon": "DOC",
            "color": "#1455a0",
        },
        {
            "title": "Rules Extracted",
            "value": str(data["rules_extracted"]),
            "delta": "Policy rules",
            "icon": "RULE",
            "color": "#047857",
        },
        {
            "title": "Features Extracted",
            "value": str(data["features_extracted"]),
            "delta": "Structured records",
            "icon": "FEAT",
            "color": "#7c3aed",
        },
        {
            "title": "Evaluation Score",
            "value": f"{data['average_evaluation_score']:.2f}",
            "delta": "Quality",
            "icon": "EVAL",
            "color": "#b45309",
        },
    ]
)

render_metric_row(
    [
        {
            "title": "Current Decision",
            "value": str(data["claim_decision"]).replace("_", " ").title(),
            "delta": f"{data['claims_evaluated']} claim(s) evaluated",
            "icon": "DEC",
            "color": "#1455a0",
        },
        {
            "title": "Policy Questions",
            "value": str(data["questions_answered"]),
            "delta": "Answered",
            "icon": "RAG",
            "color": "#047857",
        },
        {
            "title": "Avg Confidence",
            "value": f"{data['average_confidence']:.2f}",
            "delta": "Decision / ML / Explainability",
            "icon": "CONF",
            "color": "#7c3aed",
        },
        {
            "title": "Workflow Coverage",
            "value": f"{float(data['module_completion_rate']):.0%}",
            "delta": f"{data['completed_modules']} of {data['total_modules']} modules",
            "icon": "FLOW",
            "color": "#b45309",
        },
    ]
)

left, right = st.columns([1.3, 1.0])
with left:
    _render_donut(decision_rows, "Decision", "Count", "Claim Decision Distribution")
with right:
    st.markdown("#### Workflow Status")
    for module_name, completed in data.get("module_completion", {}).items():
        name_col, status_col = st.columns([3, 1])
        with name_col:
            st.write(module_name)
        with status_col:
            render_status_badge("Completed" if completed else "Waiting")

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    _render_horizontal_bar(
        document_type_frame.to_dict("records"),
        "Document Type",
        "Count",
        "Document Type Distribution",
        "#1455a0",
    )
with row1_col2:
    _render_horizontal_bar(
        rule_type_frame.to_dict("records"),
        "Rule Type",
        "Count",
        "Rule Type Distribution",
        "#047857",
    )

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    _render_horizontal_bar(
        feature_frame.to_dict("records"),
        "Feature",
        "Count",
        "Feature Coverage",
        "#7c3aed",
    )
with row2_col2:
    _render_score_bars(score_frame.to_dict("records"), "Confidence and Quality Signals")

st.markdown("#### Operational Snapshot")
snapshot_rows = [
    {
        "Documents Uploaded": data["documents_uploaded"],
        "Pages Processed": data["pages_processed"],
        "Document Type": data["document_type"],
        "Policies Processed": data["policies_processed"],
        "Rules Extracted": data["rules_extracted"],
        "Features Extracted": data["features_extracted"],
        "Questions Answered": data["questions_answered"],
        "Claims Evaluated": data["claims_evaluated"],
        "Current Decision": str(data["claim_decision"]).replace("_", " ").title(),
        "Average Confidence": data["average_confidence"],
        "Evaluation Score": data["average_evaluation_score"],
        "Workflow Coverage": f"{float(data['module_completion_rate']):.0%}",
    }
]
render_table(pd.DataFrame(snapshot_rows), title="Dashboard Snapshot", search=False)

download_payload = {
    **data,
    "decision_distribution": decision_rows,
    "document_type_distribution": document_type_frame.to_dict("records"),
    "rule_type_distribution": rule_type_frame.to_dict("records"),
    "feature_distribution": feature_frame.to_dict("records"),
    "score_distribution": score_frame.to_dict("records"),
}
st.download_button(
    "Download Dashboard Data JSON",
    json.dumps(download_payload, indent=2),
    file_name="analytics_dashboard.json",
    mime="application/json",
)

with st.expander("Analytics JSON", expanded=False):
    render_json_viewer(download_payload, "Analytics JSON", expanded=False)

render_footer()

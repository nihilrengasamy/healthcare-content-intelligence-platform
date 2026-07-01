"""Evaluation integration page."""

from __future__ import annotations

import json
from typing import Any

from utils import pandas_compat as pd
import plotly.graph_objects as go
import streamlit as st

from components.chart_helpers import render_bar_chart
from components.footer import render_footer
from components.header import render_header
from components.json_viewer import render_json_viewer
from components.metric_cards import render_metric_row
from components.sidebar import render_sidebar
from components.status_badges import render_status_badge
from components.table_view import render_table
from utils.session_manager import SessionManager
from utils.theme import apply_theme


def _summary_payload(summary: Any) -> dict[str, Any]:
    """Return primary summary payload from session state."""
    if isinstance(summary, list) and summary:
        item = summary[0]
        return item.get("summary", item) if isinstance(item, dict) else {}
    if isinstance(summary, dict):
        return summary.get("summary", summary)
    return {}


def _score_frame(evaluation: dict[str, Any]) -> pd.DataFrame:
    """Create evaluation score DataFrame."""
    rows = [
        {"Metric": "Overall", "Score": evaluation.get("overall_score", 0.0)},
        {"Metric": "RAG", "Score": evaluation.get("rag_score", 0.0)},
        {"Metric": "Rules", "Score": evaluation.get("rule_score", 0.0)},
        {"Metric": "Features", "Score": evaluation.get("feature_score", 0.0)},
        {"Metric": "Summary", "Score": evaluation.get("summary_score", 0.0)},
        {"Metric": "Explanation", "Score": evaluation.get("explanation_score", 0.0)},
    ]
    return pd.DataFrame(rows)


def _module_status_frame(evaluation: dict[str, Any]) -> pd.DataFrame:
    """Create module status table."""
    module_status = evaluation.get("module_status", {})
    rows = [
        {
            "Module": item.get("label", key.title()),
            "Status": item.get("status", "pending").title(),
            "Score": item.get("score", 0.0),
        }
        for key, item in module_status.items()
        if isinstance(item, dict)
    ]
    return pd.DataFrame(rows)


def _readiness_summary(pipeline: dict[str, Any]) -> tuple[str, str]:
    """Return executive readiness heading and explanation."""
    overall = float(pipeline.get("overall_score", 0.0))
    coverage = float(pipeline.get("evaluation_coverage", 0.0))
    pending = pipeline.get("pending_modules", [])
    if pipeline.get("passed") is True and not pending:
        return (
            "Platform quality looks strong across the evaluated workflow.",
            "All major pipeline modules were evaluated and the current outputs are in a strong interview-ready range.",
        )
    if pipeline.get("passed") is True and pending:
        return (
            "Completed modules are performing well, with a few evaluation steps still pending.",
            f"The evaluated portions of the platform scored {overall:.2f}, but coverage is {coverage:.0%} because some modules have not been assessed yet.",
        )
    return (
        "The platform is partially ready, but a few quality gaps still need attention.",
        f"The current completed-module score is {overall:.2f} with {coverage:.0%} evaluation coverage, so the page is signaling what to improve next rather than failing silently.",
    )


def _render_radar(scores: pd.DataFrame) -> None:
    """Render radar chart for evaluation scores."""
    categories = scores["Metric"].tolist()
    values = scores["Score"].tolist()
    fig = go.Figure(
        data=go.Scatterpolar(r=values + values[:1], theta=categories + categories[:1], fill="toself")
    )
    fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="Evaluation", page_icon="EVAL", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Evaluation")
render_header(
    "Evaluation",
    "Evaluate AI output quality, grounding, citations, extraction quality, and full pipeline readiness.",
)

if st.button("Run Platform Evaluation", type="primary"):
    try:
        from modules.evaluation import HealthcareAIEvaluator

        evaluator = HealthcareAIEvaluator()
        rag_response = st.session_state.get("rag_response") or {}
        outputs = {
            "rag_response": rag_response,
            "rules": st.session_state.get("rules", []),
            "features": st.session_state.get("features", []),
            "summary": _summary_payload(st.session_state.get("summary")),
            "decision": st.session_state.get("claim_decision") or {},
            "explanation": st.session_state.get("explanation") or {},
        }
        with st.spinner("Evaluating full healthcare AI pipeline"):
            pipeline_eval = evaluator.evaluate_pipeline(outputs)
            report = evaluator.generate_evaluation_report(pipeline_eval)
        evaluation = {"pipeline": pipeline_eval, "report": report}
        st.session_state["evaluation"] = evaluation
        st.success("Evaluation completed.")
    except Exception as error:
        st.error(f"Evaluation failed: {error}")

evaluation = st.session_state.get("evaluation")
if evaluation:
    pipeline = evaluation.get("pipeline", {})
    scores = _score_frame(pipeline)
    module_status = _module_status_frame(pipeline)
    readiness_title, readiness_body = _readiness_summary(pipeline)
    render_metric_row(
        [
            {"title": "Overall Score", "value": f"{pipeline.get('overall_score', 0):.2f}", "delta": "Completed Modules", "icon": "ALL", "color": "#1455a0"},
            {"title": "Coverage", "value": f"{pipeline.get('evaluation_coverage', 0):.0%}", "delta": "Modules Evaluated", "icon": "COV", "color": "#047857"},
            {"title": "Rule Score", "value": f"{pipeline.get('rule_score', 0):.2f}", "delta": "Schema", "icon": "RULE", "color": "#7c3aed"},
            {"title": "Passed", "value": str(pipeline.get("passed", False)), "delta": "Status", "icon": "PASS", "color": "#b45309"},
        ]
    )
    st.markdown("#### Executive Readiness Summary")
    st.info(f"{readiness_title} {readiness_body}")

    status_col1, status_col2 = st.columns([2, 1])
    with status_col1:
        st.markdown("#### Module Evaluation Status")
        if not module_status.empty:
            for row in module_status.to_dict("records"):
                left, mid, right = st.columns([3, 1, 1])
                with left:
                    st.write(row.get("Module", ""))
                with mid:
                    render_status_badge("Completed" if row.get("Status") == "Completed" else "Waiting")
                with right:
                    st.write(f"{float(row.get('Score', 0.0)):.2f}")
        else:
            st.info("No module evaluation status is available yet.")
    with status_col2:
        st.markdown("#### Pending Modules")
        pending_modules = pipeline.get("pending_modules", [])
        if pending_modules:
            for module in pending_modules:
                st.write(f"- {module.replace('_', ' ').title()}")
        else:
            st.success("All core evaluation modules have been assessed.")

    col1, col2 = st.columns(2)
    with col1:
        render_bar_chart(scores, "Metric", "Score", "Evaluation Scores")
    with col2:
        _render_radar(scores)
    st.markdown("#### Recommendations")
    for recommendation in pipeline.get("recommendations", []):
        st.write(f"- {recommendation}")
    if pipeline.get("critical_issues"):
        with st.expander("Critical Issues", expanded=True):
            for issue in pipeline["critical_issues"]:
                st.warning(issue)
    else:
        st.success("No critical evaluation issues were detected in the completed modules.")

    if pipeline.get("rag_score", 0.0) == 0.0 and "rag" in pipeline.get("pending_modules", []):
        st.warning(
            "RAG is currently unevaluated, not necessarily poor quality. Run Policy Chat with a real question and citation-backed answer, then rerun evaluation."
        )

    render_table(scores, title="Evaluation Score Table", search=False)
    render_table(module_status, title="Module Status Table", search=False)
    st.download_button(
        "Download Evaluation Report JSON",
        json.dumps(evaluation.get("report", {}), indent=2),
        file_name="evaluation_report.json",
        mime="application/json",
    )
    with st.expander("Evaluation JSON", expanded=False):
        render_json_viewer(evaluation, "Evaluation JSON", expanded=False)
else:
    st.info("Run evaluation after generating RAG, rules, features, decisions, and explanations.")

render_footer()



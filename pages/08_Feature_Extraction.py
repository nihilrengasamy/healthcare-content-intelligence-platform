"""Feature extraction page."""

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


def _feature_summary_frame(features: list[dict[str, Any]]) -> pd.DataFrame:
    """Build feature summary DataFrame."""
    rows = []
    for index, item in enumerate(features, start=1):
        rows.append(
            {
                "Record": index,
                "ICD Codes": ", ".join(item.get("icd_codes", [])),
                "CPT Codes": ", ".join(item.get("cpt_codes", [])),
                "Procedure": item.get("procedure", ""),
                "Service": item.get("service", ""),
                "Therapy Weeks": item.get("therapy_weeks"),
                "Prior Auth": item.get("prior_authorization_required"),
                "Coverage": item.get("coverage_type", ""),
                "Confidence": f"{float(item.get('confidence', 0.0)):.2f}",
            }
        )
    return pd.DataFrame(rows)


def _post_process_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and consolidate feature records for display."""
    try:
        import modules.feature_extractor as feature_extractor_module

        feature_extractor_module = importlib.reload(feature_extractor_module)
        extractor = feature_extractor_module.HealthcareFeatureExtractor()
        consolidate = getattr(extractor, "consolidate_feature_records", None)
        return consolidate(features) if callable(consolidate) else features
    except Exception:
        return features


def _render_primary_feature_profile(feature: dict[str, Any]) -> None:
    """Render a polished primary feature profile for interview-ready display."""
    st.markdown("### Primary Policy Feature Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Procedure**  \n{feature.get('procedure') or 'Not identified'}")
        st.markdown(f"**Diagnosis**  \n{feature.get('diagnosis') or 'Not identified'}")
        st.markdown(f"**Coverage**  \n{feature.get('coverage_type') or 'Not identified'}")
    with col2:
        st.markdown(
            f"**CPT Codes**  \n{', '.join(feature.get('cpt_codes', [])) or 'None'}"
        )
        st.markdown(
            f"**ICD Codes**  \n{', '.join(feature.get('icd_codes', [])) or 'None'}"
        )
        st.markdown(
            f"**HCPCS Codes**  \n{', '.join(feature.get('hcpcs_codes', [])) or 'None'}"
        )
    with col3:
        therapy_weeks = feature.get("therapy_weeks")
        st.markdown(
            f"**Therapy Weeks**  \n{therapy_weeks if therapy_weeks is not None else 'Not specified'}"
        )
        st.markdown(
            f"**Prior Authorization**  \n{feature.get('prior_authorization_required')}"
        )
        st.markdown(
            f"**Confidence**  \n{float(feature.get('confidence', 0.0)):.2f}"
        )

    documentation = feature.get("documentation_required", [])
    exclusions = feature.get("excluded_services", [])
    covered = feature.get("covered_services", [])
    if covered:
        st.markdown("**Covered Services**")
        st.markdown("\n".join(f"- {item}" for item in covered[:5]))
    if exclusions:
        st.markdown("**Excluded Services**")
        st.markdown("\n".join(f"- {item}" for item in exclusions[:5]))
    if documentation:
        st.markdown("**Documentation Required**")
        st.markdown("\n".join(f"- {item}" for item in documentation[:5]))


st.set_page_config(page_title="Feature Extraction", page_icon="FEATURES", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("Feature Extraction")
render_header(
    "Feature Extraction",
    "Extract structured healthcare features from policies, summaries, and rules.",
)

payloads = st.session_state.get("uploaded_documents", [])
if not payloads:
    st.warning("Upload and extract PDFs before feature extraction.")
else:
    try:
        from modules.prompt_manager import PromptManager

        with st.expander("Feature Extraction Prompt Template", expanded=False):
            st.code(PromptManager().get_prompt("feature_extraction"))
    except Exception as error:
        st.info(f"Prompt manager unavailable: {error}")

    source = st.radio("Feature source", ["Documents", "Summary", "Rules"], horizontal=True)
    if st.button("Extract Features", type="primary"):
        try:
            import modules.feature_extractor as feature_extractor_module

            feature_extractor_module = importlib.reload(feature_extractor_module)
            extractor = feature_extractor_module.HealthcareFeatureExtractor()
            with st.spinner("Extracting structured features"):
                if source == "Summary" and st.session_state.get("summary"):
                    feature_record = extractor.extract_features_from_summary(st.session_state["summary"])
                    features = [feature_record]
                elif source == "Rules" and st.session_state.get("rules"):
                    feature_record = extractor.extract_features_from_rules(st.session_state["rules"])
                    features = [feature_record]
                else:
                    features = extractor.extract_features_from_documents(_all_documents(payloads))
                consolidate = getattr(extractor, "consolidate_feature_records", None)
                if callable(consolidate):
                    features = consolidate(features)
            st.session_state["features"] = features
            st.success(f"Extracted {len(features)} feature record(s).")
        except Exception as error:
            st.error(f"Feature extraction failed: {error}")

features = st.session_state.get("features", [])
if features:
    cleaned_features = _post_process_features(features)
    if cleaned_features != features:
        st.session_state["features"] = cleaned_features
        features = cleaned_features
    records_with_icd = sum(1 for item in features if item.get("icd_codes"))
    records_with_cpt = sum(1 for item in features if item.get("cpt_codes"))
    records_with_prior_auth = sum(1 for item in features if item.get("prior_authorization_required") is True)
    avg_confidence = sum(float(item.get("confidence", 0.0)) for item in features) / len(features)
    render_metric_row(
        [
            {"title": "Feature Records", "value": str(len(features)), "delta": "Extracted", "icon": "REC", "color": "#1455a0"},
            {"title": "ICD Present", "value": str(records_with_icd), "delta": "Records", "icon": "ICD", "color": "#047857"},
            {"title": "CPT Present", "value": str(records_with_cpt), "delta": "Records", "icon": "CPT", "color": "#7c3aed"},
            {"title": "Prior Auth", "value": str(records_with_prior_auth), "delta": "Records", "icon": "AUTH", "color": "#b45309"},
        ]
    )
    st.metric("Average Confidence", f"{avg_confidence:.2f}")
    _render_primary_feature_profile(features[0])
    render_table(_feature_summary_frame(features), title="Extracted Features", search=True)
    st.download_button(
        "Download Features JSON",
        json.dumps(features, indent=2),
        file_name="features.json",
        mime="application/json",
    )
    render_json_viewer(features, "Features JSON", expanded=False)
else:
    st.info("Extracted features will appear here.")

render_footer()



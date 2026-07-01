"""ML prediction integration page."""

from __future__ import annotations

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


def _feature_records() -> list[dict[str, Any]]:
    """Return feature records from session state."""
    features = st.session_state.get("features", [])
    if isinstance(features, list):
        return [item for item in features if isinstance(item, dict)]
    if isinstance(features, dict):
        return [features]
    return []


def _to_ml_features(feature: dict[str, Any]) -> dict[str, Any]:
    """Convert extracted features into ML model input features."""
    contract_terms = feature.get("contract_terms", {}) if isinstance(feature.get("contract_terms"), dict) else {}
    rule_results = st.session_state.get("rule_results", [])
    matched = sum(1 for result in rule_results if isinstance(result, dict) and result.get("matched") is True)
    failed = sum(1 for result in rule_results if isinstance(result, dict) and result.get("matched") is not True)
    documentation_complete = feature.get("documentation_complete")
    if documentation_complete is None:
        documentation_complete = bool(feature.get("documentation_required"))

    provider_specialty_match = feature.get("provider_specialty_match")
    if provider_specialty_match is None:
        provider_specialty_match = False

    return {
        "patient_age": feature.get("patient_age") or 0,
        "therapy_weeks": feature.get("therapy_weeks") or 0,
        "prior_authorization_required": bool(feature.get("prior_authorization_required", False)),
        "icd_code_present": bool(feature.get("icd_codes")),
        "cpt_code_present": bool(feature.get("cpt_codes")),
        "contract_allowed_amount": contract_terms.get("allowed_amount") or 0,
        "copay": contract_terms.get("copay") or 0,
        "coinsurance": contract_terms.get("coinsurance") or 0,
        "documentation_complete": bool(documentation_complete),
        "provider_specialty_match": bool(provider_specialty_match),
        "rule_match_count": matched,
        "rule_failure_count": failed,
    }


def _prediction_frame(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    """Create prediction table."""
    return pd.DataFrame(
        [
            {
                "Record": index,
                "Raw Model Probability": item.get("raw_model_probability", 0.0),
                "Approval Probability": item.get("approval_probability", 0.0),
                "Predicted Approval": item.get("predicted_approval", False),
                "Fraud Risk": item.get("fraud_risk", 0.0),
                "Medical Necessity": item.get("medical_necessity_score", 0.0),
                "Confidence": item.get("model_confidence", 0.0),
            }
            for index, item in enumerate(predictions, start=1)
        ]
    )


st.set_page_config(page_title="ML Prediction", page_icon="ML", layout="wide")
apply_theme()
SessionManager.initialize()
render_sidebar("ML Prediction")
render_header(
    "ML Prediction",
    "Train a proof-of-concept model and generate claim approval, fraud risk, and medical necessity signals.",
)

feature_records = _feature_records()
if not feature_records:
    st.warning("Extract features before running ML prediction.")
else:
    num_records = st.slider("Synthetic training records", min_value=100, max_value=2000, value=500, step=100)
    if st.button("Train Model and Predict", type="primary"):
        try:
            from modules.ml_model import HealthcareClaimMLModel

            with st.spinner("Training model on synthetic claims and generating predictions"):
                model = HealthcareClaimMLModel()
                training_df = model.generate_synthetic_dataset(num_records=num_records)
                metrics = model.train_model(training_df)
                ml_inputs = [_to_ml_features(feature) for feature in feature_records]
                predictions = model.predict_batch(ml_inputs)
                feature_importance = {}
                if model.model is not None and hasattr(model.model, "feature_importances_"):
                    feature_importance = dict(zip(model.features_used, model.model.feature_importances_.tolist()))
            st.session_state["ml_prediction"] = predictions[0] if predictions else None
            st.session_state["analytics"]["ml_predictions"] = predictions
            st.session_state["analytics"]["ml_metrics"] = metrics
            st.session_state["analytics"]["feature_importance"] = feature_importance
            st.success("ML prediction completed.")
        except Exception as error:
            st.error(f"ML prediction failed: {error}")

prediction = st.session_state.get("ml_prediction")
predictions = st.session_state.get("analytics", {}).get("ml_predictions", [])
if prediction:
    render_metric_row(
        [
            {"title": "Approval Probability", "value": f"{prediction.get('approval_probability', 0):.2f}", "delta": "Harmonized", "icon": "APP", "color": "#047857"},
            {"title": "Fraud Risk", "value": f"{prediction.get('fraud_risk', 0):.2f}", "delta": "Risk", "icon": "RISK", "color": "#b91c1c"},
            {"title": "Medical Necessity", "value": f"{prediction.get('medical_necessity_score', 0):.2f}", "delta": "Score", "icon": "MED", "color": "#1455a0"},
            {"title": "Confidence", "value": f"{prediction.get('model_confidence', 0):.2f}", "delta": "Model", "icon": "CONF", "color": "#7c3aed"},
        ]
    )
    st.caption(
        f"Raw model probability: {float(prediction.get('raw_model_probability', 0.0)):.2f} | "
        f"Final approval signal is adjusted using fraud risk and medical necessity."
    )
    if predictions:
        render_table(_prediction_frame(predictions), title="Prediction Table", search=False)
    importance = st.session_state.get("analytics", {}).get("feature_importance", {})
    if importance:
        render_table(
            pd.DataFrame([{"Feature": key, "Importance": value} for key, value in importance.items()]),
            title="Feature Importance",
            search=False,
        )
    st.download_button(
        "Download Prediction JSON",
        json.dumps({"primary": prediction, "batch": predictions}, indent=2),
        file_name="ml_prediction.json",
        mime="application/json",
    )
    render_json_viewer({"primary": prediction, "batch": predictions}, "Prediction JSON", expanded=False)
else:
    st.info("ML prediction output will appear here.")

render_footer()



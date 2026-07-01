"""Unit tests for healthcare claim ML model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modules.ml_model import HealthcareClaimMLModel


def _features() -> dict[str, object]:
    """Build sample prediction features.

    Args:
        None.

    Returns:
        Feature dictionary.
    """
    return {
        "patient_age": 55,
        "therapy_weeks": 6,
        "prior_authorization_required": True,
        "icd_code_present": True,
        "cpt_code_present": True,
        "contract_allowed_amount": 750,
        "copay": 50,
        "coinsurance": 0.2,
        "documentation_complete": True,
        "provider_specialty_match": True,
        "rule_match_count": 3,
        "rule_failure_count": 0,
    }


def test_synthetic_dataset_generation() -> None:
    """Verify synthetic dataset generation returns requested row count."""
    model = HealthcareClaimMLModel()

    df = model.generate_synthetic_dataset(num_records=50)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50


def test_required_columns_exist() -> None:
    """Verify synthetic dataset includes all required columns."""
    model = HealthcareClaimMLModel()

    df = model.generate_synthetic_dataset(num_records=10)

    expected_columns = set(model.features_used + ["claim_approved"])
    assert expected_columns.issubset(df.columns)


def test_preprocessing_with_target() -> None:
    """Verify preprocessing returns X and y when target exists."""
    model = HealthcareClaimMLModel()
    df = model.generate_synthetic_dataset(num_records=20)

    X, y = model.preprocess_features(df)

    assert list(X.columns) == model.features_used
    assert len(X) == len(y) == 20


def test_preprocessing_without_target() -> None:
    """Verify preprocessing returns X only when target is absent."""
    model = HealthcareClaimMLModel()

    X = model.preprocess_features(pd.DataFrame([_features()]))

    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == model.features_used


def test_model_training() -> None:
    """Verify model training returns metrics and stores the model."""
    model = HealthcareClaimMLModel(n_estimators=10)
    df = model.generate_synthetic_dataset(num_records=120)

    metrics = model.train_model(df)

    assert model.model is not None
    assert set(metrics) == {"accuracy", "precision", "recall", "f1_score"}
    assert model.training_records == 120


def test_prediction_with_trained_model() -> None:
    """Verify prediction works after training."""
    model = HealthcareClaimMLModel(n_estimators=10)
    model.train_model(model.generate_synthetic_dataset(num_records=120))

    prediction = model.predict(_features())

    assert "error" not in prediction
    assert 0.0 <= prediction["approval_probability"] <= 1.0
    assert isinstance(prediction["predicted_approval"], bool)
    assert 0.0 <= prediction["fraud_risk"] <= 1.0
    assert 0.0 <= prediction["medical_necessity_score"] <= 1.0


def test_prediction_before_training() -> None:
    """Verify prediction before training returns structured error."""
    model = HealthcareClaimMLModel()

    prediction = model.predict(_features())

    assert prediction["error"] == "Model is not trained."
    assert prediction["approval_probability"] == 0.0


def test_batch_prediction() -> None:
    """Verify batch prediction returns one prediction per record."""
    model = HealthcareClaimMLModel(n_estimators=10)
    model.train_model(model.generate_synthetic_dataset(num_records=120))

    predictions = model.predict_batch([_features(), _features()])

    assert len(predictions) == 2
    assert all("approval_probability" in prediction for prediction in predictions)


def test_model_evaluation() -> None:
    """Verify evaluate_model returns metrics and confusion matrix."""
    model = HealthcareClaimMLModel(n_estimators=10)
    df = model.generate_synthetic_dataset(num_records=120)
    X, y = model.preprocess_features(df)
    model.train_model(df)

    metrics = model.evaluate_model(X, y)

    assert "confusion_matrix" in metrics
    assert "accuracy" in metrics


def test_save_and_load_model(tmp_path: Path) -> None:
    """Verify trained model can be saved and loaded."""
    output_path = tmp_path / "claim_model.joblib"
    model = HealthcareClaimMLModel(n_estimators=10)
    model.train_model(model.generate_synthetic_dataset(num_records=120))

    assert model.save_model(output_path) is True

    loaded_model = HealthcareClaimMLModel()
    assert loaded_model.load_model(output_path) is True

    prediction = loaded_model.predict(_features())
    assert "error" not in prediction


def test_model_statistics() -> None:
    """Verify model statistics expose training metadata."""
    model = HealthcareClaimMLModel(n_estimators=10)
    model.train_model(model.generate_synthetic_dataset(num_records=120))

    statistics = model.get_model_statistics()

    assert statistics["model_type"] == "RandomForestClassifier"
    assert statistics["trained"] is True
    assert statistics["training_records"] == 120
    assert statistics["features_used"] == model.features_used


def test_missing_input_fields() -> None:
    """Verify missing input fields are filled with defaults for prediction."""
    model = HealthcareClaimMLModel(n_estimators=10)
    model.train_model(model.generate_synthetic_dataset(num_records=120))

    prediction = model.predict({"patient_age": 55, "therapy_weeks": 6})

    assert "error" not in prediction
    assert 0.0 <= prediction["approval_probability"] <= 1.0


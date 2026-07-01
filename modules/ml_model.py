"""Machine learning model for healthcare claim predictive signals.

This module trains, saves, loads, and runs a lightweight Scikit-learn model
that estimates claim approval probability. It also derives proof-of-concept
fraud risk and medical necessity scores from structured healthcare features.
It does not make final claim decisions.

Example:
    ```python
    from modules.ml_model import HealthcareClaimMLModel

    ml_model = HealthcareClaimMLModel()

    df = ml_model.generate_synthetic_dataset(num_records=500)
    metrics = ml_model.train_model(df)

    features = {
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
        "rule_failure_count": 0
    }

    prediction = ml_model.predict(features)

    print(metrics)
    print(prediction)
    ```
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


class HealthcareClaimMLModel:
    """Trains and serves a healthcare claim prediction model."""

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        n_estimators: int = 100,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the healthcare claim ML model.

        Args:
            random_state: Random seed used for data generation and model
                training.
            test_size: Fraction of data reserved for model testing.
            n_estimators: Number of trees in the RandomForestClassifier.
            logger: Optional logger instance.

        Returns:
            None.

        Raises:
            ValueError: If ``test_size`` or ``n_estimators`` is invalid.
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")
        if n_estimators < 1:
            raise ValueError("n_estimators must be greater than or equal to 1.")

        self.random_state = random_state
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.logger = logger or logging.getLogger(__name__)
        self.model: RandomForestClassifier | None = None
        self.features_used = self._feature_columns()
        self.training_records = 0
        self.last_metrics: dict[str, Any] = {}

    def generate_synthetic_dataset(self, num_records: int = 500) -> pd.DataFrame:
        """Generate a synthetic healthcare claims dataset.

        Args:
            num_records: Number of synthetic claim records to generate.

        Returns:
            Pandas DataFrame containing model features and ``claim_approved``.
            Returns an empty DataFrame if ``num_records`` is invalid.
        """
        if not isinstance(num_records, int) or num_records <= 0:
            self.logger.warning("num_records must be a positive integer.")
            return pd.DataFrame(columns=self._feature_columns() + ["claim_approved"])

        rng = np.random.default_rng(self.random_state)
        df = pd.DataFrame(
            {
                "patient_age": rng.integers(1, 90, size=num_records),
                "therapy_weeks": rng.integers(0, 13, size=num_records),
                "prior_authorization_required": rng.choice([True, False], size=num_records, p=[0.65, 0.35]),
                "icd_code_present": rng.choice([True, False], size=num_records, p=[0.9, 0.1]),
                "cpt_code_present": rng.choice([True, False], size=num_records, p=[0.9, 0.1]),
                "contract_allowed_amount": rng.normal(750, 180, size=num_records).clip(100, 2500),
                "copay": rng.choice([0, 25, 50, 75, 100], size=num_records),
                "coinsurance": rng.choice([0.0, 0.1, 0.2, 0.3], size=num_records),
                "documentation_complete": rng.choice([True, False], size=num_records, p=[0.78, 0.22]),
                "provider_specialty_match": rng.choice([True, False], size=num_records, p=[0.82, 0.18]),
                "rule_match_count": rng.integers(0, 6, size=num_records),
                "rule_failure_count": rng.integers(0, 5, size=num_records),
            }
        )

        approval_score = (
            0.18
            + 0.05 * df["therapy_weeks"].clip(0, 8)
            + 0.12 * df["prior_authorization_required"].astype(int)
            + 0.16 * df["documentation_complete"].astype(int)
            + 0.12 * df["provider_specialty_match"].astype(int)
            + 0.10 * df["icd_code_present"].astype(int)
            + 0.10 * df["cpt_code_present"].astype(int)
            + 0.04 * df["rule_match_count"]
            - 0.16 * df["rule_failure_count"]
            - 0.00004 * (df["contract_allowed_amount"] - 750).clip(lower=0)
        )
        approval_probability = approval_score.clip(0.02, 0.98)
        random_draw = rng.random(num_records)
        df["claim_approved"] = (random_draw < approval_probability).astype(int)

        self.logger.info("Synthetic data generated: %s records.", len(df))
        return df

    def preprocess_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
        """Prepare feature data for Scikit-learn.

        Args:
            df: Input DataFrame containing model features and optionally
                ``claim_approved``.

        Returns:
            ``(X, y)`` when the target label exists; otherwise ``X`` only.

        Raises:
            This method logs invalid input and returns an empty DataFrame rather
            than crashing.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.warning("Empty or invalid DataFrame provided for preprocessing.")
            empty_x = pd.DataFrame(columns=self.features_used)
            return (empty_x, pd.Series(dtype=int)) if isinstance(df, pd.DataFrame) and "claim_approved" in df else empty_x

        prepared = df.copy()
        for column in self.features_used:
            if column not in prepared:
                prepared[column] = self._default_value_for_column(column)

        X = prepared[self.features_used].copy()
        for column in X.columns:
            if X[column].dtype == bool:
                X[column] = X[column].astype(int)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna({column: self._default_value_for_column(column) for column in X.columns})

        if "claim_approved" in prepared:
            y = prepared["claim_approved"].fillna(0).astype(int)
            return X, y
        return X

    def train_model(self, df: pd.DataFrame) -> dict[str, float]:
        """Train a RandomForestClassifier on healthcare claim features.

        Args:
            df: Training DataFrame containing features and ``claim_approved``.

        Returns:
            Dictionary of accuracy, precision, recall, and F1 score. Returns an
            error payload when training cannot proceed.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.error("Cannot train model with an empty dataset.")
            return self._empty_metrics()
        if "claim_approved" not in df:
            self.logger.error("Training dataset is missing claim_approved.")
            return self._empty_metrics()

        self.logger.info("Model training started.")
        processed = self.preprocess_features(df)
        if not isinstance(processed, tuple):
            return self._empty_metrics()
        X, y = processed
        if X.empty or y.empty or y.nunique() < 2:
            self.logger.error("Training data must contain features and both target classes.")
            return self._empty_metrics()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.model.fit(X_train, y_train)
        metrics = self.evaluate_model(X_test, y_test)
        self.training_records = len(df)
        self.last_metrics = {
            key: value for key, value in metrics.items() if key != "confusion_matrix"
        }
        self.logger.info("Model training completed. Metrics: %s", self.last_metrics)
        return self.last_metrics

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict claim approval probability and related ML signals.

        Args:
            features: Single feature dictionary.

        Returns:
            Prediction dictionary containing approval probability,
            predicted approval, fraud risk, medical necessity score, and model
            confidence. Returns a structured error if prediction cannot run.
        """
        self.logger.info("Prediction requested.")
        if self.model is None:
            self.logger.error("Prediction requested before model training.")
            return self._prediction_error("Model is not trained.")
        if not isinstance(features, dict) or not features:
            return self._prediction_error("Features must be a non-empty dictionary.")

        X = self.preprocess_features(pd.DataFrame([features]))
        if isinstance(X, tuple) or X.empty:
            return self._prediction_error("Unable to preprocess features.")

        try:
            probabilities = self.model.predict_proba(X)[0]
            class_labels = list(self.model.classes_)
            positive_index = class_labels.index(1) if 1 in class_labels else int(np.argmax(probabilities))
            raw_model_probability = float(probabilities[positive_index])
            fraud_risk = self._calculate_fraud_risk(features, raw_model_probability)
            medical_necessity_score = self._calculate_medical_necessity_score(features)
            approval_probability = self._harmonize_approval_probability(
                raw_model_probability=raw_model_probability,
                fraud_risk=fraud_risk,
                medical_necessity_score=medical_necessity_score,
                features=features,
            )
            predicted_approval = self._determine_predicted_approval(
                approval_probability=approval_probability,
                fraud_risk=fraud_risk,
                medical_necessity_score=medical_necessity_score,
            )
            model_confidence = self._calculate_model_confidence(
                raw_model_probability=raw_model_probability,
                approval_probability=approval_probability,
                fraud_risk=fraud_risk,
                medical_necessity_score=medical_necessity_score,
            )
            return {
                "approval_probability": round(approval_probability, 2),
                "predicted_approval": bool(predicted_approval),
                "fraud_risk": round(fraud_risk, 2),
                "medical_necessity_score": round(medical_necessity_score, 2),
                "model_confidence": round(model_confidence, 2),
                "raw_model_probability": round(raw_model_probability, 2),
            }
        except Exception as error:
            self.logger.error("Prediction failed: %s", error)
            return self._prediction_error("Prediction failed.")

    def predict_batch(self, feature_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict ML signals for multiple feature records.

        Args:
            feature_records: List of feature dictionaries.

        Returns:
            List of prediction dictionaries.
        """
        if not isinstance(feature_records, list):
            self.logger.error("feature_records must be a list.")
            return []
        return [self.predict(features) for features in feature_records]

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        """Evaluate the trained model.

        Args:
            X_test: Test feature matrix.
            y_test: Test target labels.

        Returns:
            Evaluation metrics including confusion matrix.
        """
        if self.model is None:
            self.logger.error("Cannot evaluate an untrained model.")
            return {**self._empty_metrics(), "confusion_matrix": []}
        if not isinstance(X_test, pd.DataFrame) or X_test.empty:
            return {**self._empty_metrics(), "confusion_matrix": []}

        try:
            predictions = self.model.predict(X_test)
            return {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
                "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 3),
                "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 3),
                "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 3),
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            }
        except Exception as error:
            self.logger.error("Model evaluation failed: %s", error)
            return {**self._empty_metrics(), "confusion_matrix": []}

    def save_model(self, output_path: str | Path) -> bool:
        """Save the trained model using joblib.

        Args:
            output_path: Destination model path.

        Returns:
            ``True`` when saved successfully; otherwise ``False``.
        """
        if self.model is None:
            self.logger.error("Cannot save an untrained model.")
            return False

        path = Path(output_path)
        payload = {
            "model": self.model,
            "features_used": self.features_used,
            "training_records": self.training_records,
            "last_metrics": self.last_metrics,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "n_estimators": self.n_estimators,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, path)
            self.logger.info("Model saved: %s", path)
            return True
        except Exception as error:
            self.logger.error("Failed to save model to %s: %s", path, error)
            return False

    def load_model(self, input_path: str | Path) -> bool:
        """Load a trained model using joblib.

        Args:
            input_path: Source model path.

        Returns:
            ``True`` when loaded successfully; otherwise ``False``.
        """
        path = Path(input_path)
        if not path.exists():
            self.logger.error("Model file does not exist: %s", path)
            return False

        try:
            payload = joblib.load(path)
            if isinstance(payload, dict) and "model" in payload:
                self.model = payload["model"]
                self.features_used = payload.get("features_used", self.features_used)
                self.training_records = int(payload.get("training_records", 0))
                self.last_metrics = payload.get("last_metrics", {})
            else:
                self.model = payload
            self.logger.info("Model loaded: %s", path)
            return True
        except Exception as error:
            self.logger.error("Failed to load model from %s: %s", path, error)
            return False

    def get_model_statistics(self) -> dict[str, Any]:
        """Return model metadata and training statistics.

        Args:
            None.

        Returns:
            Model statistics dictionary.
        """
        return {
            "model_type": "RandomForestClassifier",
            "trained": self.model is not None,
            "features_used": self.features_used,
            "training_records": self.training_records,
            "last_metrics": self.last_metrics,
        }

    def _feature_columns(self) -> list[str]:
        """Return model feature columns.

        Args:
            None.

        Returns:
            Ordered list of model feature names.
        """
        return [
            "patient_age",
            "therapy_weeks",
            "prior_authorization_required",
            "icd_code_present",
            "cpt_code_present",
            "contract_allowed_amount",
            "copay",
            "coinsurance",
            "documentation_complete",
            "provider_specialty_match",
            "rule_match_count",
            "rule_failure_count",
        ]

    def _default_value_for_column(self, column: str) -> int | float:
        """Return default value for a feature column.

        Args:
            column: Feature column name.

        Returns:
            Numeric default value.
        """
        defaults: dict[str, int | float] = {
            "patient_age": 0,
            "therapy_weeks": 0,
            "prior_authorization_required": 0,
            "icd_code_present": 0,
            "cpt_code_present": 0,
            "contract_allowed_amount": 0.0,
            "copay": 0.0,
            "coinsurance": 0.0,
            "documentation_complete": 0,
            "provider_specialty_match": 0,
            "rule_match_count": 0,
            "rule_failure_count": 0,
        }
        return defaults.get(column, 0)

    def _empty_metrics(self) -> dict[str, float]:
        """Return empty model metrics.

        Args:
            None.

        Returns:
            Metrics dictionary with zero values.
        """
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    def _prediction_error(self, message: str) -> dict[str, Any]:
        """Build a structured prediction error payload.

        Args:
            message: Error message.

        Returns:
            Prediction error dictionary.
        """
        return {
            "approval_probability": 0.0,
            "predicted_approval": False,
            "fraud_risk": 0.0,
            "medical_necessity_score": 0.0,
            "model_confidence": 0.0,
            "raw_model_probability": 0.0,
            "error": message,
        }

    def _calculate_fraud_risk(
        self,
        features: dict[str, Any],
        approval_probability: float,
    ) -> float:
        """Calculate a lightweight fraud risk signal.

        Args:
            features: Feature dictionary.
            approval_probability: Model approval probability.

        Returns:
            Fraud risk score between 0 and 1.
        """
        rule_failures = float(features.get("rule_failure_count", 0))
        rule_matches = float(features.get("rule_match_count", 0))
        risk = 0.08
        risk += 0.22 * (1 - int(bool(features.get("icd_code_present", False))))
        risk += 0.22 * (1 - int(bool(features.get("cpt_code_present", False))))
        risk += 0.18 * (1 - int(bool(features.get("documentation_complete", False))))
        risk += 0.12 * (1 - int(bool(features.get("provider_specialty_match", False))))
        risk += 0.08 * min(rule_failures, 4)
        risk -= 0.04 * min(rule_matches, 4)
        risk += 0.10 * max(0.0, 0.55 - approval_probability)
        return float(np.clip(risk, 0.0, 1.0))

    def _calculate_medical_necessity_score(self, features: dict[str, Any]) -> float:
        """Calculate a lightweight medical necessity score.

        Args:
            features: Feature dictionary.

        Returns:
            Medical necessity score between 0 and 1.
        """
        therapy_weeks = float(features.get("therapy_weeks", 0))
        rule_matches = float(features.get("rule_match_count", 0))
        rule_failures = float(features.get("rule_failure_count", 0))
        score = 0.18
        score += 0.22 * min(therapy_weeks, 6) / 6.0
        score += 0.12 * int(bool(features.get("prior_authorization_required", False)))
        score += 0.14 * int(bool(features.get("icd_code_present", False)))
        score += 0.12 * int(bool(features.get("cpt_code_present", False)))
        score += 0.12 * int(bool(features.get("documentation_complete", False)))
        score += 0.06 * int(bool(features.get("provider_specialty_match", False)))
        score += 0.05 * min(rule_matches, 4)
        score -= 0.08 * min(rule_failures, 4)
        return float(np.clip(score, 0.0, 1.0))

    def _harmonize_approval_probability(
        self,
        raw_model_probability: float,
        fraud_risk: float,
        medical_necessity_score: float,
        features: dict[str, Any],
    ) -> float:
        """Combine model probability with risk and necessity into a coherent approval score."""
        approval_score = (
            0.55 * raw_model_probability
            + 0.30 * medical_necessity_score
            + 0.15 * (1.0 - fraud_risk)
        )
        if fraud_risk >= 0.8:
            approval_score -= 0.30
        elif fraud_risk >= 0.65:
            approval_score -= 0.15

        if medical_necessity_score < 0.30:
            approval_score -= 0.20
        elif medical_necessity_score < 0.45:
            approval_score -= 0.10

        if not bool(features.get("documentation_complete", False)):
            approval_score -= 0.12
        if not bool(features.get("icd_code_present", False)):
            approval_score -= 0.10
        if not bool(features.get("cpt_code_present", False)):
            approval_score -= 0.10

        return float(np.clip(approval_score, 0.0, 1.0))

    def _determine_predicted_approval(
        self,
        approval_probability: float,
        fraud_risk: float,
        medical_necessity_score: float,
    ) -> bool:
        """Determine approval prediction using harmonized thresholds."""
        if fraud_risk >= 0.8:
            return False
        if medical_necessity_score < 0.30:
            return False
        return approval_probability >= 0.55

    def _calculate_model_confidence(
        self,
        raw_model_probability: float,
        approval_probability: float,
        fraud_risk: float,
        medical_necessity_score: float,
    ) -> float:
        """Estimate confidence based on signal consistency and decision separation."""
        distance_from_boundary = abs(approval_probability - 0.5) * 2.0
        consistency = 1.0 - abs(raw_model_probability - approval_probability)
        contradiction_penalty = 0.0
        if approval_probability >= 0.55 and fraud_risk >= 0.65:
            contradiction_penalty += 0.20
        if approval_probability >= 0.55 and medical_necessity_score < 0.40:
            contradiction_penalty += 0.20
        if approval_probability < 0.45 and medical_necessity_score > 0.75 and fraud_risk < 0.25:
            contradiction_penalty += 0.10

        confidence = 0.45 + 0.30 * distance_from_boundary + 0.25 * consistency - contradiction_penalty
        return float(np.clip(confidence, 0.0, 1.0))

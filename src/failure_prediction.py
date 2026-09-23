"""
Supervised Failure Prediction Module using Microsoft LightGBM.
Computes calibrated failure probabilities and extracts feature importances for root-cause analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    ALL_MODEL_FEATURES,
    FAILURE_MODEL_PATH,
    FAILURE_ONNX_PATH,
    LIGHTGBM_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


class FailurePredictor:
    """Microsoft LightGBM Failure Classifier with probability calibration and explainability."""

    def __init__(self, params: Optional[dict] = None):
        self.params = params or LIGHTGBM_PARAMS.copy()
        self.model = LGBMClassifier(**self.params)
        self.features: List[str] = ALL_MODEL_FEATURES.copy()
        self.is_fitted: bool = False
        self.evaluation_metrics: Dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "FailurePredictor":
        """Train LightGBM model on training split and evaluate on validation."""
        self.features = X_train.columns.tolist()
        logger.info(
            f"Training Microsoft LightGBM on {len(X_train)} samples with {len(self.features)} features..."
        )

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
        )
        self.is_fitted = True

        if eval_set:
            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)[:, 1]
            self.evaluation_metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_val, y_proba)),
            }
            logger.info(f"Validation Metrics: {self.evaluation_metrics}")

        return self

    def predict_probability(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (failure_probability_pct, failure_prediction).
        failure_probability_pct: Calibrated probability [0.0% to 100.0%].
        failure_prediction: Binary class (0 = Healthy, 1 = Predicted Failure).
        """
        if not self.is_fitted:
            raise RuntimeError("FailurePredictor must be fitted before predict.")

        probas = self.model.predict_proba(X[self.features])[:, 1]
        preds = (probas >= 0.50).astype(int)
        failure_probability_pct = probas * 100.0
        return failure_probability_pct, preds

    def get_prediction_shap_contributions(
        self,
        X: pd.DataFrame,
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """
        Compute native Microsoft LightGBM TreeSHAP feature contributions for each sample.
        Returns top_k (feature, contribution_val) tuples driving the prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to compute SHAP contributions.")

        # pred_contrib=True returns [N_samples, N_features + 1]
        raw_contribs = self.model.predict(X[self.features], pred_contrib=True)
        results: List[List[Tuple[str, float]]] = []

        for row_idx in range(len(X)):
            feat_contribs = raw_contribs[row_idx, :-1]  # Exclude base bias value at -1
            ranked = sorted(
                zip(self.features, feat_contribs),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            results.append([(k, round(float(v), 4)) for k, v in ranked[:top_k]])

        return results

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importance ranking for root-cause diagnosis."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame(
            {"feature": self.features, "importance": importances}
        ).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save(self, filepath: Optional[Path] = None) -> None:
        """Serialize trained model to disk."""
        path = filepath or FAILURE_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "features": self.features,
                "metrics": self.evaluation_metrics,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"LightGBM failure model saved to: {path}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "FailurePredictor":
        """Load serialized model from disk."""
        path = filepath or FAILURE_MODEL_PATH
        if not Path(path).exists():
            raise FileNotFoundError(f"Trained failure model not found at {path}")
        payload = joblib.load(path)
        predictor = cls()
        predictor.model = payload["model"]
        predictor.features = payload["features"]
        predictor.evaluation_metrics = payload["metrics"]
        predictor.is_fitted = payload["is_fitted"]
        logger.info(f"LightGBM failure model loaded from: {path}")
        return predictor


def train_failure_model(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = "machine_failure",
    test_size: float = TEST_SIZE,
    save_path: Optional[Path] = None,
) -> Tuple[FailurePredictor, Dict[str, float]]:
    """
    Time-aware chronological split (first 80% train, last 20% test).
    Trains Microsoft LightGBM and evaluates precision, recall, and ROC-AUC.
    """
    feats = features or ALL_MODEL_FEATURES
    # Sort chronologically by UDI / timestamp to prevent lookahead leakage
    df_sorted = df.sort_values(by="udi").reset_index(drop=True)

    split_idx = int(len(df_sorted) * (1.0 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train, y_train = train_df[feats], train_df[target_col]
    X_test, y_test = test_df[feats], test_df[target_col]

    predictor = FailurePredictor()
    predictor.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    predictor.save(save_path)

    # Log full classification report
    y_pred = predictor.model.predict(X_test)
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))

    return predictor, predictor.evaluation_metrics

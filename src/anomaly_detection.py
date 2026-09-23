"""
Unsupervised Anomaly Detection Module using Isolation Forest.
Detects multivariate abnormal sensor states without requiring failure labels.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import (
    ALL_MODEL_FEATURES,
    ANOMALY_CONTAMINATION,
    ANOMALY_MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Isolation Forest anomaly detection wrapper with calibrated anomaly percentage scoring."""

    def __init__(self, contamination: float = ANOMALY_CONTAMINATION, random_state: int = RANDOM_STATE):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.score_min = -0.5
        self.score_max = 0.5

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        """Fit Isolation Forest on feature matrix."""
        logger.info(f"Fitting Isolation Forest on {len(X)} samples with {X.shape[1]} features...")
        self.model.fit(X)
        self.is_fitted = True

        # Compute empirical score range for min-max scaling to 0..100%
        raw_scores = self.model.decision_function(X)
        self.score_min = float(np.percentile(raw_scores, 0.5))
        self.score_max = float(np.percentile(raw_scores, 99.5))
        logger.info(f"Anomaly score calibration range: [{self.score_min:.4f}, {self.score_max:.4f}]")
        return self

    def predict_anomaly_score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (anomaly_score_pct, is_anomaly).
        anomaly_score_pct: 0% = completely normal baseline, 100% = severe outlier anomaly.
        is_anomaly: 1 if flagged anomalous, 0 otherwise.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted or loaded before calling predict.")

        raw_scores = self.model.decision_function(X)
        # raw_scores: higher is normal, lower is anomalous.
        # Invert and scale to 0..100%:
        # Clip to calibrated min/max
        clipped = np.clip(raw_scores, self.score_min, self.score_max)
        # Invert so higher = more anomalous
        normalized = (self.score_max - clipped) / (self.score_max - self.score_min + 1e-8)
        anomaly_score_pct = np.clip(normalized * 100.0, 0.0, 100.0)

        preds = self.model.predict(X)
        is_anomaly = np.where(preds == -1, 1, 0)
        return anomaly_score_pct, is_anomaly

    def save(self, filepath: Optional[Path] = None) -> None:
        """Serialize trained anomaly model to disk."""
        path = filepath or ANOMALY_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "score_min": self.score_min,
                "score_max": self.score_max,
                "contamination": self.contamination,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"Anomaly model saved to: {path}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "AnomalyDetector":
        """Load serialized model from disk."""
        path = filepath or ANOMALY_MODEL_PATH
        if not Path(path).exists():
            raise FileNotFoundError(f"Trained anomaly model not found at {path}")
        payload = joblib.load(path)
        detector = cls(contamination=payload["contamination"])
        detector.model = payload["model"]
        detector.score_min = payload["score_min"]
        detector.score_max = payload["score_max"]
        detector.is_fitted = payload["is_fitted"]
        logger.info(f"Anomaly model loaded from: {path}")
        return detector


def train_anomaly_model(
    df: pd.DataFrame,
    features: Optional[list] = None,
    save_path: Optional[Path] = None,
) -> AnomalyDetector:
    """Train Isolation Forest on normal operational baseline records."""
    feats = features or ALL_MODEL_FEATURES
    # Train primarily on normal data so it learns normal operating dynamics
    normal_df = df[df["machine_failure"] == 0] if "machine_failure" in df.columns else df
    X = normal_df[feats]

    detector = AnomalyDetector()
    detector.fit(X)
    detector.save(save_path)
    return detector

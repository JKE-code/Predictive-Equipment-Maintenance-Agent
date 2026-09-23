"""
Automated Validation Suite for Predictive Equipment Maintenance Pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.agent import MaintenanceTicket, PredictiveMaintenanceAgent
from src.anomaly_detection import AnomalyDetector
from src.config import ALL_MODEL_FEATURES, RAW_DATA_PATH
from src.data_processing import load_raw_data, synthesize_timestamps, validate_and_clean_data
from src.failure_prediction import FailurePredictor
from src.feature_engineering import compute_physical_indicators, compute_rolling_temporal_features


def test_data_ingestion_and_cleaning():
    """Verify raw data loads, timestamps synthesize chronologically, and zero nulls exist."""
    raw = load_raw_data(RAW_DATA_PATH)
    assert len(raw) == 10000
    assert "udi" in raw.columns
    assert "product_id" in raw.columns

    with_time = synthesize_timestamps(raw)
    assert "timestamp" in with_time.columns
    assert with_time["timestamp"].is_monotonic_increasing

    cleaned = validate_and_clean_data(with_time)
    assert cleaned.isnull().sum().sum() == 0


def test_feature_engineering():
    """Verify physical indicators and rolling window features are computed properly."""
    sample_df = pd.DataFrame(
        {
            "udi": list(range(1, 21)),
            "air_temp_k": [298.0 + (i * 0.1) for i in range(20)],
            "process_temp_k": [308.0 + (i * 0.15) for i in range(20)],
            "rotational_speed_rpm": [1500.0 - (i * 5.0) for i in range(20)],
            "torque_nm": [40.0 + (i * 0.5) for i in range(20)],
            "tool_wear_min": [float(i * 5) for i in range(20)],
        }
    )

    phys = compute_physical_indicators(sample_df)
    assert "temp_diff_k" in phys.columns
    assert "power_watts" in phys.columns
    assert "strain_index" in phys.columns
    assert np.all(phys["temp_diff_k"] > 0)
    assert np.all(phys["power_watts"] > 0)

    temporal = compute_rolling_temporal_features(phys, window_steps=5)
    assert "rolling_mean_process_temp_5" in temporal.columns
    assert "rolling_std_torque_5" in temporal.columns
    assert "torque_slope_5" in temporal.columns
    assert temporal.isnull().sum().sum() == 0


def test_anomaly_detection_model():
    """Verify Isolation Forest anomaly detector trains and scores 0-100%."""
    # Synthetic normal data
    np.random.seed(42)
    data = {feat: np.random.normal(loc=50.0, scale=5.0, size=200) for feat in ALL_MODEL_FEATURES}
    df = pd.DataFrame(data)

    detector = AnomalyDetector(contamination=0.05)
    detector.fit(df)

    scores, is_anom = detector.predict_anomaly_score(df)
    assert len(scores) == 200
    assert len(is_anom) == 200
    assert np.all((scores >= 0.0) & (scores <= 100.0))
    assert set(is_anom).issubset({0, 1})


def test_failure_predictor_probabilities():
    """Verify LightGBM model fits and produces calibrated probabilities."""
    np.random.seed(42)
    data = {feat: np.random.normal(loc=50.0, scale=5.0, size=300) for feat in ALL_MODEL_FEATURES}
    df = pd.DataFrame(data)
    y = np.random.choice([0, 1], size=300, p=[0.9, 0.1])

    predictor = FailurePredictor()
    predictor.fit(df, y)

    probas, preds = predictor.predict_probability(df)
    assert len(probas) == 300
    assert np.all((probas >= 0.0) & (probas <= 100.0))
    assert set(preds).issubset({0, 1})

    feat_imp = predictor.get_feature_importances()
    assert len(feat_imp) == len(ALL_MODEL_FEATURES)


def test_predictive_maintenance_agent_triage():
    """Verify agent health computation and root-cause diagnostic logic."""
    agent = PredictiveMaintenanceAgent()

    # Normal row test
    h_norm, r_norm = agent.compute_health_and_risk(10.0, 15.0)
    assert h_norm >= 80.0
    assert r_norm == "NORMAL"

    # Critical row test
    h_crit, r_crit = agent.compute_health_and_risk(85.0, 90.0)
    assert h_crit <= 25.0
    assert r_crit == "CRITICAL"

    # Heat Dissipation Failure diagnosis check
    hdf_row = pd.Series(
        {
            "product_id": "M14860",
            "product_type": "M",
            "timestamp": "2026-01-01 12:00:00",
            "temp_diff_k": 7.5,
            "rotational_speed_rpm": 1250,
            "power_watts": 4500,
            "tool_wear_min": 50,
            "torque_nm": 40,
            "failure_probability_pct": 85.0,
            "anomaly_score_pct": 75.0,
        }
    )
    ticket = agent.evaluate_reading(hdf_row)
    assert ticket is not None
    assert isinstance(ticket, MaintenanceTicket)
    assert "HDF" in ticket.diagnosed_failure_mode
    assert "radiator" in ticket.recommended_action.lower()


def test_native_treeshap_contributions():
    """Verify Microsoft LightGBM native TreeSHAP computes feature attributions."""
    np.random.seed(42)
    data = {feat: np.random.normal(loc=50.0, scale=5.0, size=50) for feat in ALL_MODEL_FEATURES}
    df = pd.DataFrame(data)
    y = np.random.choice([0, 1], size=50, p=[0.8, 0.2])

    predictor = FailurePredictor()
    predictor.fit(df, y)

    shap_res = predictor.get_prediction_shap_contributions(df.head(5), top_k=3)
    assert len(shap_res) == 5
    assert len(shap_res[0]) == 3
    # Check that each contribution has (feature_name, float_val)
    feat_name, val = shap_res[0][0]
    assert feat_name in ALL_MODEL_FEATURES
    assert isinstance(val, float)


def test_fastapi_endpoints():
    """Verify FastAPI microservice /health and /predict endpoints."""
    from fastapi.testclient import TestClient
    from src.api import app

    with TestClient(app) as client:
        # Test /health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"

        # Test /predict
        payload = {
            "product_id": "M14860",
            "product_type": "M",
            "air_temp_k": 298.1,
            "process_temp_k": 308.6,
            "rotational_speed_rpm": 1551.0,
            "torque_nm": 42.8,
            "tool_wear_min": 10.0,
        }
        res_predict = client.post("/predict", json=payload)
        assert res_predict.status_code == 200
        body = res_predict.json()
        assert "health_score" in body
        assert "failure_probability_pct" in body
        assert "top_contributing_factors" in body
        assert len(body["top_contributing_factors"]) > 0

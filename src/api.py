"""
FastAPI Production Microservice for Predictive Equipment Maintenance Agent.
Provides RESTful IoT endpoints for real-time sensor scoring, TreeSHAP explainability, and work order retrieval.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import PredictiveMaintenanceAgent
from src.anomaly_detection import AnomalyDetector
from src.config import (
    ALL_MODEL_FEATURES,
    ALERTS_LOG_PATH,
    ANOMALY_MODEL_PATH,
    EQUIPMENT_STATUS_PATH,
    FAILURE_MODEL_PATH,
    TELEMETRY_DATA_PATH,
)
from src.failure_prediction import FailurePredictor
from src.feature_engineering import compute_physical_indicators, compute_rolling_temporal_features

from contextlib import asynccontextmanager

# Global model state
anomaly_detector: Optional[AnomalyDetector] = None
failure_predictor: Optional[FailurePredictor] = None
agent: Optional[PredictiveMaintenanceAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global anomaly_detector, failure_predictor, agent
    if ANOMALY_MODEL_PATH.exists():
        anomaly_detector = AnomalyDetector.load(ANOMALY_MODEL_PATH)
    if FAILURE_MODEL_PATH.exists():
        failure_predictor = FailurePredictor.load(FAILURE_MODEL_PATH)
    agent = PredictiveMaintenanceAgent()
    yield

app = FastAPI(
    title="Predictive Equipment Maintenance Agent API",
    description="Enterprise-grade REST microservice for real-time sensor scoring, Microsoft LightGBM failure prediction, TreeSHAP feature attribution, and autonomous agent triage.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorReadingInput(BaseModel):
    product_id: str = Field(default="M14860", description="Equipment / Machine Serial Identifier")
    product_type: str = Field(default="M", description="Product Quality Variant (L, M, H)")
    air_temp_k: float = Field(default=298.1, description="Ambient air temperature in Kelvin")
    process_temp_k: float = Field(default=308.6, description="Machine process temperature in Kelvin")
    rotational_speed_rpm: float = Field(default=1551.0, description="Rotational spindle speed in RPM")
    torque_nm: float = Field(default=42.8, description="Mechanical torque in Nm")
    tool_wear_min: float = Field(default=0.0, description="Cumulative tool wear in minutes")
    udi: Optional[int] = Field(default=1, description="Cycle sequence identifier")
    timestamp: Optional[str] = Field(default="2026-01-01 00:00:00", description="Telemetry timestamp")


class ShapContribution(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    equipment_id: str
    product_type: str
    health_score: float
    failure_probability_pct: float
    anomaly_score_pct: float
    risk_tier: str
    is_failure_predicted: bool
    top_contributing_factors: List[ShapContribution]
    maintenance_ticket: Optional[dict] = None


@app.get("/")
def root():
    return {
        "service": "Predictive Equipment Maintenance Agent API",
        "status": "ONLINE",
        "models": {
            "failure_classifier": "Microsoft LightGBM (native TreeSHAP)",
            "anomaly_detector": "Scikit-Learn Isolation Forest",
        },
        "docs_url": "/docs",
    }


@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "models_loaded": (anomaly_detector is not None and failure_predictor is not None),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sensor_reading(payload: SensorReadingInput):
    if anomaly_detector is None or failure_predictor is None or agent is None:
        raise HTTPException(status_code=503, detail="Models are not yet loaded.")

    # Convert to single-row DataFrame
    data_dict = payload.model_dump()
    df_single = pd.DataFrame([data_dict])

    # Compute physical indicators
    df_phys = compute_physical_indicators(df_single)
    df_feat = compute_rolling_temporal_features(df_phys, window_steps=15)

    X = df_feat[ALL_MODEL_FEATURES]

    # Compute inference
    anom_score_pct, _ = anomaly_detector.predict_anomaly_score(X)
    fail_prob_pct, fail_pred = failure_predictor.predict_probability(X)
    shap_factors = failure_predictor.get_prediction_shap_contributions(X, top_k=5)[0]

    row_series = df_feat.iloc[0].copy()
    row_series["anomaly_score_pct"] = float(anom_score_pct[0])
    row_series["failure_probability_pct"] = float(fail_prob_pct[0])

    health_score, risk_tier = agent.compute_health_and_risk(
        row_series["failure_probability_pct"], row_series["anomaly_score_pct"]
    )
    row_series["health_score"] = health_score
    row_series["risk_tier"] = risk_tier

    ticket = agent.evaluate_reading(row_series, shap_factors=shap_factors)
    ticket_dict = ticket.__dict__ if ticket else None

    return PredictionResponse(
        equipment_id=payload.product_id,
        product_type=payload.product_type,
        health_score=health_score,
        failure_probability_pct=round(float(fail_prob_pct[0]), 2),
        anomaly_score_pct=round(float(anom_score_pct[0]), 2),
        risk_tier=risk_tier,
        is_failure_predicted=bool(fail_pred[0] == 1),
        top_contributing_factors=[
            ShapContribution(feature=k, contribution=v) for k, v in shap_factors
        ],
        maintenance_ticket=ticket_dict,
    )


@app.get("/fleet/status")
def get_fleet_status(limit: int = 50):
    if not EQUIPMENT_STATUS_PATH.exists():
        raise HTTPException(status_code=404, detail="Equipment status matrix not found.")
    df = pd.read_csv(EQUIPMENT_STATUS_PATH)
    return df.head(limit).to_dict(orient="records")


@app.get("/alerts/recent")
def get_recent_alerts(limit: int = 20):
    if not ALERTS_LOG_PATH.exists():
        raise HTTPException(status_code=404, detail="Alerts log not found.")
    df = pd.read_csv(ALERTS_LOG_PATH)
    return df.head(limit).to_dict(orient="records")

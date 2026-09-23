"""
Master Execution Pipeline: Predictive Equipment Maintenance Agent.
Orchestrates Phase 1 through Phase 5:
Data Processing -> Feature Engineering -> ML Training -> Agent Triage -> Export for Dashboards.
"""

import logging
from pathlib import Path
import pandas as pd

from src.agent import PredictiveMaintenanceAgent
from src.alert_engine import AlertEngine
from src.anomaly_detection import train_anomaly_model
from src.config import (
    ALL_MODEL_FEATURES,
    ALERTS_LOG_PATH,
    ANOMALY_MODEL_PATH,
    EQUIPMENT_STATUS_PATH,
    FAILURE_MODEL_PATH,
    RAW_DATA_PATH,
    TELEMETRY_DATA_PATH,
)
from src.data_processing import run_data_pipeline
from src.failure_prediction import train_failure_model
from src.feature_engineering import run_feature_pipeline
from src.sensor_simulator import prepare_simulation_scenarios
from dashboard.powerbi_export_helper import export_powerbi_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Pipeline")


def run_full_pipeline():
    logger.info("=" * 70)
    logger.info("STARTING PREDICTIVE MAINTENANCE PIPELINE")
    logger.info("=" * 70)

    # 1. Data Ingestion & Preprocessing
    logger.info(">>> Phase 1: Data Ingestion & Timestamp Synthesis...")
    df_clean = run_data_pipeline(raw_path=RAW_DATA_PATH, output_path=TELEMETRY_DATA_PATH)

    # 2. Time-Series Feature Engineering
    logger.info(">>> Phase 2: Time-Series Feature Engineering...")
    df_features = run_feature_pipeline(
        input_path=str(TELEMETRY_DATA_PATH),
        output_path=str(TELEMETRY_DATA_PATH),
    )

    # 3. Machine Learning Model Training
    logger.info(">>> Phase 3A: Training Unsupervised Anomaly Detector (Isolation Forest)...")
    anomaly_detector = train_anomaly_model(df_features, save_path=ANOMALY_MODEL_PATH)

    logger.info(">>> Phase 3B: Training Supervised Failure Predictor (Microsoft LightGBM)...")
    failure_predictor, metrics = train_failure_model(df_features, save_path=FAILURE_MODEL_PATH)

    # 4. Model Scoring across Full Telemetry
    logger.info(">>> Computing Full Dataset Inference & Calibration...")
    X_all = df_features[ALL_MODEL_FEATURES]
    anomaly_scores, is_anom = anomaly_detector.predict_anomaly_score(X_all)
    fail_probas, fail_preds = failure_predictor.predict_probability(X_all)

    df_features["anomaly_score_pct"] = anomaly_scores
    df_features["is_anomaly"] = is_anom
    df_features["failure_probability_pct"] = fail_probas
    df_features["predicted_failure"] = fail_preds

    # 5. Agent Autonomous Triage & Work Order Generation
    logger.info(">>> Phase 4: Autonomous Maintenance Agent Triage & Alert Generation...")
    agent = PredictiveMaintenanceAgent()

    health_scores = []
    risk_tiers = []
    for _, row in df_features.iterrows():
        h, r = agent.compute_health_and_risk(
            row["failure_probability_pct"], row["anomaly_score_pct"]
        )
        health_scores.append(h)
        risk_tiers.append(r)

    df_features["health_score"] = health_scores
    df_features["risk_tier"] = risk_tiers

    # Save final enriched telemetry
    df_features.to_csv(TELEMETRY_DATA_PATH, index=False)
    logger.info(f"Saved full scored telemetry to {TELEMETRY_DATA_PATH}")

    # Generate Maintenance Tickets
    tickets = agent.batch_evaluate(df_features)
    alert_engine = AlertEngine()
    alert_engine.log_tickets(tickets)
    alerts_df = alert_engine.export_alerts(ALERTS_LOG_PATH)

    # Export Fleet Equipment Status
    status_df = alert_engine.generate_equipment_status(df_features, EQUIPMENT_STATUS_PATH)

    # 6. Export Tables for Power BI Desktop & Prepare Live Simulation Streams
    logger.info(">>> Phase 5: Exporting Tables for Power BI Desktop & Live Stream Simulator...")
    pbi_files = export_powerbi_tables(df_features, alerts_df)
    sim_scenarios = prepare_simulation_scenarios(df_features)

    # Summary
    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total Telemetry Records: {len(df_features)}")
    logger.info(f"LightGBM Validation Metrics: {metrics}")
    logger.info(f"Generated Maintenance Work Orders: {len(alerts_df)}")
    logger.info(f"Equipment Fleet Count: {len(status_df)}")
    logger.info(f"Power BI Tables: {list(pbi_files.keys())}")
    logger.info(f"Live Simulation Scenarios: {list(sim_scenarios.keys())}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()

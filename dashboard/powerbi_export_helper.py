"""
Power BI Export Helper: Formats and exports clean, relational tables optimized for Power BI Desktop.
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def export_powerbi_tables(
    telemetry_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Generate the 3 clean tables for Power BI Desktop:
    1. Fleet Overview (latest state of each equipment)
    2. Telemetry Trends (time-series sensor history)
    3. Maintenance Alerts Queue (tickets with root causes and recommended actions)
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fleet Overview
    fleet_df = (
        telemetry_df.sort_values(by="udi")
        .groupby("product_id")
        .last()
        .reset_index()
    )
    fleet_cols = [
        "product_id",
        "product_type",
        "timestamp",
        "health_score",
        "failure_probability_pct",
        "anomaly_score_pct",
        "risk_tier",
        "air_temp_k",
        "process_temp_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    fleet_cols = [c for c in fleet_cols if c in fleet_df.columns]
    fleet_file = out_dir / "powerbi_fleet_overview.csv"
    fleet_df[fleet_cols].to_csv(fleet_file, index=False)

    # 2. Telemetry Trends (sample or full time-series)
    trend_cols = [
        "udi",
        "timestamp",
        "product_id",
        "product_type",
        "air_temp_k",
        "process_temp_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temp_diff_k",
        "power_watts",
        "strain_index",
        "failure_probability_pct",
        "anomaly_score_pct",
        "health_score",
        "risk_tier",
        "machine_failure",
    ]
    trend_cols = [c for c in trend_cols if c in telemetry_df.columns]
    trend_file = out_dir / "powerbi_sensor_trends.csv"
    telemetry_df[trend_cols].to_csv(trend_file, index=False)

    # 3. Alerts Queue
    alerts_file = out_dir / "powerbi_alerts_queue.csv"
    alerts_df.to_csv(alerts_file, index=False)

    logger.info(f"Power BI tables exported to {out_dir}")
    return {
        "fleet_overview": fleet_file,
        "sensor_trends": trend_file,
        "alerts_queue": alerts_file,
    }

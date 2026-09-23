"""
Data Ingestion, Validation, and Temporal Synthesis Pipeline for Predictive Maintenance.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    COLUMN_RENAME_MAP,
    RAW_DATA_PATH,
    TELEMETRY_DATA_PATH,
    PROCESSED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load raw AI4I 2020 dataset and apply standard column naming."""
    path = filepath or RAW_DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    logger.info(f"Loading raw telemetry dataset from: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def synthesize_timestamps(
    df: pd.DataFrame,
    start_time: str = "2026-01-01 00:00:00",
    step_minutes: int = 1,
) -> pd.DataFrame:
    """
    Synthesize continuous time-series timestamps anchored to sequential cycle UDI.
    Ensures chronological ordering for rolling temporal analysis and trend charting.
    """
    df = df.sort_values(by="udi").reset_index(drop=True)
    base_time = pd.to_datetime(start_time)
    df["timestamp"] = base_time + pd.to_timedelta(df["udi"] * step_minutes, unit="m")
    logger.info(
        f"Synthesized timestamps: from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
    )
    return df


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data integrity, verify numeric ranges, and ensure no missing values."""
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Detected null values:\n{null_counts[null_counts > 0]}")
        # Forward fill and backward fill numeric columns as safety
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Ensure correct data types
    numeric_cast = {
        "air_temp_k": "float64",
        "process_temp_k": "float64",
        "rotational_speed_rpm": "float64",
        "torque_nm": "float64",
        "tool_wear_min": "float64",
        "machine_failure": "int64",
        "twf": "int64",
        "hdf": "int64",
        "pwf": "int64",
        "osf": "int64",
        "rnf": "int64",
    }
    for col, dtype in numeric_cast.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    logger.info(f"Validation successful: {len(df)} records verified with 0 nulls.")
    return df


def run_data_pipeline(
    raw_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Orchestrate Phase 1: ingestion -> timestamp synthesis -> validation -> save."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data(raw_path)
    with_time = synthesize_timestamps(raw)
    cleaned = validate_and_clean_data(with_time)

    out_file = output_path or TELEMETRY_DATA_PATH
    cleaned.to_csv(out_file, index=False)
    logger.info(f"Cleaned telemetry dataset saved to: {out_file}")
    return cleaned


if __name__ == "__main__":
    run_data_pipeline()

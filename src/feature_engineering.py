"""
Time-Series Feature Engineering and Physical Indicator Extraction.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    ENGINEERED_FEATURES,
    TELEMETRY_DATA_PATH,
)

logger = logging.getLogger(__name__)


def compute_physical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract physics-grounded engineering metrics:
    - Temperature differential (thermal gradient)
    - Mechanical shaft power (Watts)
    - Mechanical overstrain metric (Tool wear * Torque)
    """
    df = df.copy()

    # Thermal gradient between process heat and ambient air
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]

    # Mechanical shaft power (P = omega * Torque = (2*pi*RPM/60) * Torque)
    angular_velocity = (2 * np.pi * df["rotational_speed_rpm"]) / 60.0
    df["power_watts"] = angular_velocity * df["torque_nm"]

    # Overstrain index: cumulative wear multiplied by active torque
    df["strain_index"] = df["tool_wear_min"] * df["torque_nm"]

    return df


def compute_rolling_temporal_features(
    df: pd.DataFrame,
    window_steps: int = 15,
) -> pd.DataFrame:
    """
    Compute rolling time-series statistics:
    - Rolling mean (recent moving baseline)
    - Rolling standard deviation (instability / vibration proxy)
    - Rate of change (trend / slope over 5 intervals)
    """
    df = df.copy()

    # Rolling window statistics for thermal dynamics
    df[f"rolling_mean_process_temp_{window_steps}"] = (
        df["process_temp_k"].rolling(window=window_steps, min_periods=1).mean()
    )
    df[f"rolling_std_process_temp_{window_steps}"] = (
        df["process_temp_k"].rolling(window=window_steps, min_periods=1).std().fillna(0.0)
    )

    # Rolling statistics for rotational speed
    df[f"rolling_mean_speed_{window_steps}"] = (
        df["rotational_speed_rpm"].rolling(window=window_steps, min_periods=1).mean()
    )
    df[f"rolling_std_speed_{window_steps}"] = (
        df["rotational_speed_rpm"].rolling(window=window_steps, min_periods=1).std().fillna(0.0)
    )

    # Rolling statistics for torque (mechanical torque variance proxy)
    df[f"rolling_mean_torque_{window_steps}"] = (
        df["torque_nm"].rolling(window=window_steps, min_periods=1).mean()
    )
    df[f"rolling_std_torque_{window_steps}"] = (
        df["torque_nm"].rolling(window=window_steps, min_periods=1).std().fillna(0.0)
    )

    # Slopes / Rate of change over 5-step interval
    df["torque_slope_5"] = (df["torque_nm"] - df["torque_nm"].shift(5)).fillna(0.0) / 5.0
    df["temp_slope_5"] = (df["process_temp_k"] - df["process_temp_k"].shift(5)).fillna(0.0) / 5.0

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline on telemetry DataFrame."""
    logger.info("Computing physics-based and time-series rolling features...")
    df_phys = compute_physical_indicators(df)
    df_features = compute_rolling_temporal_features(df_phys)

    # Verify all expected engineered features exist
    for feat in ENGINEERED_FEATURES:
        if feat not in df_features.columns:
            raise KeyError(f"Expected engineered feature '{feat}' missing from output.")

    logger.info(f"Engineered {len(ENGINEERED_FEATURES)} temporal & physical features successfully.")
    return df_features


def run_feature_pipeline(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load processed telemetry, compute features, and save back."""
    in_file = input_path or str(TELEMETRY_DATA_PATH)
    df = pd.read_csv(in_file)
    df_feat = engineer_features(df)

    out_file = output_path or in_file
    df_feat.to_csv(out_file, index=False)
    logger.info(f"Updated feature-rich dataset saved to: {out_file}")
    return df_feat


if __name__ == "__main__":
    run_feature_pipeline()

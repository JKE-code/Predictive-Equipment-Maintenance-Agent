"""
Sensor Stream Simulator: Replays historical sensor telemetry as real-time IoT feeds.
Allows live demonstration without requiring physical hardware sensors.
"""

import logging
from pathlib import Path
import time
from typing import Generator, Optional
import pandas as pd

from src.config import (
    SIMULATION_DIR,
    TELEMETRY_DATA_PATH,
)

logger = logging.getLogger(__name__)


def prepare_simulation_scenarios(
    telemetry_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Extract key scenarios from the dataset and save them as simulation streams:
    1. Normal operational stream
    2. Degradation leading to Heat Dissipation Failure (HDF)
    3. Degradation leading to Power Failure (PWF)
    4. Degradation leading to Overstrain Failure (OSF)
    """
    sim_dir = output_dir or SIMULATION_DIR
    sim_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {}

    # Scenario 1: Normal Operation (50 records without failure)
    normal_slice = telemetry_df[telemetry_df["machine_failure"] == 0].iloc[100:160].copy()
    normal_path = sim_dir / "scenario_normal.csv"
    normal_slice.to_csv(normal_path, index=False)
    scenarios["Normal Operation"] = normal_path

    # Scenario 2: Heat Dissipation Failure (HDF) - 40 normal records + failure event
    hdf_failures = telemetry_df[telemetry_df["hdf"] == 1]
    if len(hdf_failures) > 0:
        fail_idx = hdf_failures.index[0]
        start_idx = max(0, fail_idx - 35)
        end_idx = min(len(telemetry_df), fail_idx + 10)
        hdf_slice = telemetry_df.iloc[start_idx:end_idx].copy()
        hdf_path = sim_dir / "scenario_hdf.csv"
        hdf_slice.to_csv(hdf_path, index=False)
        scenarios["Heat Dissipation Failure (HDF)"] = hdf_path

    # Scenario 3: Power Failure (PWF)
    pwf_failures = telemetry_df[telemetry_df["pwf"] == 1]
    if len(pwf_failures) > 0:
        fail_idx = pwf_failures.index[0]
        start_idx = max(0, fail_idx - 35)
        end_idx = min(len(telemetry_df), fail_idx + 10)
        pwf_slice = telemetry_df.iloc[start_idx:end_idx].copy()
        pwf_path = sim_dir / "scenario_pwf.csv"
        pwf_slice.to_csv(pwf_path, index=False)
        scenarios["Power Failure (PWF)"] = pwf_path

    # Scenario 4: Overstrain Failure (OSF)
    osf_failures = telemetry_df[telemetry_df["osf"] == 1]
    if len(osf_failures) > 0:
        fail_idx = osf_failures.index[0]
        start_idx = max(0, fail_idx - 35)
        end_idx = min(len(telemetry_df), fail_idx + 10)
        osf_slice = telemetry_df.iloc[start_idx:end_idx].copy()
        osf_path = sim_dir / "scenario_osf.csv"
        osf_slice.to_csv(osf_path, index=False)
        scenarios["Overstrain Failure (OSF)"] = osf_path

    logger.info(f"Prepared {len(scenarios)} live simulation scenarios in {sim_dir}")
    return scenarios


def stream_scenario(
    scenario_path: Path,
    delay_seconds: float = 0.5,
) -> Generator[pd.Series, None, None]:
    """Yields rows one by one with a simulated time interval."""
    df = pd.read_csv(scenario_path)
    for _, row in df.iterrows():
        yield row
        if delay_seconds > 0:
            time.sleep(delay_seconds)

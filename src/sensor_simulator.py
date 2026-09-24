"""
Sensor Stream Simulator: Replays historical sensor telemetry as real-time IoT feeds.
Allows live demonstration without requiring physical hardware sensors.
"""

import logging
from pathlib import Path
import time
from typing import Generator, Optional
import numpy as np
import pandas as pd

from src.config import (
    SIMULATION_DIR,
    TELEMETRY_DATA_PATH,
)

logger = logging.getLogger(__name__)


def prepare_simulation_scenarios(
    telemetry_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Generate progressive, physics-grounded simulation streams (20 steps each):
    1. Normal operational stream
    2. Progressive Heat Dissipation Failure (HDF)
    3. Progressive Power Failure (PWF)
    4. Progressive Overstrain Failure (OSF)
    5. Progressive Tool Wear Failure (TWF)
    """
    from src.feature_engineering import engineer_features

    sim_dir = output_dir or SIMULATION_DIR
    sim_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {}
    steps = 20

    # Scenario 1: Normal Operation Baseline (20 steps steady nominal)
    normal_raw = pd.DataFrame({
        "udi": range(1, steps + 1),
        "product_id": ["M14860"] * steps,
        "product_type": ["M"] * steps,
        "air_temp_k": np.linspace(298.1, 298.5, steps),
        "process_temp_k": np.linspace(308.3, 308.7, steps),
        "rotational_speed_rpm": np.linspace(1505, 1495, steps),
        "torque_nm": np.linspace(39.5, 40.5, steps),
        "tool_wear_min": np.linspace(25, 45, steps),
        "machine_failure": [0] * steps,
        "twf": [0] * steps,
        "hdf": [0] * steps,
        "pwf": [0] * steps,
        "osf": [0] * steps,
        "rnf": [0] * steps,
    })
    normal_df = engineer_features(normal_raw)
    normal_path = sim_dir / "scenario_normal.csv"
    normal_df.to_csv(normal_path, index=False)
    scenarios["Normal Operation Baseline"] = normal_path

    # Scenario 2: Heat Dissipation Failure (HDF) - Thermal runaway
    # temp_diff collapses below 8.6 K and speed drops below 1380 rpm
    hdf_raw = pd.DataFrame({
        "udi": range(1, steps + 1),
        "product_id": ["L47181"] * steps,
        "product_type": ["L"] * steps,
        "air_temp_k": np.linspace(298.2, 303.8, steps),
        "process_temp_k": np.linspace(308.2, 311.8, steps),
        "rotational_speed_rpm": np.linspace(1510, 1335, steps),
        "torque_nm": np.linspace(39.0, 43.5, steps),
        "tool_wear_min": np.linspace(35, 55, steps),
        "machine_failure": [0] * 15 + [1] * 5,
        "twf": [0] * steps,
        "hdf": [0] * 15 + [1] * 5,
        "pwf": [0] * steps,
        "osf": [0] * steps,
        "rnf": [0] * steps,
    })
    hdf_df = engineer_features(hdf_raw)
    hdf_path = sim_dir / "scenario_hdf.csv"
    hdf_df.to_csv(hdf_path, index=False)
    scenarios["Heat Dissipation Failure (HDF)"] = hdf_path

    # Scenario 3: Power Failure (PWF) - Motor electrical overload
    # Power = Torque * (2*pi*RPM/60) surges past 9,000 W
    pwf_raw = pd.DataFrame({
        "udi": range(1, steps + 1),
        "product_id": ["L47190"] * steps,
        "product_type": ["L"] * steps,
        "air_temp_k": np.linspace(298.0, 298.8, steps),
        "process_temp_k": np.linspace(308.1, 309.2, steps),
        "rotational_speed_rpm": np.linspace(1500, 1590, steps),
        "torque_nm": np.concatenate([np.linspace(38.0, 44.0, 10), np.linspace(48.0, 68.0, 10)]),
        "tool_wear_min": np.linspace(30, 48, steps),
        "machine_failure": [0] * 15 + [1] * 5,
        "twf": [0] * steps,
        "hdf": [0] * steps,
        "pwf": [0] * 15 + [1] * 5,
        "osf": [0] * steps,
        "rnf": [0] * steps,
    })
    pwf_df = engineer_features(pwf_raw)
    pwf_path = sim_dir / "scenario_pwf.csv"
    pwf_df.to_csv(pwf_path, index=False)
    scenarios["Power Failure (PWF)"] = pwf_path

    # Scenario 4: Overstrain Failure (OSF) - Heavy load with worn tool
    # Tool wear * Torque exceeds 11,000 threshold
    osf_raw = pd.DataFrame({
        "udi": range(1, steps + 1),
        "product_id": ["L47205"] * steps,
        "product_type": ["L"] * steps,
        "air_temp_k": np.linspace(298.1, 298.9, steps),
        "process_temp_k": np.linspace(308.2, 309.6, steps),
        "rotational_speed_rpm": np.linspace(1460, 1370, steps),
        "torque_nm": np.concatenate([np.linspace(40.0, 47.0, 10), np.linspace(52.0, 66.0, 10)]),
        "tool_wear_min": np.linspace(170, 215, steps),
        "machine_failure": [0] * 14 + [1] * 6,
        "twf": [0] * steps,
        "hdf": [0] * steps,
        "pwf": [0] * steps,
        "osf": [0] * 14 + [1] * 6,
        "rnf": [0] * steps,
    })
    osf_df = engineer_features(osf_raw)
    osf_path = sim_dir / "scenario_osf.csv"
    osf_df.to_csv(osf_path, index=False)
    scenarios["Overstrain Failure (OSF)"] = osf_path

    # Scenario 5: Tool Wear Failure (TWF) - Tool life exhaustion
    # Tool wear exceeds 200-240 min limit
    twf_raw = pd.DataFrame({
        "udi": range(1, steps + 1),
        "product_id": ["M14910"] * steps,
        "product_type": ["M"] * steps,
        "air_temp_k": np.linspace(298.0, 298.7, steps),
        "process_temp_k": np.linspace(308.0, 309.4, steps),
        "rotational_speed_rpm": np.linspace(1520, 1475, steps),
        "torque_nm": np.linspace(38.0, 48.0, steps),
        "tool_wear_min": np.linspace(185, 240, steps),
        "machine_failure": [0] * 15 + [1] * 5,
        "twf": [0] * 15 + [1] * 5,
        "hdf": [0] * steps,
        "pwf": [0] * steps,
        "osf": [0] * steps,
        "rnf": [0] * steps,
    })
    twf_df = engineer_features(twf_raw)
    twf_path = sim_dir / "scenario_twf.csv"
    twf_df.to_csv(twf_path, index=False)
    scenarios["Tool Wear Failure (TWF)"] = twf_path

    logger.info(f"Prepared {len(scenarios)} progressive simulation scenarios in {sim_dir}")
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

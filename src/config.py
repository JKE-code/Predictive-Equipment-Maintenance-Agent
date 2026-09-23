"""
Configuration parameters and operational constants for Predictive Equipment Maintenance Agent.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ai4i2020.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TELEMETRY_DATA_PATH = PROCESSED_DATA_DIR / "sensor_telemetry.csv"
EQUIPMENT_STATUS_PATH = PROCESSED_DATA_DIR / "equipment_status.csv"
ALERTS_LOG_PATH = PROCESSED_DATA_DIR / "alerts_log.csv"
SIMULATION_DIR = DATA_DIR / "streaming_sim"

MODELS_DIR = BASE_DIR / "models"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.pkl"
FAILURE_MODEL_PATH = MODELS_DIR / "failure_model.pkl"
FAILURE_ONNX_PATH = MODELS_DIR / "failure_model.onnx"

# Column Mappings from raw ai4i2020.csv
COLUMN_RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "product_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

# Raw Telemetry Feature Columns
TELEMETRY_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

# Engineered Features
ENGINEERED_FEATURES = [
    "temp_diff_k",
    "power_watts",
    "strain_index",
    "rolling_mean_process_temp_15",
    "rolling_std_process_temp_15",
    "rolling_mean_speed_15",
    "rolling_std_speed_15",
    "rolling_mean_torque_15",
    "rolling_std_torque_15",
    "torque_slope_5",
    "temp_slope_5",
]

# All ML Model Features
ALL_MODEL_FEATURES = TELEMETRY_FEATURES + ENGINEERED_FEATURES

# Physical Safety and Diagnosis Limits (UCI AI4I 2020 Domain Specification)
HDF_TEMP_DIFF_THRESHOLD = 8.6       # K (HDF triggered if temp_diff < 8.6 K and speed < 1380 rpm)
HDF_SPEED_THRESHOLD = 1380          # rpm
PWF_MIN_WATTS = 3500.0              # W (PWF if mechanical power < 3500 or > 9000 W)
PWF_MAX_WATTS = 9000.0              # W
TWF_WEAR_MINUTES = 200.0            # min (Tool wear failure typical onset at 200-240 min)
OSF_STRAIN_THRESHOLDS = {           # Tool wear [min] * Torque [Nm]
    "L": 11000.0,
    "M": 12000.0,
    "H": 13000.0,
}

# Risk Tier Cutoffs (Failure Probability / Anomaly Score)
RISK_BANDS = {
    "NORMAL": (0.0, 0.25),
    "WATCH": (0.25, 0.50),
    "WARNING": (0.50, 0.75),
    "CRITICAL": (0.75, 1.00),
}

# Model Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
ANOMALY_CONTAMINATION = 0.035       # ~3.4% failure rate in baseline dataset
LIGHTGBM_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "is_unbalance": True,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

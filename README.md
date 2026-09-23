# ⚙️ AI-Powered Predictive Equipment Maintenance Agent

> An end-to-end, Microsoft-first predictive maintenance intelligence platform built for telemetry processing, unsupervised anomaly detection, calibrated failure forecasting, autonomous maintenance triage, and executive reporting.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Microsoft LightGBM](https://img.shields.io/badge/ML-Microsoft%20LightGBM-blue)](https://github.com/microsoft/LightGBM)
[![Power BI Desktop](https://img.shields.io/badge/BI-Microsoft%20Power%20BI-yellow)](https://powerbi.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Problem Statement & Requirements Traceability

**Problem Statement:**  
*Build a predictive maintenance system that analyzes historical sensor data and predicts the probability of equipment failure.*

| CodeAThon Minimum Requirement | Implemented Solution | Component / Code Reference |
| :--- | :--- | :--- |
| **1. Sensor data processing** | Automated ingestion, missing value imputation, and chronological sequence sorting. | [`src/data_processing.py`](file:///d:/Predict_Failure/src/data_processing.py) |
| **2. Time-series analysis** | Physics-derived metrics ($\Delta T$, Shaft Power, Mechanical Strain) and rolling temporal window statistics. | [`src/feature_engineering.py`](file:///d:/Predict_Failure/src/feature_engineering.py) |
| **3. Abnormal pattern detection** | Unsupervised Isolation Forest calibrated to a 0–100% multivariate Anomaly Index. | [`src/anomaly_detection.py`](file:///d:/Predict_Failure/src/anomaly_detection.py) |
| **4. Failure prediction** | Supervised **Microsoft LightGBM** classifier with chronological train/test split. | [`src/failure_prediction.py`](file:///d:/Predict_Failure/src/failure_prediction.py) |
| **5. Failure probability score** | Calibrated `predict_proba()` output providing exact $P(\text{failure}) \in [0, 100\%]$. | [`src/failure_prediction.py`](file:///d:/Predict_Failure/src/failure_prediction.py) |
| **6. Maintenance alert generation** | Autonomous triage agent that isolates root causes and generates prioritized maintenance work orders. | [`src/agent.py`](file:///d:/Predict_Failure/src/agent.py), [`src/alert_engine.py`](file:///d:/Predict_Failure/src/alert_engine.py) |
| **7. Sensor trend visualization** | Power BI Desktop reporting and an interactive real-time telemetry streaming simulator. | [`dashboard/powerbi_guide.md`](file:///d:/Predict_Failure/dashboard/powerbi_guide.md), [`dashboard/demo_runner.py`](file:///d:/Predict_Failure/dashboard/demo_runner.py) |

---

## 2. Architecture & Data Flow

```text
┌────────────────────────────────────────────────────────────────────────┐
│             DATA SOURCE: ai4i2020.csv (10,000 Sensor Cycles)           │
│    (Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear, etc.) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 1: INGESTION & TEMPORAL SYNTHESIS                  │
│    • Continuous 1-minute datetime synthesis anchored to UDI            │
│    • Zero-null validation & dtype standardization                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 2: TIME-SERIES FEATURE ENGINEERING                 │
│    • Thermal gradient: ΔT = Process Temp - Air Temp                    │
│    • Mechanical power: Power = (2π * Speed / 60) * Torque              │
│    • Overstrain index: Tool Wear * Torque                              │
│    • Rolling statistics (Mean & Std Dev over 15-step windows)          │
│    • Rates of change / slopes over 5-step windows                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│   PHASE 3A: ANOMALY DETECTION     │ │  PHASE 3B: FAILURE PREDICTION    │
│   • Scikit-Learn Isolation Forest │ │  • Microsoft LightGBM Classifier │
│   • Calibrated Anomaly Index (0-1)│ │  • Calibrated Failure P(fail) %  │
└─────────────────┬─────────────────┘ └──────────────────┬───────────────┘
                  │                                      │
                  └─────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 4: PREDICTIVE MAINTENANCE AGENT                    │
│    • Health Score = 100 - max(P_fail, Anomaly_score * 0.85)            │
│    • Risk Tiering: NORMAL (0-25%), WATCH, WARNING, CRITICAL (75-100%)  │
│    • Physical Root-Cause Triage: HDF, PWF, OSF, TWF, RNF               │
│    • Maintenance Work Order Dispatch: Prescriptive actions & urgency   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│     POWER BI DESKTOP REPORT       │ │    LIVE STREAMING SIMULATOR UI   │
│   • Fleet Health Overview Cards   │ │   • Streamlit interactive UI     │
│   • Multi-Sensor Trend Visuals    │ │   • Live animated sensor gauges  │
│   • Actionable Work Orders Queue  │ │   • Real-time alert popup cards  │
└───────────────────────────────────┘ └──────────────────────────────────┘
```

---

## 3. Microsoft Ecosystem: 100% Free Tools (Zero Paid Cloud)

This solution utilizes industry-standard tools created and maintained by Microsoft without requiring any paid subscriptions or credit card dependencies:

- **Microsoft LightGBM:** Microsoft's open-source, highly efficient gradient boosting framework powers the core failure prediction model, delivering superior accuracy, fast training, and native probability calibration.
- **Power BI Desktop:** The free Windows desktop edition is used for multi-page fleet reporting, machine drilldowns, and time-series trend analysis.
- **Visual Studio Code:** Primary open-source IDE for Python development, testing, and script execution.
- **Microsoft ONNX Runtime:** Open Neural Network Exchange runtime for high-performance, cross-platform CPU model execution.

---

## 4. Quick Start Guide

### Step 1: Clone & Install Dependencies
```bash
git clone https://github.com/<your-repo>/predictive-maintenance-agent.git
cd predictive-maintenance-agent

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Automated Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

### Step 3: Execute End-to-End Pipeline
```bash
python main.py
```
This single command runs:
1. Ingestion of `data/raw/ai4i2020.csv` and timestamp synthesis.
2. Physics-grounded and rolling time-series feature engineering.
3. Training of the Isolation Forest anomaly detector.
4. Training and evaluation of the Microsoft LightGBM classifier.
5. Autonomous triage and generation of maintenance work orders.
6. Export of Power BI tables (`data/processed/powerbi_*.csv`) and pre-sliced live simulation streams (`data/streaming_sim/`).

### Step 4: Launch Live Interactive Demonstration UI
```bash
python -m streamlit run dashboard/demo_runner.py
```
In the browser:
- Select a failure scenario (e.g. *Heat Dissipation Failure*, *Power Failure*, or *Overstrain Failure*).
- Click **Run Stream** to simulate real-time IoT feeds.
- Watch live gauge movements, dynamic trendlines, native **TreeSHAP feature contribution charts**, and immediate Maintenance Work Order dispatch.

### Step 5: Launch Production REST API Microservice
```bash
uvicorn src.api:app --reload --port 8000
```
- Interactive OpenAPI / Swagger UI: `http://localhost:8000/docs`
- Endpoints:
  - `POST /predict`: Ingest real-time telemetry, get calibrated failure probability, Anomaly Score, TreeSHAP feature attributions, and maintenance work orders.
  - `GET /health`: Microservice health status.
  - `GET /fleet/status`: Current equipment fleet health snapshot.
  - `GET /alerts/recent`: Active work order queue.

### Step 6: Power BI Desktop Reporting
Follow the step-by-step instructions in [`dashboard/powerbi_guide.md`](file:///d:/Predict_Failure/dashboard/powerbi_guide.md) to load the three exported CSV files into Power BI Desktop.

---

## 5. Autonomous Agent Diagnostic Matrix

The agent analyzes physical failure signatures and prescribes concrete technician actions:

| Failure Mode | Physical Trigger Signature | Agent Diagnosis | Prescribed Maintenance Action |
| :--- | :--- | :--- | :--- |
| **HDF** (Heat Dissipation) | $\Delta T < 8.6\text{ K}$ and $\text{Speed} < 1380\text{ RPM}$ | Cooling system degradation or insufficient airflow. | Inspect radiator fins for debris, test coolant pump, and inspect fan motor. |
| **PWF** (Power Failure) | $\text{Power} < 3500\text{ W}$ or $> 9000\text{ W}$ | Mechanical shaft power outside safe envelope $[3.5\text{kW}, 9\text{kW}]$. | Conduct motor inverter electrical inspection, measure phase resistance. |
| **OSF** (Overstrain Failure) | $\text{Tool Wear} \times \text{Torque} > \text{Threshold}$ | Mechanical overstrain on cutting assembly. | Reduce feed rate/cutting depth immediately; replace tooling assembly. |
| **TWF** (Tool Wear Failure) | $\text{Tool Wear} \ge 200\text{ min}$ | Cumulative tool wear exceeded safe threshold. | Schedule cutting tool replacement during next pause. |
| **ANOMALY** | High multivariate anomaly score | Multivariate sensor covariance deviation. | Perform spindle vibration analysis and check drive belt tension. |

---

## 6. Repository Layout

```text
d:/Predict_Failure/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Automated GitHub Actions CI pipeline
├── data/
│   ├── raw/
│   │   ├── ai4i2020.csv              # Primary benchmark dataset (10,000 cycles)
│   │   └── predictive_maintenance_uci.csv # Archived copy
│   ├── processed/
│   │   ├── sensor_telemetry.csv      # Full feature-rich scored telemetry
│   │   ├── equipment_status.csv      # Fleet equipment health matrix
│   │   ├── alerts_log.csv            # Generated maintenance tickets
│   │   ├── powerbi_fleet_overview.csv# Clean table for Power BI fleet visual
│   │   ├── powerbi_sensor_trends.csv # Clean table for Power BI trend visual
│   │   └── powerbi_alerts_queue.csv  # Clean table for Power BI work orders
│   └── streaming_sim/                # Pre-sliced scenario streams for live demo
├── models/
│   ├── anomaly_model.pkl             # Trained Isolation Forest model
│   └── failure_model.pkl             # Trained Microsoft LightGBM model
├── src/
│   ├── __init__.py
│   ├── api.py                        # FastAPI production microservice
│   ├── config.py                     # Physical limits, risk cutoffs, parameters
│   ├── data_processing.py            # Ingestion, validation, timestamp synthesis
│   ├── feature_engineering.py        # Rolling temporal stats & physics metrics
│   ├── anomaly_detection.py          # Isolation Forest anomaly scoring
│   ├── failure_prediction.py         # Microsoft LightGBM classifier & TreeSHAP
│   ├── agent.py                      # Autonomous triage & work order generation
│   ├── alert_engine.py               # Ticket queuing, deduplication, and export
│   └── sensor_simulator.py           # Software sensor stream replay harness
├── dashboard/
│   ├── demo_runner.py                # Streamlit live simulation demo UI with SHAP
│   ├── powerbi_export_helper.py      # Power BI formatting utility
│   └── powerbi_guide.md              # Power BI setup and visual design guide
├── tests/
│   └── test_pipeline.py              # Automated validation test suite
├── main.py                           # Master end-to-end pipeline runner
├── proceedings.md                    # Project roadmap and evaluation document
├── requirements.txt                  # Pinned dependencies
└── README.md                         # Project documentation
```

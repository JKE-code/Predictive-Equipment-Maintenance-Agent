# Project Proceedings: Predictive Equipment Maintenance Agent

**Repository:** `d:/Predict_Failure`  
**Problem Statement:** Build a predictive maintenance system that analyzes historical sensor data and predicts the probability of equipment failure.  
**Constraint:** 100% Free implementation using Microsoft and open-source tools with zero paid cloud dependencies.

---

## 1. Project Understanding & Scope Alignment

### Can We Build the Entire System Without Physical Sensors?
**Yes, 100%.** In industrial data science, hackathons, and enterprise POCs, software systems are designed to consume telemetry through standard data streams (APIs, message queues, or streaming data files). 
- **Physical sensors are NOT required.**
- The software sensor simulator reads historical sensor telemetry sequentially and streams it at 1–2 second intervals, mimicking real IoT telemetry feeds.
- The entire processing pipeline, time-series calculations, ML inference, risk evaluation, autonomous maintenance agent, and Power BI dashboards run on this streaming software harness without any physical hardware.

### Problem Statement Requirements Traceability

| PS Minimum Requirement | Implementation Component | Technical Mechanism |
| :--- | :--- | :--- |
| **1. Sensor data processing** | Data Ingestion & Preprocessing Pipeline | Cleaning, missing value imputation, timestamp sorting, outlier filtering, min-max/robust scaling. |
| **2. Time-series analysis** | Temporal Feature Engineering Module | Rolling window stats (mean, std, min, max), rate of change ($\Delta x / \Delta t$), slope/drift, EWMA (exponential moving average). |
| **3. Abnormal pattern detection** | Unsupervised Anomaly Engine | Dual-layer: Physical operational thresholds + Isolation Forest for multivariate abnormal patterns. |
| **4. Failure prediction** | Supervised Predictive Classifier | Microsoft LightGBM classifier trained on windowed failure indicators. |
| **5. Failure probability score** | Calibrated Probability Engine | Model `predict_proba()` output calibrated to an interpretable 0–100% risk score and machine health index. |
| **6. Maintenance alert generation** | Predictive Maintenance Agent | Automated triage engine: Evaluates risk bands, identifies contributing sensors (root cause), generates priority work orders and mitigation recommendations. |
| **7. Sensor trend visualization** | Power BI Desktop Dashboard + Live Demo UI | Power BI `.pbix` for multi-machine fleet monitoring and sensor trend analysis, plus a real-time playback simulation runner for live judging. |

---

## 2. Dataset Analysis & Selection Decision

Two candidate datasets were uploaded and analyzed in `d:\Predict_Failure`:
1. `ai4i2020.csv` (522 KB, 10,000 records, 14 columns)
2. `predictive_maintenance_uci.csv` (405 KB, 10,000 records, 12 columns)

### Detailed Comparison

| Feature / Attribute | `ai4i2020.csv` | `predictive_maintenance_uci.csv` | Advantage / Impact |
| :--- | :--- | :--- | :--- |
| **Row Count** | 10,000 rows | 10,000 rows | Tied (identical telemetry points) |
| **Failure Distribution** | 339 failures (3.39%), 9,661 normal | 339 failures (3.39%), 9,661 normal | Realistic industrial class imbalance |
| **UDI (Unique Device Index)** | **Present** (1 to 10,000) | **Missing** | **Crucial:** `UDI` provides the chronological sequence order necessary for sequential time-series rolling calculations and playback simulation. |
| **Product ID** | **Present** (e.g. `M14860`, `L47181`, `H29424`) | **Missing** | **Crucial:** Enables multi-asset fleet tracking, machine-level drilldown, and slicers in Power BI. |
| **Type Indicator** | `L` (50%), `M` (30%), `H` (20%) | `L`, `M`, `H` | Machine quality variants (Low, Medium, High). |
| **Telemetry Sensors** | Air Temp [K], Process Temp [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min] | Air temp, Process temp, Rotational speed, Torque, Tool wear | Standard industrial telemetry. Units explicitly defined in `ai4i2020.csv`. |
| **Failure Mode Labels** | 5 distinct modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) | 5 distinct modes | Critical for Agent root-cause diagnosis. |

### Selection Verdict: `ai4i2020.csv` is Superior
`ai4i2020.csv` will be the **primary benchmark dataset**. 
- It preserves the `UDI` sequence and `Product ID`, allowing us to construct continuous time-series timestamps and individual machine histories.
- **Do we need more data?** No. 10,000 records with 339 verified multi-mode failures is the gold standard for predictive maintenance benchmarks (UCI AI4I 2020). It contains rich physical relationships:
  - **HDF (Heat Dissipation Failure):** $\Delta T = \text{Process Temp} - \text{Air Temp} < 8.6\text{ K}$ and $\text{Rotational Speed} < 1380\text{ rpm}$.
  - **PWF (Power Failure):** Power $= \text{Torque} \times \omega$ falls outside $[3500\text{ W}, 9000\text{ W}]$.
  - **OSF (Overstrain Failure):** $\text{Tool Wear} \times \text{Torque}$ exceeds variant thresholds.
  - **TWF (Tool Wear Failure):** Cumulative tool wear exceeds 200–240 minutes.
  - **RNF (Random Failure):** Uncorrelated operational anomalies.

---

## 3. Microsoft Ecosystem: 100% Free Tools Analysis

To ensure **zero out-of-pocket cost and zero unexpected cloud billing**, the project will use tools that are permanently free or open-source from Microsoft:

| Tool / Technology | Microsoft Affiliation | Cost Status | Role in Project |
| :--- | :--- | :--- | :--- |
| **Visual Studio Code** | Microsoft Core Product | **100% Free** | Primary IDE for development, debugging, and notebook exploration. |
| **LightGBM** | Developed & maintained by Microsoft | **100% Free (Open Source)** | Core ML engine for failure prediction and probability scoring. Provides extreme speed, high accuracy, and direct native interpretability. |
| **ONNX Runtime** | Co-developed & maintained by Microsoft | **100% Free (Open Source)** | Model serialization and cross-platform high-performance CPU inference. |
| **Power BI Desktop** | Microsoft Business Intelligence | **100% Free (Local Windows App)** | Executive dashboard, fleet health monitoring, interactive sensor trend visualization. Runs 100% locally with zero subscription required. |
| **GitHub & Codespaces** | Microsoft Subsidiary | **100% Free** | Version control, documentation, public repo presentation. |
| **SQLite / Parquet** | Standard Open Standards | **100% Free** | Embedded local persistence for telemetry, predictions, and generated alerts without paying for Azure SQL or Cosmos DB. |
| *Microsoft Fabric (Optional)* | Microsoft Cloud SaaS | *Free 60-Day Trial (Subject to Work/School Email)* | Can be presented as the target enterprise cloud deployment architecture in the slide deck, while the demo executes locally for 100% reliability. |

> [!NOTE]
> **Why avoid relying exclusively on Microsoft Fabric / Azure cloud services during the live demo?**  
> Fabric trials require an enterprise/school Entra ID (Azure AD) tenant, can expire, and require an active internet connection. By building the core solution on **VS Code + Microsoft LightGBM + Power BI Desktop + ONNX**, the solution is **100% free forever, runs entirely offline, never incurs costs, and showcases Microsoft-developed technology at its core.**

---

## 4. System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│               DATA SOURCE: ai4i2020.csv (10,000 Records)               │
│   (Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear, etc.)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 1: INGESTION & TEMPORAL SYNTHESIS                  │
│   • Map UDI to continuous timestamp sequence (1-min intervals)         │
│   • Missing value audit & unit normalization                           │
│   • Feature validation & train/test chronological splitting            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 2: TIME-SERIES FEATURE ENGINE                      │
│   • Rolling statistics (Mean, Std Dev, Min, Max over 5, 15, 30 steps)  │
│   • Thermal gradient: ΔT = Process Temp - Air Temp                     │
│   • Mechanical Power: Power = (2π * Speed / 60) * Torque               │
│   • Overstrain Index: Tool Wear * Torque                               │
│   • Rate of Change (Spike detection): ΔSensor / Δt                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│   PHASE 3A: ANOMALY DETECTION     │ │  PHASE 3B: FAILURE PREDICTION    │
│   • Physical Safety Thresholds    │ │  • Microsoft LightGBM Classifier │
│   • Isolation Forest Anomaly Score│ │  • Calibrated Probability Output │
│   • Output: Anomaly Index (0-100%)│ │  • Output: P(Failure) (0-100%)   │
└─────────────────┬─────────────────┘ └──────────────────┬───────────────┘
                  │                                      │
                  └─────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 4: PREDICTIVE MAINTENANCE AGENT                    │
│   • Health Score Calculation: Health = 100 - max(P_fail, Anomaly_score)│
│   • Failure Mode Triage: Classifies likely mode (HDF, PWF, OSF, TWF)   │
│   • Root-Cause Attribution: Identifies leading sensor contributors     │
│   • Prescriptive Action Dispatch: Generates actionable work orders     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│    PHASE 5A: POWER BI DASHBOARD   │ │   PHASE 5B: STREAM SIMULATOR UI  │
│   • Fleet Equipment Health Matrix │ │   • Live 1-sec telemetry stream  │
│   • Multi-sensor Trend Charts     │ │   • Moving real-time gauges      │
│   • Alert Log & Equipment Slicer  │ │   • Dynamic alert popup events   │
└───────────────────────────────────┘ └──────────────────────────────────┘
```

---

## 5. Detailed Step-by-Step Implementation Plan

### Phase 1: Data Structuring & Chronological Synthesis
- **Input:** `ai4i2020.csv`.
- **Tasks:**
  1. Organize workspace: Move `ai4i2020.csv` to `data/raw/ai4i2020.csv`.
  2. Synthesize continuous timestamps: Map `UDI = 1..10000` to `Timestamp = 2026-01-01 00:00:00` with 1-minute increments per machine cycle.
  3. Validate zero nulls and correct data types (`float64` for sensors, `int64` for failure flags).
  4. Save standard baseline dataset in `data/processed/sensor_telemetry.parquet` and `sensor_telemetry.csv`.

### Phase 2: Time-Series Feature Engineering
- **Script:** `src/feature_engineering.py`.
- **Derived Features:**
  - `temp_diff`: `Process temperature [K] - Air temperature [K]`
  - `power_watts`: $\frac{2\pi \times \text{Rotational speed}}{60} \times \text{Torque}$
  - `strain_index`: $\text{Tool wear [min]} \times \text{Torque [Nm]}$
  - `rolling_mean_temp_15`: 15-step rolling average of Process Temperature
  - `rolling_std_torque_15`: 15-step rolling standard deviation of Torque (vibration/instability proxy)
  - `rolling_std_speed_15`: 15-step rolling standard deviation of Rotational Speed
  - `torque_slope_5`: 5-step rate of change of Torque
  - `temp_slope_5`: 5-step rate of change of Temperature

### Phase 3: Machine Learning Model Development
- **Anomaly Detection (`src/anomaly_detection.py`):**
  - Train `IsolationForest` on normal operational records (`Machine failure == 0`).
  - Score anomalous observations on unseen data; scale anomaly scores into a clean 0–100% anomaly index.
- **Failure Prediction (`src/failure_prediction.py`):**
  - Train **Microsoft LightGBM (`LGBMClassifier`)** using temporal train/test split (first 80% train, last 20% test).
  - Tune hyper-parameters to address class imbalance (`scale_pos_weight`, `is_unbalance=True`).
  - Evaluate ROC-AUC, Precision, Recall, and PR-AUC.
  - Implement probability calibration using `predict_proba`.
- **Model Packaging:**
  - Save standard scikit-learn/LightGBM model to `models/failure_model.pkl`.
  - Export model to **Microsoft ONNX** format (`models/failure_model.onnx`) and benchmark inference latency with `onnxruntime`.

### Phase 4: Predictive Maintenance Agent & Alert Engine
- **Script:** `src/agent.py` & `src/alert_engine.py`.
- **Triage & Diagnosis Logic:**
  - Risk Classification:
    - `0% - 25%`: NORMAL (Health: 90–100%)
    - `26% - 50%`: WATCH (Health: 70–89%)
    - `51% - 75%`: WARNING (Health: 40–69%)
    - `76% - 100%`: CRITICAL (Health: 0–39%)
  - **Root-Cause Attribution:** Interrogates sensor thresholds to match specific failure modes:
    - If `temp_diff < 8.6` & `speed < 1380`: "Heat Dissipation Failure (HDF) - Coolant loop or ventilation compromised."
    - If `power_watts < 3500` or `power_watts > 9000`: "Power Failure (PWF) - Motor drive overload or electrical defect."
    - If `strain_index > threshold`: "Overstrain Failure (OSF) - Mechanical overload on cutting surface."
    - If `tool_wear > 200`: "Tool Wear Failure (TWF) - Tool past maximum wear threshold."
  - **Action Dispatcher:** Formats structured tickets with Recommended Action, Urgency Level, and Estimated Window to Failure.

### Phase 5: Dashboards & Presentation Demonstrations
- **Power BI Desktop Report (`dashboard/Predictive_Maintenance.pbix`):**
  - Connects to `data/processed/equipment_status.csv` and `data/processed/alerts.csv`.
  - Visual 1: Fleet Overview KPI cards (Active Assets, Fleet Health %, Active Alerts, Critical Failures Predicted).
  - Visual 2: Equipment Slicer (filter by `Product ID` or `Type`).
  - Visual 3: Time-Series Sensor Trend Chart (Process Temp, Torque, Speed over time with failure event markers).
  - Visual 4: Actionable Alert Queue (interactive table with root cause and recommended actions).
- **Interactive Simulation Runner (`dashboard/demo_runner.py`):**
  - Lightweight Streamlit app.
  - Allows the user/judges to press **"Start Telemetry Simulation"**.
  - Replays sensor cycles 1 by 1, updates live gauges, graphs live trends, and automatically raises the Maintenance Alert modal when a failure sequence starts.

### Phase 6: Testing, Documentation, and Delivery
- Test all components with automated test suite (`tests/test_pipeline.py`).
- Create `README.md` with complete architecture diagrams, run commands, and verification steps.
- Provide a summary of the Microsoft technologies used for presentation slides.

---

## 6. Directory Structure

```text
d:/Predict_Failure/
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv              # Primary benchmark dataset
│   ├── processed/
│   │   ├── sensor_telemetry.csv      # With synthesized timestamps & features
│   │   ├── equipment_status.csv      # Current machine health & risk ratings
│   │   └── alerts_log.csv            # Generated maintenance tickets
│   └── streaming_sim/                # Pre-sliced failure sequences for live demo
├── models/
│   ├── anomaly_model.pkl             # Trained Isolation Forest
│   ├── failure_model.pkl             # Trained Microsoft LightGBM model
│   └── failure_model.onnx            # Microsoft ONNX serialized model
├── notebooks/
│   ├── 01_eda_and_processing.ipynb   # Exploratory data analysis
│   └── 02_model_training.ipynb       # Model benchmarks & metrics
├── src/
│   ├── __init__.py
│   ├── config.py                     # Physical limits & model hyperparameters
│   ├── data_processing.py            # Cleaning & timestamp synthesis
│   ├── feature_engineering.py        # Time-series features & rolling metrics
│   ├── anomaly_detection.py          # Isolation Forest scoring
│   ├── failure_prediction.py         # LightGBM training & probability calibration
│   ├── agent.py                      # Autonomous triage & root-cause analysis
│   ├── alert_engine.py               # Ticket generator & action dispatcher
│   └── sensor_simulator.py           # Software sensor stream replay harness
├── dashboard/
│   ├── Predictive_Maintenance.pbix   # Power BI Desktop report
│   └── demo_runner.py                # Interactive Streamlit live simulation UI
├── tests/
│   └── test_pipeline.py              # Automated validation tests
├── Idea_and_basic_Architecture.md     # Reference conceptual doc
├── proceedings.md                    # Active roadmap & execution blueprint
├── requirements.txt                  # Python dependencies
└── README.md                         # Presentation & user guide
```

---

## 7. Immediate Next Steps

1. **Scaffold Directory Structure:** Create folders (`data/raw`, `data/processed`, `models`, `src`, `dashboard`, `tests`).
2. **Move & Prepare Dataset:** Move `ai4i2020.csv` into `data/raw/` and establish `requirements.txt`.
3. **Execute Pipeline:** Implement `data_processing.py` and `feature_engineering.py`.
4. **Train Models:** Train Microsoft LightGBM classifier and Isolation Forest anomaly detector.
5. **Build Agent & Dashboards:** Implement root-cause triage agent, Power BI report data exports, and Streamlit demo runner.

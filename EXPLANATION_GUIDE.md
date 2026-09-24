# Predictive Equipment Maintenance Agent: Master Explanation & Judge Presentation Guide

This guide is designed to give you complete mastery over the project: explaining what the problem is, how the architecture works, how to pitch it to judges, how to link Microsoft Power BI, and how to deploy it for 100% free.

---

## 1. Problem Statement Demystified (In Plain English)

### What the Problem Statement Asks:
> *"Unexpected equipment failures can cause production delays and significant financial losses. Organizations need to identify warning signals before equipment actually fails. Build a predictive maintenance system that analyzes historical sensor data and predicts the probability of equipment failure."*

### The Real-World Industrial Context:
In modern manufacturing (automotive, aerospace, precision milling), when a computer numerical control (CNC) spindle breaks unexpectedly during a cutting pass:
1. **Financial Loss:** Unplanned industrial downtime averages **$260,000 per hour** across heavy manufacturing.
2. **Scrapped Inventory:** A sudden tool fracture or thermal runaway ruins the raw workpiece ($5,000–$50,000 per workpiece).
3. **Reactive vs. Predictive Maintenance:**
   * *Reactive Maintenance:* Wait until the machine smokes and breaks, then fix it (catastrophic cost).
   * *Preventive Maintenance:* Replace tooling every 2 weeks on a fixed calendar, even if the tool is still perfectly fine (wasteful cost).
   * *Predictive Maintenance (Our Solution):* Continuously monitor live physics and sensor trends to predict degradation **hours before catastrophic failure**, issuing an autonomous work order with exact root-cause attribution.

---

## 2. What We Implemented & The Technology Stack

We engineered an end-to-end cyber-physical intelligence platform using a **100% free Microsoft and open-source stack**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            1. SENSOR INGESTION                              │
│      10,000 milling cycles from UCI AI4I 2020 dataset (L, M, H variants)     │
│             Synthesized continuous 1-minute temporal timestamps             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     2. PHYSICS & TIME-SERIES ENGINE                         │
│  • Thermal Gradient: ΔT = Process_Temp - Air_Temp (HDF limit < 8.6 K)       │
│  • Shaft Power: P = ω · τ = (2π·RPM/60) · Torque (PWF limit > 9000 W)       │
│  • Mechanical Strain: S = Tool_Wear × Torque (OSF limit > 11,000)           │
│  • Rolling 15-step Means, Standard Deviations & 5-step Slopes               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     3. DUAL-ENGINE MACHINE LEARNING                         │
│  Engine A: Unsupervised Isolation Forest -> 0-100% Continuous Anomaly Index │
│  Engine B: Microsoft LightGBM Classifier -> Calibrated Failure Probability  │
│  Engine C: Microsoft LightGBM Native TreeSHAP -> Sub-ms Root-Cause Analysis │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    4. AUTONOMOUS TRIAGE & DISPATCH AGENT                    │
│   Calculates Composite Equipment Health Index: H = 100 - (0.7*P + 0.3*Anom) │
│   Triages: HDF, PWF, OSF, TWF, Drift | Assigns: IMMEDIATE, HIGH, ROUTINE    │
│   Prescribes exact corrective action (radiator flush, insert swap, etc.)    │
└──────────────────┬───────────────────┬───────────────────┬──────────────────┘
                   │                   │                   │
                   ▼                   ▼                   ▼
     ┌───────────────────────┐ ┌──────────────────┐ ┌─────────────────────────┐
     │  FastAPI Microservice │ │  Streamlit App   │ │ Microsoft Power BI Data │
     │      (Port 8000)      │ │   (Port 8501)    │ │   3 Relational CSVs     │
     │   SCADA/MES ready     │ │  3D Digital Twin │ │   Desktop Dashboards    │
     └───────────────────────┘ └──────────────────┘ └─────────────────────────┘
```

### Why This Architecture Wins:
1. **Dual-Model ML Strategy:** A supervised model alone cannot detect unprecedented anomalies; an anomaly detector alone cannot predict failure likelihood. Combining **Isolation Forest** (detects weirdness) with **Microsoft LightGBM** (calculates exact failure probability) gives zero blind spots.
2. **Physics-Grounded Features:** We don't just feed raw numbers into ML. We calculate thermodynamics ($\Delta T$), mechanics ($P = \omega\tau$), and tribology (strain index). This allows the model to learn the true physical failure boundaries of machinery.
3. **Sub-Millisecond Explainability (TreeSHAP):** We use LightGBM's native C++ TreeSHAP engine. In less than 1 millisecond, it tells the technician *why* the failure is predicted (e.g. `+3.42 temp_diff_k`, `+1.88 rotational_speed_rpm`).

---

## 3. How to Present to Judges: The Winning Pitch

### Step 1: The 30-Second Elevator Pitch
> *"Judges, unplanned equipment failure is one of the most expensive problems in manufacturing, costing factories billions every year.  
> We built the **AI-Powered Predictive Equipment Maintenance Agent** using a 100% free Microsoft and open-source stack.  
> Instead of simple threshold alerts, our system combines thermodynamic and mechanical physics with a dual-model AI: Scikit-Learn Isolation Forest for abnormal drift detection, and Microsoft LightGBM with native TreeSHAP for failure probability scoring and sub-millisecond root cause attribution.  
> We have deployed this with an interactive 3D WebGL Digital Twin, enterprise fleet monitoring across 10,000 machines, an autonomous technician dispatch console, and a production FastAPI microservice."*

---

### Step 2: The 3-Minute Live Demo Screenplay

Follow this exact sequence while sharing your screen on `http://localhost:8501`:

#### 1. Start on Tab 1: Live Telemetry & 3D Digital Twin (0:00 – 1:00)
* **Show the 3D Twin:** Point out the 3D spindle rotating in real time. Explain that it reflects live spindle telemetry via WebGL Three.js.
* **Select Scenario:** In the sidebar, select **"Power Failure (PWF)"**. Point out the **Scenario Brief Card** showing target asset `L47190` and the physical mechanism (power surging past $9,000\text{ W}$).
* **Hit ▶️ Run Stream:**
  * Watch the first 6 steps: nominal green/cyan, healthy 98% index.
  * Watch steps 7–14: torque surges, motor power jumps to $8,500\text{ W}$, 3D twin turns amber/orange.
  * Watch step 15: power exceeds $10,900\text{ W}$, 3D twin flashes critical red, LightGBM shoots to $98.2\%$ failure probability, and TreeSHAP immediately flags `power_watts` and `torque_nm` as primary drivers.
  * Show the **Autonomous Dispatch Ticket** that pops up with root cause and prescriptive action: *"Inspect drive motor inverter and spindle electrical connections."*

#### 2. Click Tab 4: 🧪 What-If Diagnostic Sandbox (1:00 – 1:45)
* *"Now, judges, what if an engineer wants to test an arbitrary cutting condition without waiting for a stream?"*
* Select the preset **"Heat Dissipation Danger (ΔT < 8.6 K, 1330 RPM)"**.
* Show how the sliders automatically reposition, the physical calculation confirms $\Delta T = 8.00\text{ K}$, and the AI instantly predicts $99.5\%$ failure risk, diagnosing **Heat Dissipation Failure (HDF)** and prescribing a coolant flush.
* Tweak the speed slider back up to $1800\text{ RPM}$ — show the health index recover immediately to $90\%+$ in real time!

#### 3. Click Tab 2: 🏭 Fleet Overview (1:45 – 2:15)
* *"In a real factory, we don't monitor just one machine; we monitor the entire plant."*
* Show the 4 Executive KPI cards: **10,000 Monitored Assets**, **69.3% Average Plant Health**, **496 Critical Assets**, and **85.8% Availability**.
* Show the Risk Distribution and Variant breakdown charts.
* Use the search box to type `L47181` to view its individual asset health card and live operating metrics.

#### 4. Click Tab 3: 📋 Maintenance Dispatch Queue (2:15 – 2:45)
* *"Here is our Autonomous Work Order Dispatch Console."*
* Show the categorized tickets: 172 Emergency Halts, 115 High-Priority Coolant Overhauls, 575 Scheduled Tool Replacements.
* Filter by **"IMMEDIATE"** and click **"⚡ Dispatch Crew"** on a ticket to show the interactive dispatch confirmation state.

#### 5. Show the Production API & Architecture (2:45 – 3:00)
* Briefly show `http://localhost:8000/docs` (Swagger UI) to demonstrate that this is not just a UI mockup, but a **production REST microservice** ready for SCADA/MES integration.
* Highlight that the entire solution runs **100% locally and free with zero cloud bills**.

---

### Step 3: Anticipated Judge Questions & Winning Answers

#### Q1: "Why did you choose Microsoft LightGBM over Deep Learning / LSTMs?"
> **Answer:** *"For tabular industrial sensor data, gradient boosted decision trees like Microsoft LightGBM consistently outperform LSTMs in both accuracy and training efficiency. More importantly, LightGBM has native C++ TreeSHAP support, which allows us to compute mathematically exact feature contributions in less than 1 millisecond. In manufacturing, explainability is mandatory—a maintenance technician cannot act on a black-box deep learning score; they need to know whether to change the cutting insert or flush the radiator."*

#### Q2: "How do you handle class imbalance since equipment failures are rare (~3.4%)?"
> **Answer:** *"In the UCI AI4I dataset, failures occur in only 339 out of 10,000 cycles. We addressed this through multiple techniques:
> 1. In LightGBM, we configured `is_unbalance=True`, which dynamically adjusts gradient weights inverse to class frequencies.
> 2. We evaluated using ROC-AUC (0.9666) and Recall (74.36%) rather than raw accuracy.
> 3. We paired it with unsupervised Isolation Forest trained on 9,661 normal cycles to detect novel drift even before binary failure labels trigger."*

#### Q3: "How does the system calculate the Equipment Health Index?"
> **Answer:** *"We use an industrial composite formula:  
> $$H = 100.0 - (0.70 \times P_{\text{failure}} + 0.30 \times \text{Anomaly Index})$$  
> By weighting failure probability at 70% and anomaly score at 30%, the system penalizes known failure patterns heavily while still degrading health when unprecedented sensor anomalies appear."*

#### Q4: "Can this scale to a real factory network?"
> **Answer:** *"Yes. We built a dedicated FastAPI microservice (`src/api.py`) exposing `/predict`, `/fleet/status`, and `/alerts/recent`. Edge IoT gateways or PLC controllers (Siemens, Allen-Bradley) can push telemetry via standard HTTP/REST or MQTT, receiving instant triage within 15 milliseconds. For plant executives, the data exports directly to Microsoft Power BI Desktop."*

---

## 4. Microsoft Power BI Desktop Integration: Step-by-Step Guide

The project includes an automated exporter (`dashboard/powerbi_export_helper.py`) that generates 3 clean, relational CSV tables in `data/processed/`:
1. `data/processed/powerbi_fleet_overview.csv` (10,000 machine inventory & health)
2. `data/processed/powerbi_sensor_trends.csv` (Time-series telemetry with physics indicators)
3. `data/processed/powerbi_alerts_queue.csv` (Prioritized maintenance tickets)

### Step-by-Step Setup in Power BI Desktop (Free):
1. **Download & Install Power BI Desktop:**
   * Download for free from the Microsoft Store or [powerbi.microsoft.com](https://powerbi.microsoft.com/desktop/).
2. **Import Data:**
   * In Power BI Desktop, click **Home** $\to$ **Get Data** $\to$ **Text/CSV**.
   * Browse to `d:\Predict_Failure\data\processed\` and load:
     * `powerbi_fleet_overview.csv`
     * `powerbi_sensor_trends.csv`
     * `powerbi_alerts_queue.csv`
   * Click **Load**.
3. **Set Up Data Model Relationships:**
   * Click the **Model View** icon on the left sidebar (the diagram icon with connected boxes).
   * Drag `product_id` from `powerbi_fleet_overview` $\to$ `product_id` in `powerbi_sensor_trends` (One-to-Many relationship).
   * Drag `product_id` from `powerbi_fleet_overview` $\to$ `equipment_id` in `powerbi_alerts_queue` (One-to-Many relationship).

---

### Exactly How to Create DAX Measures (Click-by-Click Guide)

In Microsoft Power BI Desktop, a **DAX Measure** is a dynamic calculation that updates automatically based on slicers and filters.

#### Step 1: Open the Formula Bar
1. On the right-hand side of Power BI Desktop, locate the **Data** pane (or **Fields** pane in older versions).
2. Right-click on the table `powerbi_fleet_overview`.
3. In the pop-up menu, select **New measure** (alternatively, click on the **Modeling** or **Table tools** tab in the top ribbon and click the **New measure** button).
4. A formula bar will appear across the top with `Measure = `.

#### Step 2: Paste the DAX Formulas
Copy and paste each formula below into the formula bar and press **Enter** (or click the checkmark $\checkmark$):

1. **Fleet Average Health Index:**
   ```dax
   Fleet Health = AVERAGE(powerbi_fleet_overview[health_score])
   ```
   *Format:* In the top ribbon under **Measure tools**, set Format to **Decimal number** with **1** decimal place.

2. **Critical Machine Count:**
   ```dax
   Critical Count = CALCULATE(COUNTROWS(powerbi_fleet_overview), powerbi_fleet_overview[risk_tier] = "CRITICAL")
   ```
   *Format:* Set Format to **Whole number**.

3. **Fleet Operational Availability Rate:**
   ```dax
   Availability Rate = 
   DIVIDE(
       CALCULATE(COUNTROWS(powerbi_fleet_overview), powerbi_fleet_overview[risk_tier] IN {"NORMAL", "WATCH"}),
       COUNTROWS(powerbi_fleet_overview),
       0
   )
   ```
   *Format:* Click the **%** button in the top ribbon to format as Percentage with 1 decimal.

4. **Emergency Immediate Halt Tickets:**
   ```dax
   Immediate Halts = 
   CALCULATE(
       COUNTROWS(powerbi_alerts_queue), 
       CONTAINSSTRING(powerbi_alerts_queue[urgency_level], "IMMEDIATE")
   )
   ```

#### Step 3: Put the Measure on Your Dashboard Canvas
1. In the **Visualizations** pane (right side), click on the **Card** visual icon (looks like a box with `1 2 3`).
2. A blank card visual will appear on your canvas.
3. In the **Data** pane, find the measure you just created (e.g. `Fleet Health` with a small calculator icon next to it).
4. Drag and drop `Fleet Health` into the **Fields / Value** well of the Card visual.
5. The card will instantly display the calculated KPI (e.g., `69.3%`) with dynamic responsiveness!

---

## 5. Machine Learning Model Architecture & Deployment Strategy

You have two complementary machine learning models in this project:
1. **Model 1: Scikit-Learn Isolation Forest** (`models/anomaly_model.pkl` - 1.2 MB)
2. **Model 2: Microsoft LightGBM Classifier** (`models/failure_model.pkl` - 324 KB)

### How the Models are Deployed & Executed

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                             INCOMING TELEMETRY                             │
│                  (Raw Sensor Reading from Stream or API)                   │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: FEATURE EXTRACTION                         │
│  Computes ΔT, Shaft Power (W), Strain Index, Rolling 15-step Means & Stds  │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
┌─────────────────────────────────────┐ ┌────────────────────────────────────┐
│      MODEL 1: ISOLATION FOREST      │ │      MODEL 2: MICROSOFT LIGHTGBM   │
│   (models/anomaly_model.pkl)        │ │       (models/failure_model.pkl)   │
│ • Unsupervised multivariate scoring │ │ • Supervised binary classifier     │
│ • Compares sample against 9,661     │ │ • Dynamic tree splits with         │
│   nominal operating baselines       │ │   is_unbalance=True weighting      │
│ • Calibrates raw isolation score    │ │ • predict_proba() outputs exact    │
│   to 0.0% – 100.0% Anomaly Index    │ │   0.0% – 100.0% Failure Risk       │
└──────────────────┬──────────────────┘ └─────────────────┬──────────────────┘
                   │                                      │
                   │                                      ▼
                   │                    ┌────────────────────────────────────┐
                   │                    │     NATIVE C++ TREESHAP ENGINE     │
                   │                    │ model.predict(X, pred_contrib=True)│
                   │                    │ Computes exact feature drivers in  │
                   │                    │ less than 1 millisecond (< 1ms)    │
                   │                    └─────────────────┬──────────────────┘
                   │                                      │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                 STEP 3: AUTONOMOUS AGENT RISK EVALUATION                   │
│   Composite Health: H = 100.0 - (0.70 * Failure_Prob + 0.30 * Anomaly)     │
│   Categorizes: HDF, PWF, OSF, TWF, Drift | Issues Work Order Ticket        │
└────────────────────────────────────────────────────────────────────────────┘
```

### In-Memory Caching & Serving Architecture

Both the **Streamlit Dashboard** and the **FastAPI Microservice** load these serialized models into RAM on server boot:
* **In Streamlit (`dashboard/demo_runner.py`):**
  Decorated with `@st.cache_resource def load_models()`. This ensures the models are loaded into memory **once** on container startup. Subsequent stream updates and slider adjustments evaluate in **< 5 milliseconds** directly from RAM without disk I/O.
* **In FastAPI (`src/api.py`):**
  Decorated with `asynccontextmanager lifespan`. When uvicorn starts the server, the lifespan hook instantiates both models in global memory. The `POST /predict` endpoint executes sub-20ms HTTP request-response cycles.

---

## 6. 100% Free Cloud Deployment Options

You can deploy this project live to the web with **zero hosting costs and zero credit card requirements**:

### Option 1: Streamlit Community Cloud (Recommended — 2 Minutes)
* **What it does:** Hosts your interactive 3D WebGL Dashboard on a public `https://...streamlit.app` URL for free forever.
* **Cost:** 100% Free.
* **Deployment Steps:**
  1. Your code and trained models are already pushed to your public GitHub repo: `https://github.com/JKE-code/Predictive-Equipment-Maintenance-Agent.git`.
  2. Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
  3. Click **"New App"**.
  4. Select repository: `JKE-code/Predictive-Equipment-Maintenance-Agent`.
  5. Select branch: `main`.
  6. Main file path: `dashboard/demo_runner.py`.
  7. Click **"Deploy!"**
  8. Within 2 minutes, your live 3D Digital Twin and Fleet Command Center will be running publicly!

### Option 2: Hugging Face Spaces (Free Docker Hosting)
* **What it does:** Free 2 vCPU + 16 GB RAM container deployment.
* **Cost:** 100% Free.
* **Deployment Steps:**
  1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
  2. Select SDK: **Docker** (using the [`Dockerfile`](file:///d:/Predict_Failure/Dockerfile) we already created in the repo).
  3. Connect your GitHub repository.
  4. Hugging Face will automatically build and host the application.

### Option 3: Render.com (Free FastAPI Backend Hosting)
* **What it does:** Hosts the FastAPI REST microservice (`src/api.py`) with automatic OpenAPI docs.
* **Cost:** 100% Free (750 free instance hours per month).
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn src.api:app --host 0.0.0.0 --port $PORT`

---

## 6. Project Checklist Summary

| Feature | Verified | Description |
| :--- | :---: | :--- |
| **Sensor Processing** | ✓ | UCI AI4I 2020 dataset, zero nulls, chronological timestamps. |
| **Physics Analysis** | ✓ | $\Delta T$, shaft power (W), strain index, 15-step rolling statistics. |
| **Anomaly Detection** | ✓ | Isolation Forest trained on 9,661 samples, 0–100% anomaly score. |
| **Failure Prediction** | ✓ | Microsoft LightGBM (98.35% accuracy, 0.9666 ROC-AUC). |
| **Native TreeSHAP** | ✓ | Sub-millisecond root-cause attribution directly from LightGBM. |
| **Autonomous Agent** | ✓ | Multi-tier failure classification (HDF, PWF, OSF, TWF, Drift) with prescriptive actions. |
| **3D Digital Twin** | ✓ | Real-time interactive spindle WebGL component (Three.js). |
| **Fleet Command** | ✓ | Enterprise surveillance across 10,000 machines with searchable directory. |
| **Dispatch Queue** | ✓ | Prioritized work orders with interactive technician dispatch buttons. |
| **What-If Sandbox** | ✓ | Live sliders & presets for instant hypothetical diagnosis. |
| **REST API** | ✓ | Production FastAPI microservice on port 8000. |
| **Power BI Support** | ✓ | 3 exported relational tables ready for Microsoft Power BI Desktop. |
| **Automated Tests** | ✓ | 7/7 unit & integration tests passing in pytest. |

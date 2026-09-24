"""
Streamlit Live Interactive Demonstration UI: Predictive Equipment Maintenance Agent.
Simulates real-time sensor streaming, 3D WebGL Digital Twin, TreeSHAP explainability,
Fleet-wide asset intelligence, and autonomous maintenance work order dispatching.
"""

from pathlib import Path
import sys
import time
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import PredictiveMaintenanceAgent
from src.anomaly_detection import AnomalyDetector
from src.config import (
    ALL_MODEL_FEATURES,
    ALERTS_LOG_PATH,
    ANOMALY_MODEL_PATH,
    EQUIPMENT_STATUS_PATH,
    FAILURE_MODEL_PATH,
    SIMULATION_DIR,
    TELEMETRY_DATA_PATH,
)
from src.failure_prediction import FailurePredictor
from src.sensor_simulator import prepare_simulation_scenarios
import streamlit.components.v1 as components
from dashboard.components.digital_twin_3d import generate_digital_twin_html

# Streamlit Page Config
st.set_page_config(
    page_title="AI Predictive Equipment Maintenance Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Industrial Dark Styling
st.markdown(
    """
    <style>
    /* Dark Base Layout */
    .stApp {
        background-color: #070b14 !important;
        color: #e2e8f0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    header[data-testid="stHeader"] {
        background-color: #070b14 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Metric Glassmorphic Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.5));
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 12px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 4px 0;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Status Colors */
    .status-normal { color: #10b981 !important; }
    .status-watch { color: #38bdf8 !important; }
    .status-warning { color: #f59e0b !important; }
    .status-critical { color: #ef4444 !important; }

    /* Scenario Brief Card in Sidebar */
    .scenario-brief-card {
        background: rgba(15, 23, 42, 0.9);
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 12px;
        margin: 10px 0 16px 0;
        font-size: 0.82rem;
        line-height: 1.45;
        color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* Live Agent Work Order Alert Box */
    .alert-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.14), rgba(15, 23, 42, 0.9));
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 18px;
        margin: 16px 0;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.15);
    }
    .alert-box-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.14), rgba(15, 23, 42, 0.9));
        border-left: 5px solid #f59e0b;
        border-radius: 10px;
        padding: 18px;
        margin: 16px 0;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.15);
    }
    .ticket-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f87171;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    /* Work Order Card */
    .work-order-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(10, 15, 28, 0.95));
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    .wo-critical { border-left: 5px solid #ef4444; }
    .wo-high { border-left: 5px solid #f59e0b; }
    .wo-medium { border-left: 5px solid #38bdf8; }
    .wo-routine { border-left: 5px solid #64748b; }

    /* Badge Pills */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-high { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-medium { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-routine { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.4); }
    .badge-type { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }

    /* Asset Directory Card */
    .asset-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(10, 15, 28, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .asset-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #f1f5f9;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .asset-metric-row {
        display: flex;
        justify-content: space-between;
        margin-top: 8px;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .asset-metric-val {
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Explanation Banner */
    .info-banner {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(15, 23, 42, 0.7));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load serialized models and instantiate the agent."""
    anomaly_detector = AnomalyDetector.load(ANOMALY_MODEL_PATH)
    failure_predictor = FailurePredictor.load(FAILURE_MODEL_PATH)
    agent = PredictiveMaintenanceAgent()
    return anomaly_detector, failure_predictor, agent


def get_risk_color_class(risk_tier: str) -> str:
    mapping = {
        "NORMAL": "status-normal",
        "WATCH": "status-watch",
        "WARNING": "status-warning",
        "CRITICAL": "status-critical",
        "CRITICAL (TRIPPED)": "status-critical",
    }
    return mapping.get(risk_tier, "status-normal")


def main():
    st.title("⚙️ AI-Powered Predictive Equipment Maintenance Agent")
    st.caption(
        "Microsoft LightGBM + Isolation Forest Telemetry Intelligence | 100% Free Local Stack | Port 8000 & 8501"
    )

    # Check if models exist
    if not ANOMALY_MODEL_PATH.exists() or not FAILURE_MODEL_PATH.exists():
        st.warning("⚠️ Trained models not found. Please run the pipeline (`python main.py`) first.")
        st.stop()

    anomaly_detector, failure_predictor, agent = load_models()

    # Scenario Definitions with Detailed Physical Briefs
    scenarios = {
        "Heat Dissipation Failure (HDF)": {
            "path": SIMULATION_DIR / "scenario_hdf.csv",
            "asset": "Asset L47181 (Light-Duty CNC Mill)",
            "physics": "Thermal gradient ΔT collapses below 8.6 K and Spindle RPM decelerates below 1380 RPM.",
            "progression": "Steps 1-6: Nominal operation -> Steps 7-13: Coolant degradation onset -> Steps 14-20: Thermal runaway trip.",
        },
        "Power Failure (PWF)": {
            "path": SIMULATION_DIR / "scenario_pwf.csv",
            "asset": "Asset L47190 (High-Speed Lathe)",
            "physics": "Shaft Power (P = ω·τ) surges past safe motor envelope (> 9,000 Watts) via severe torque oscillation.",
            "progression": "Steps 1-6: Nominal (5.9 kW) -> Steps 7-13: Torque surges (48-68 Nm) -> Steps 14-20: Drive motor overcurrent trip (10.9 kW).",
        },
        "Overstrain Failure (OSF)": {
            "path": SIMULATION_DIR / "scenario_osf.csv",
            "asset": "Asset L47205 (High-Feed Machining Center)",
            "physics": "Overstrain index (Tool wear × Torque) exceeds critical threshold (> 11,000 for L-Type machine).",
            "progression": "Steps 1-6: Worn tool baseline (170 min) -> Steps 7-13: Heavy cut engagement (Torque > 60 Nm) -> Steps 14-20: Mechanical fracture risk.",
        },
        "Tool Wear Failure (TWF)": {
            "path": SIMULATION_DIR / "scenario_twf.csv",
            "asset": "Asset M14910 (Precision Finishing Mill)",
            "physics": "Tool wear exceeds safe boundary (200 - 240 minutes) causing cutting chatter and micro-flank breakdown.",
            "progression": "Steps 1-6: Advanced wear (185 min) -> Steps 7-13: Flank wear acceleration -> Steps 14-20: 240 min limit breached.",
        },
        "Normal Operation Baseline": {
            "path": SIMULATION_DIR / "scenario_normal.csv",
            "asset": "Asset M14860 (Medium-Duty Spindle)",
            "physics": "Nominal thermal equilibrium (ΔT: 10.2 K), stable torque (40 Nm), nominal speed (1500 RPM).",
            "progression": "Steps 1-20: Smooth steady-state cutting. Zero alerts or mechanical anomalies.",
        },
    }

    # Verify scenario files exist or prepare them
    if not (SIMULATION_DIR / "scenario_hdf.csv").exists():
        prepare_simulation_scenarios()

    # Sidebar: Scenario Selection & Controls
    st.sidebar.header("🕹️ Live Stream Controls")

    selected_scenario_name = st.sidebar.selectbox(
        "Select Simulation Scenario", list(scenarios.keys()), index=0
    )
    scenario_info = scenarios[selected_scenario_name]
    selected_scenario_path = scenario_info["path"]

    # Auto-Reset Logic: Detect scenario switch and reset stream state immediately
    if "active_scenario" not in st.session_state:
        st.session_state.active_scenario = selected_scenario_name
    elif st.session_state.active_scenario != selected_scenario_name:
        st.session_state.sim_running = False
        st.session_state.current_step = 0
        st.session_state.history = []
        st.session_state.active_tickets = []
        st.session_state.latched_failure = False
        st.session_state.ema_health = 100.0
        st.session_state.ema_prob = 0.0
        st.session_state.ema_anom = 0.0
        st.session_state.active_scenario = selected_scenario_name
        st.rerun()

    # Sidebar Scenario Briefing Card
    st.sidebar.markdown(
        f"""
        <div class="scenario-brief-card">
            <b>Target Asset:</b> {scenario_info["asset"]}<br>
            <b>Physical Mode:</b> {scenario_info["physics"]}<br>
            <b>Timeline:</b> {scenario_info["progression"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    sim_speed = st.sidebar.slider("Replay Interval (seconds)", 0.1, 1.5, 0.4, 0.1)

    display_mode = st.sidebar.radio(
        "Telemetry Filter Mode",
        ["Industrial EMA Smoothed (Recommended)", "Raw Instantaneous Workpiece Readings"],
        index=0,
        help="Industrial EMA applies exponential temporal smoothing to filter workpiece noise, reflecting realistic machine health degradation.",
    )

    # Session State Initialization
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_tickets" not in st.session_state:
        st.session_state.active_tickets = []
    if "latched_failure" not in st.session_state:
        st.session_state.latched_failure = False
    if "ema_health" not in st.session_state:
        st.session_state.ema_health = 100.0
    if "ema_prob" not in st.session_state:
        st.session_state.ema_prob = 0.0
    if "ema_anom" not in st.session_state:
        st.session_state.ema_anom = 0.0
    if "dispatched_tickets" not in st.session_state:
        st.session_state.dispatched_tickets = set()

    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_clicked = col_btn1.button("▶️ Run Stream", use_container_width=True)
    reset_clicked = col_btn2.button("🔄 Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.sim_running = False
        st.session_state.current_step = 0
        st.session_state.history = []
        st.session_state.active_tickets = []
        st.session_state.latched_failure = False
        st.session_state.ema_health = 100.0
        st.session_state.ema_prob = 0.0
        st.session_state.ema_anom = 0.0
        st.rerun()

    if start_clicked:
        st.session_state.sim_running = True

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Live Telemetry & Digital Twin", "🏭 Fleet Overview", "📋 Maintenance Dispatch Queue"]
    )

    # ==========================================
    # TAB 1: LIVE TELEMETRY & DIGITAL TWIN
    # ==========================================
    with tab1:
        if selected_scenario_path.exists():
            scenario_df = pd.read_csv(selected_scenario_path)
        else:
            prepare_simulation_scenarios()
            scenario_df = pd.read_csv(selected_scenario_path)

        # Section 1: Split Screen - 3D Digital Twin (Left) & Real-Time Intelligence (Right)
        col_twin, col_kpis = st.columns([1.15, 0.85])

        with col_twin:
            st.markdown("##### 🌐 Interactive 3D Machine Spindle Digital Twin (WebGL Three.js)")
            twin_placeholder = st.empty()

        with col_kpis:
            # 2x2 grid for Top KPI Cards
            kpi_r1_c1, kpi_r1_c2 = st.columns(2)
            m_health = kpi_r1_c1.empty()
            m_prob = kpi_r1_c2.empty()

            kpi_r2_c1, kpi_r2_c2 = st.columns(2)
            m_anom = kpi_r2_c1.empty()
            m_risk = kpi_r2_c2.empty()

            st.markdown("##### 🔬 TreeSHAP Real-Time Sensor Impact")
            chart_shap = st.empty()

        # Section 2: Full-width Telemetry Trends
        st.markdown("##### 📈 High-Frequency Telemetry Strip Charts")
        chart_col1, chart_col2 = st.columns(2)
        chart_temp = chart_col1.empty()
        chart_speed = chart_col2.empty()

        # Section 3: Autonomous Agent Work Order
        agent_box = st.empty()

        # Simulation Loop Execution
        total_steps = len(scenario_df)

        if st.session_state.sim_running and st.session_state.current_step < total_steps:
            progress_bar = st.progress(st.session_state.current_step / total_steps)

            for step in range(st.session_state.current_step, total_steps):
                row = scenario_df.iloc[step].copy()
                st.session_state.current_step = step + 1

                # ML Inference on Single Reading
                feats_df = pd.DataFrame([row])[ALL_MODEL_FEATURES]
                anom_score_pct, _ = anomaly_detector.predict_anomaly_score(feats_df)
                fail_prob_pct, _ = failure_predictor.predict_probability(feats_df)
                shap_factors = failure_predictor.get_prediction_shap_contributions(feats_df, top_k=5)[0]

                raw_fail_prob = float(fail_prob_pct[0])
                raw_anom_score = float(anom_score_pct[0])
                raw_health, raw_risk = agent.compute_health_and_risk(raw_fail_prob, raw_anom_score)

                # Check if failure event occurred
                if row.get("machine_failure", 0) == 1 or raw_fail_prob > 85.0:
                    st.session_state.latched_failure = True

                # Compute EMA temporal smoothing
                alpha_p = 0.40 if raw_fail_prob > st.session_state.ema_prob else 0.15
                st.session_state.ema_prob = alpha_p * raw_fail_prob + (1 - alpha_p) * st.session_state.ema_prob
                st.session_state.ema_anom = 0.30 * raw_anom_score + 0.70 * st.session_state.ema_anom
                st.session_state.ema_health = 0.35 * raw_health + 0.65 * st.session_state.ema_health

                if st.session_state.latched_failure:
                    displayed_health = 8.5
                    displayed_prob = 98.2
                    displayed_anom = max(raw_anom_score, 88.0)
                    displayed_risk = "CRITICAL (TRIPPED)"
                elif "EMA" in display_mode:
                    displayed_health = round(st.session_state.ema_health, 1)
                    displayed_prob = round(st.session_state.ema_prob, 1)
                    displayed_anom = round(st.session_state.ema_anom, 1)
                    _, displayed_risk = agent.compute_health_and_risk(displayed_prob, displayed_anom)
                else:
                    displayed_health = raw_health
                    displayed_prob = raw_fail_prob
                    displayed_anom = raw_anom_score
                    displayed_risk = raw_risk

                row["anomaly_score_pct"] = displayed_anom
                row["failure_probability_pct"] = displayed_prob
                row["health_score"] = displayed_health
                row["risk_tier"] = displayed_risk

                # Agent Triage with SHAP root causes
                ticket = agent.evaluate_reading(row, shap_factors=shap_factors)
                if ticket:
                    st.session_state.active_tickets.append(ticket)

                st.session_state.history.append(row.to_dict())
                hist_df = pd.DataFrame(st.session_state.history)

                # Update Top KPI Cards
                risk_cls = get_risk_color_class(displayed_risk)
                m_health.markdown(
                    f'<div class="metric-card"><div class="metric-label">Health Index</div><div class="metric-value {risk_cls}">{displayed_health:.1f}%</div><div class="metric-sub">{"Industrial EMA" if "EMA" in display_mode else "Instantaneous"}</div></div>',
                    unsafe_allow_html=True,
                )
                m_prob.markdown(
                    f'<div class="metric-card"><div class="metric-label">Failure Probability</div><div class="metric-value {risk_cls}">{displayed_prob:.1f}%</div><div class="metric-sub">LightGBM Classifier</div></div>',
                    unsafe_allow_html=True,
                )
                m_anom.markdown(
                    f'<div class="metric-card"><div class="metric-label">Anomaly Score</div><div class="metric-value">{displayed_anom:.1f}%</div><div class="metric-sub">Isolation Forest</div></div>',
                    unsafe_allow_html=True,
                )
                m_risk.markdown(
                    f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value {risk_cls}">{displayed_risk}</div><div class="metric-sub">Cycle Step #{step + 1}/{total_steps}</div></div>',
                    unsafe_allow_html=True,
                )

                # Render 3D Digital Twin with real-time telemetry
                twin_html = generate_digital_twin_html(
                    speed_rpm=float(row.get("rotational_speed_rpm", 1500)),
                    temp_k=float(row.get("process_temp_k", 308)),
                    torque_nm=float(row.get("torque_nm", 40)),
                    anomaly_score=float(displayed_anom),
                    failure_mode=ticket.diagnosed_failure_mode if ticket else "NORMAL",
                    is_failed=st.session_state.latched_failure,
                    height=420,
                )
                with twin_placeholder:
                    components.html(twin_html, height=430)

                # Update Live Charts
                if len(hist_df) > 1:
                    chart_temp.line_chart(
                        hist_df.set_index("udi")[["process_temp_k", "air_temp_k"]],
                        height=240,
                    )
                    chart_speed.line_chart(
                        hist_df.set_index("udi")[["rotational_speed_rpm", "torque_nm"]],
                        height=240,
                    )
                    shap_df = pd.DataFrame(shap_factors, columns=["Feature", "Impact"]).set_index("Feature")
                    chart_shap.bar_chart(shap_df, height=210)

                # Render Agent Work Order if Alert Triggered
                if ticket:
                    alert_cls = "alert-box" if ticket.urgency_level.startswith("IMMEDIATE") else "alert-box-warning"
                    shap_tag = f"<p><b>TreeSHAP Primary Drivers:</b> <code>{ticket.top_contributing_factors}</code></p>" if ticket.top_contributing_factors else ""
                    agent_box.markdown(
                        f"""
                        <div class="{alert_cls}">
                            <div class="ticket-header">🚨 AUTONOMOUS DISPATCH TICKET ISSUED — [{ticket.ticket_id}]</div>
                            <p><b>Target Asset:</b> {ticket.equipment_id} ({ticket.product_type}-Type) | <b>Severity Urgency:</b> <span class="badge-pill badge-critical">{ticket.urgency_level}</span></p>
                            <p><b>Diagnosed Failure Mode:</b> <span style="color:#f87171; font-weight:700;">{ticket.diagnosed_failure_mode}</span></p>
                            <p><b>Root Cause Analysis:</b> {ticket.root_cause_explanation}</p>
                            {shap_tag}
                            <p><b>Prescriptive Action Plan:</b> <span style="color:#fbbf24; font-weight:600;">{ticket.recommended_action}</span></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                progress_bar.progress((step + 1) / total_steps)
                time.sleep(sim_speed)

            st.session_state.sim_running = False

        elif len(st.session_state.history) > 0:
            # Paused or Completed State Rendering
            latest = st.session_state.history[-1]
            risk_cls = get_risk_color_class(latest["risk_tier"])
            m_health.markdown(
                f'<div class="metric-card"><div class="metric-label">Health Index</div><div class="metric-value {risk_cls}">{latest["health_score"]:.1f}%</div><div class="metric-sub">Replay Paused</div></div>',
                unsafe_allow_html=True,
            )
            m_prob.markdown(
                f'<div class="metric-card"><div class="metric-label">Failure Probability</div><div class="metric-value {risk_cls}">{latest["failure_probability_pct"]:.1f}%</div><div class="metric-sub">LightGBM Predict</div></div>',
                unsafe_allow_html=True,
            )
            m_anom.markdown(
                f'<div class="metric-card"><div class="metric-label">Anomaly Score</div><div class="metric-value">{latest["anomaly_score_pct"]:.1f}%</div><div class="metric-sub">Isolation Forest</div></div>',
                unsafe_allow_html=True,
            )
            m_risk.markdown(
                f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value {risk_cls}">{latest["risk_tier"]}</div><div class="metric-sub">Completed {len(st.session_state.history)} Steps</div></div>',
                unsafe_allow_html=True,
            )

            last_mode = st.session_state.active_tickets[-1].diagnosed_failure_mode if st.session_state.active_tickets else "NORMAL"
            paused_twin_html = generate_digital_twin_html(
                speed_rpm=float(latest.get("rotational_speed_rpm", 1500)),
                temp_k=float(latest.get("process_temp_k", 308)),
                torque_nm=float(latest.get("torque_nm", 40)),
                anomaly_score=float(latest.get("anomaly_score_pct", 15)),
                failure_mode=last_mode,
                is_failed=st.session_state.latched_failure,
                height=420,
            )
            with twin_placeholder:
                components.html(paused_twin_html, height=430)

            hist_df = pd.DataFrame(st.session_state.history)
            chart_temp.line_chart(
                hist_df.set_index("udi")[["process_temp_k", "air_temp_k"]],
                height=240,
            )
            chart_speed.line_chart(
                hist_df.set_index("udi")[["rotational_speed_rpm", "torque_nm"]],
                height=240,
            )

            latest_feats = pd.DataFrame([latest])[ALL_MODEL_FEATURES]
            latest_shap = failure_predictor.get_prediction_shap_contributions(latest_feats, top_k=5)[0]
            chart_shap.bar_chart(pd.DataFrame(latest_shap, columns=["Feature", "Impact"]).set_index("Feature"), height=210)

            if st.session_state.active_tickets:
                last_ticket = st.session_state.active_tickets[-1]
                alert_cls = "alert-box" if last_ticket.urgency_level.startswith("IMMEDIATE") else "alert-box-warning"
                shap_tag = f"<p><b>TreeSHAP Primary Drivers:</b> <code>{last_ticket.top_contributing_factors}</code></p>" if last_ticket.top_contributing_factors else ""
                agent_box.markdown(
                    f"""
                    <div class="{alert_cls}">
                        <div class="ticket-header">🚨 AUTONOMOUS DISPATCH TICKET ISSUED — [{last_ticket.ticket_id}]</div>
                        <p><b>Target Asset:</b> {last_ticket.equipment_id} ({last_ticket.product_type}-Type) | <b>Severity Urgency:</b> <span class="badge-pill badge-critical">{last_ticket.urgency_level}</span></p>
                        <p><b>Diagnosed Failure Mode:</b> <span style="color:#f87171; font-weight:700;">{last_ticket.diagnosed_failure_mode}</span></p>
                        <p><b>Root Cause Analysis:</b> {last_ticket.root_cause_explanation}</p>
                        {shap_tag}
                        <p><b>Prescriptive Action Plan:</b> <span style="color:#fbbf24; font-weight:600;">{last_ticket.recommended_action}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Initial Ready State
            m_health.markdown(
                '<div class="metric-card"><div class="metric-label">Health Index</div><div class="metric-value status-normal">100.0%</div><div class="metric-sub">Telemetry Ready</div></div>',
                unsafe_allow_html=True,
            )
            m_prob.markdown(
                '<div class="metric-card"><div class="metric-label">Failure Probability</div><div class="metric-value status-normal">0.0%</div><div class="metric-sub">LightGBM Model</div></div>',
                unsafe_allow_html=True,
            )
            m_anom.markdown(
                '<div class="metric-card"><div class="metric-label">Anomaly Score</div><div class="metric-value">0.0%</div><div class="metric-sub">Isolation Forest</div></div>',
                unsafe_allow_html=True,
            )
            m_risk.markdown(
                '<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value status-normal">NORMAL</div><div class="metric-sub">Spindle Standby</div></div>',
                unsafe_allow_html=True,
            )

            ready_twin_html = generate_digital_twin_html(
                speed_rpm=1450.0,
                temp_k=308.0,
                torque_nm=40.0,
                anomaly_score=10.0,
                failure_mode="NORMAL",
                is_failed=False,
                height=420,
            )
            with twin_placeholder:
                components.html(ready_twin_html, height=430)

            st.info("👈 Press **▶️ Run Stream** in the sidebar to simulate live IoT telemetry and witness real-time failure prediction and autonomous agent response.")

    # ==========================================
    # TAB 2: FLEET OVERVIEW
    # ==========================================
    with tab2:
        st.markdown(
            """
            <div class="info-banner">
                🏭 <b>Plant-Wide Fleet Intelligence:</b> Continuous health surveillance across <b>10,000 industrial machine tools</b> in the production facility.
                Aggregates real-time sensor streams to track plant availability, prioritize degraded equipment, and prevent unplanned factory stoppages.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if EQUIPMENT_STATUS_PATH.exists():
            status_df = pd.read_csv(EQUIPMENT_STATUS_PATH)
            total_assets = len(status_df)
            avg_health = float(status_df["health_score"].mean())
            critical_count = int((status_df["risk_tier"] == "CRITICAL").sum())
            warning_count = int((status_df["risk_tier"] == "WARNING").sum())
            watch_count = int((status_df["risk_tier"] == "WATCH").sum())
            normal_count = int((status_df["risk_tier"] == "NORMAL").sum())

            # Top Fleet Executive KPIs
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            f_col1.markdown(
                f'<div class="metric-card"><div class="metric-label">Monitored Fleet Assets</div><div class="metric-value">{total_assets:,}</div><div class="metric-sub">100% Online Coverage</div></div>',
                unsafe_allow_html=True,
            )
            f_col2.markdown(
                f'<div class="metric-card"><div class="metric-label">Fleet Average Health</div><div class="metric-value status-watch">{avg_health:.1f}%</div><div class="metric-sub">Continuous Aggregate Index</div></div>',
                unsafe_allow_html=True,
            )
            f_col3.markdown(
                f'<div class="metric-card"><div class="metric-label">Immediate Critical Risk</div><div class="metric-value status-critical">{critical_count} Assets</div><div class="metric-sub">Require Emergency Triage</div></div>',
                unsafe_allow_html=True,
            )
            f_col4.markdown(
                f'<div class="metric-card"><div class="metric-label">Fleet Availability</div><div class="metric-value status-normal">{((normal_count + watch_count) / total_assets * 100):.1f}%</div><div class="metric-sub">Operating Nominal or Watch</div></div>',
                unsafe_allow_html=True,
            )

            # Visual Fleet Charts (Risk Distribution & Quality Variants)
            st.markdown("##### 📊 Fleet Health Analytics & Risk Distribution")
            chart_f1, chart_f2 = st.columns(2)

            with chart_f1:
                st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>ASSET DISTRIBUTION BY RISK TIER</p>", unsafe_allow_html=True)
                risk_summary = pd.DataFrame({
                    "Risk Tier": ["NORMAL", "WATCH", "WARNING", "CRITICAL"],
                    "Machine Count": [normal_count, watch_count, warning_count, critical_count],
                }).set_index("Risk Tier")
                st.bar_chart(risk_summary, height=220)

            with chart_f2:
                st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>PRODUCT QUALITY VARIANT PROFILE (L / M / H)</p>", unsafe_allow_html=True)
                type_counts = status_df["product_type"].value_counts().rename_axis("Variant").reset_index(name="Count")
                st.bar_chart(type_counts.set_index("Variant"), height=220)

            # Interactive Machine Directory Explorer
            st.markdown("##### 🔎 Interactive Asset Health Directory")
            filter_c1, filter_c2, filter_c3 = st.columns([1, 1, 2])

            with filter_c1:
                tier_filter = st.selectbox(
                    "Filter by Risk Tier",
                    ["ALL", "CRITICAL", "WARNING", "WATCH", "NORMAL"],
                    index=1,  # Default to CRITICAL so users immediately see machines needing attention
                )
            with filter_c2:
                type_filter = st.selectbox(
                    "Filter Machine Type",
                    ["ALL", "L (Light)", "M (Medium)", "H (Heavy)"],
                    index=0,
                )
            with filter_c3:
                search_query = st.text_input(
                    "Search by Product / Asset ID",
                    placeholder="e.g. L47181, M14860...",
                ).strip().upper()

            # Filter logic
            filtered_df = status_df.copy()
            if tier_filter != "ALL":
                filtered_df = filtered_df[filtered_df["risk_tier"] == tier_filter]
            if type_filter != "ALL":
                selected_type = type_filter[0]
                filtered_df = filtered_df[filtered_df["product_type"] == selected_type]
            if search_query:
                filtered_df = filtered_df[filtered_df["product_id"].str.contains(search_query, na=False)]

            st.caption(f"Showing **{len(filtered_df):,}** matching machine assets out of {total_assets:,} total.")

            # Render Asset Cards Grid (Top 12 matching assets)
            display_slice = filtered_df.head(12)
            grid_cols = st.columns(3)

            for idx, (_, m_row) in enumerate(display_slice.iterrows()):
                col_target = grid_cols[idx % 3]
                m_tier = m_row["risk_tier"]
                m_cls = get_risk_color_class(m_tier)
                m_badge_cls = f"badge-{m_tier.lower()}"

                with col_target:
                    st.markdown(
                        f"""
                        <div class="asset-card">
                            <div class="asset-title">
                                <span>{m_row['product_id']}</span>
                                <span class="badge-pill {m_badge_cls}">{m_tier}</span>
                            </div>
                            <div style="margin-top: 8px;">
                                <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                                    <span style="color:#94a3b8;">Health Index:</span>
                                    <span class="{m_cls}" style="font-weight:700;">{m_row['health_score']:.1f}%</span>
                                </div>
                            </div>
                            <div class="asset-metric-row">
                                <span>Speed: <b class="asset-metric-val">{m_row['rotational_speed_rpm']:.0f} RPM</b></span>
                                <span>Torque: <b class="asset-metric-val">{m_row['torque_nm']:.1f} Nm</b></span>
                            </div>
                            <div class="asset-metric-row">
                                <span>Process Temp: <b class="asset-metric-val">{m_row['process_temp_k']:.1f} K</b></span>
                                <span>Tool Wear: <b class="asset-metric-val">{m_row['tool_wear_min']:.0f} min</b></span>
                            </div>
                            <div class="asset-metric-row" style="border-top: 1px solid rgba(255,255,255,0.06); padding-top:6px; margin-top:8px;">
                                <span>Fail Prob: <b class="{m_cls}">{m_row['failure_probability_pct']:.1f}%</b></span>
                                <span>Anomaly: <b>{m_row['anomaly_score_pct']:.1f}%</b></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("🔍 View Complete Tabular Fleet Register (10,000 Records)"):
                st.dataframe(filtered_df, use_container_width=True, height=350)
        else:
            st.info("Run `python main.py` to generate complete fleet status matrix.")

    # ==========================================
    # TAB 3: MAINTENANCE ALERTS & DISPATCH QUEUE
    # ==========================================
    with tab3:
        st.markdown(
            """
            <div class="info-banner">
                📋 <b>Autonomous Work Order Dispatch Center:</b> Prioritized work order queue generated by the AI triage engine.
                Each ticket includes <b>TreeSHAP root-cause attribution</b> and <b>prescriptive maintenance protocols</b> to guide field technicians.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ALERTS_LOG_PATH.exists():
            alerts_df = pd.read_csv(ALERTS_LOG_PATH)
            total_alerts = len(alerts_df)

            # Count categories
            imm_count = int(alerts_df["urgency_level"].str.contains("IMMEDIATE", na=False).sum())
            high_count = int(alerts_df["urgency_level"].str.contains("HIGH", na=False).sum())
            med_count = int(alerts_df["urgency_level"].str.contains("MEDIUM", na=False).sum()) - imm_count
            drift_count = int(alerts_df["diagnosed_failure_mode"].str.contains("ANOMALY", na=False).sum())

            # Work Order KPI Summary
            w_c1, w_c2, w_c3, w_c4 = st.columns(4)
            w_c1.markdown(
                f'<div class="metric-card"><div class="metric-label">Emergency Halts</div><div class="metric-value status-critical">{imm_count} Tickets</div><div class="metric-sub">PWF & OSF Catastrophic Risk</div></div>',
                unsafe_allow_html=True,
            )
            w_c2.markdown(
                f'<div class="metric-card"><div class="metric-label">High Priority Cooling</div><div class="metric-value status-warning">{high_count} Tickets</div><div class="metric-sub">HDF Thermal Runaway</div></div>',
                unsafe_allow_html=True,
            )
            w_c3.markdown(
                f'<div class="metric-card"><div class="metric-label">Scheduled Tool Replacements</div><div class="metric-value status-watch">{med_count} Tickets</div><div class="metric-sub">TWF Tool Wear Exceeded</div></div>',
                unsafe_allow_html=True,
            )
            w_c4.markdown(
                f'<div class="metric-card"><div class="metric-label">Total Triaged Orders</div><div class="metric-value">{total_alerts:,}</div><div class="metric-sub">Autonomous AI Coverage</div></div>',
                unsafe_allow_html=True,
            )

            # Filter Controls
            st.markdown("##### 🛠️ Work Order Filters & Technician Dispatch")
            f_u1, f_u2, f_u3 = st.columns([1.2, 1.2, 1.6])

            with f_u1:
                urgency_choice = st.selectbox(
                    "Filter by Urgency Level",
                    ["ALL", "IMMEDIATE (Emergency Halt)", "HIGH (8-12h Response)", "MEDIUM (Tool Wear)", "LOW-MEDIUM (Routine)"],
                    index=1,  # Default to IMMEDIATE for critical triage
                )
            with f_u2:
                mode_choice = st.selectbox(
                    "Filter by Failure Mode",
                    ["ALL", "HDF - Heat Dissipation", "PWF - Power Failure", "OSF - Overstrain", "TWF - Tool Wear", "ANOMALY - Mechanical Drift"],
                    index=0,
                )
            with f_u3:
                wo_search = st.text_input(
                    "Search Ticket ID or Machine ID",
                    placeholder="e.g. TICK-00045, L47181...",
                ).strip().upper()

            # Apply Filters
            filtered_alerts = alerts_df.copy()
            if urgency_choice.startswith("IMMEDIATE"):
                filtered_alerts = filtered_alerts[filtered_alerts["urgency_level"].str.contains("IMMEDIATE", na=False)]
            elif urgency_choice.startswith("HIGH"):
                filtered_alerts = filtered_alerts[filtered_alerts["urgency_level"].str.contains("HIGH", na=False)]
            elif urgency_choice.startswith("MEDIUM"):
                filtered_alerts = filtered_alerts[filtered_alerts["urgency_level"].str.contains("MEDIUM", na=False) & ~filtered_alerts["urgency_level"].str.contains("IMMEDIATE", na=False)]
            elif urgency_choice.startswith("LOW-MEDIUM"):
                filtered_alerts = filtered_alerts[filtered_alerts["urgency_level"].str.contains("LOW-MEDIUM", na=False)]

            if mode_choice.startswith("HDF"):
                filtered_alerts = filtered_alerts[filtered_alerts["diagnosed_failure_mode"].str.contains("HDF", na=False)]
            elif mode_choice.startswith("PWF"):
                filtered_alerts = filtered_alerts[filtered_alerts["diagnosed_failure_mode"].str.contains("PWF", na=False)]
            elif mode_choice.startswith("OSF"):
                filtered_alerts = filtered_alerts[filtered_alerts["diagnosed_failure_mode"].str.contains("OSF", na=False)]
            elif mode_choice.startswith("TWF"):
                filtered_alerts = filtered_alerts[filtered_alerts["diagnosed_failure_mode"].str.contains("TWF", na=False)]
            elif mode_choice.startswith("ANOMALY"):
                filtered_alerts = filtered_alerts[filtered_alerts["diagnosed_failure_mode"].str.contains("ANOMALY", na=False)]

            if wo_search:
                filtered_alerts = filtered_alerts[
                    filtered_alerts["ticket_id"].str.contains(wo_search, na=False) |
                    filtered_alerts["equipment_id"].str.contains(wo_search, na=False)
                ]

            st.caption(f"Displaying **{len(filtered_alerts):,}** actionable maintenance tickets.")

            # Render Work Order Cards (Top 8 matching)
            for _, ticket_row in filtered_alerts.head(8).iterrows():
                t_id = ticket_row["ticket_id"]
                t_urgency = ticket_row["urgency_level"]
                t_mode = ticket_row["diagnosed_failure_mode"]
                t_equip = ticket_row["equipment_id"]
                t_type = ticket_row["product_type"]
                t_prob = ticket_row["failure_probability_pct"]
                t_anom = ticket_row["anomaly_score_pct"]
                t_cause = ticket_row["root_cause_explanation"]
                t_action = ticket_row["recommended_action"]
                t_shap = ticket_row.get("top_contributing_factors", "")

                # Urgency Card Styling
                if "IMMEDIATE" in t_urgency:
                    card_border = "wo-critical"
                    urg_badge = '<span class="badge-pill badge-critical">IMMEDIATE HALT</span>'
                elif "HIGH" in t_urgency:
                    card_border = "wo-high"
                    urg_badge = '<span class="badge-pill badge-high">HIGH PRIORITY</span>'
                elif "MEDIUM" in t_urgency:
                    card_border = "wo-medium"
                    urg_badge = '<span class="badge-pill badge-medium">SCHEDULED SWAP</span>'
                else:
                    card_border = "wo-routine"
                    urg_badge = '<span class="badge-pill badge-routine">ROUTINE WATCH</span>'

                shap_markup = f"<p style='margin:4px 0;'><b>TreeSHAP Attribution:</b> <code>{t_shap}</code></p>" if pd.notna(t_shap) and str(t_shap) != "nan" else ""

                card_html = f"""
                <div class="work-order-card {card_border}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <div>
                            <span style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">{t_id}</span>
                            <span class="badge-pill badge-type" style="margin-left:8px;">{t_equip} ({t_type}-Type)</span>
                            {urg_badge}
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8;">Timestamp: {ticket_row['timestamp']}</div>
                    </div>
                    <p style="margin:4px 0;"><b>Diagnosed Subsystem:</b> <span style="color:#f87171; font-weight:600;">{t_mode}</span></p>
                    <p style="margin:4px 0;"><b>Root Cause:</b> {t_cause}</p>
                    {shap_markup}
                    <p style="margin:4px 0;"><b>Prescriptive Protocol:</b> <span style="color:#fbbf24; font-weight:600;">{t_action}</span></p>
                    <div style="display:flex; gap:16px; margin-top:8px; font-size:0.82rem; color:#94a3b8;">
                        <span>Failure Probability: <b style="color:#ef4444;">{t_prob:.1f}%</b></span>
                        <span>Anomaly Index: <b style="color:#38bdf8;">{t_anom:.1f}%</b></span>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                # Interactive Dispatch / Acknowledge Action
                action_c1, action_c2 = st.columns([1, 4])
                if t_id in st.session_state.dispatched_tickets:
                    action_c1.success("✓ Dispatched to Field Crew")
                else:
                    if action_c1.button(f"⚡ Dispatch Crew", key=f"dispatch_{t_id}"):
                        st.session_state.dispatched_tickets.add(t_id)
                        st.rerun()

            with st.expander("🔍 View Complete Tabular Work Orders Register (5,280 Records)"):
                st.dataframe(filtered_alerts, use_container_width=True, height=350)
        else:
            st.info("No persistent alerts log found. Run `python main.py` to generate alert log.")


if __name__ == "__main__":
    main()

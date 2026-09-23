"""
Streamlit Live Interactive Demonstration UI: Predictive Equipment Maintenance Agent.
Simulates real-time sensor streaming, live gauges, sensor trend charts, and autonomous agent triage.
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

# Streamlit Page Config
st.set_page_config(
    page_title="AI Predictive Maintenance Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Sleek Dark Glassmorphism Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    .status-normal { color: #10b981; }
    .status-watch { color: #38bdf8; }
    .status-warning { color: #f59e0b; }
    .status-critical { color: #ef4444; }

    .alert-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.1);
    }
    .ticket-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f87171;
        margin-bottom: 8px;
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
    }
    return mapping.get(risk_tier, "status-normal")


def main():
    st.title("⚙️ AI-Powered Predictive Equipment Maintenance Agent")
    st.caption(
        "Microsoft LightGBM + Isolation Forest Telemetry Intelligence | 100% Free Local Stack"
    )

    # Check if models exist
    if not ANOMALY_MODEL_PATH.exists() or not FAILURE_MODEL_PATH.exists():
        st.warning("⚠️ Trained models not found. Please run the pipeline (`python main.py`) first.")
        st.stop()

    anomaly_detector, failure_predictor, agent = load_models()

    # Sidebar: Scenario Selection & Controls
    st.sidebar.header("🕹️ Live Stream Controls")

    scenarios = {
        "Heat Dissipation Failure (HDF)": SIMULATION_DIR / "scenario_hdf.csv",
        "Power Failure (PWF)": SIMULATION_DIR / "scenario_pwf.csv",
        "Overstrain Failure (OSF)": SIMULATION_DIR / "scenario_osf.csv",
        "Normal Operation Baseline": SIMULATION_DIR / "scenario_normal.csv",
    }

    # Verify scenario files exist or prepare them
    if not (SIMULATION_DIR / "scenario_hdf.csv").exists():
        if TELEMETRY_DATA_PATH.exists():
            df_tel = pd.read_csv(TELEMETRY_DATA_PATH)
            prepare_simulation_scenarios(df_tel)

    selected_scenario_name = st.sidebar.selectbox(
        "Select Simulation Scenario", list(scenarios.keys()), index=0
    )
    selected_scenario_path = scenarios[selected_scenario_name]

    sim_speed = st.sidebar.slider("Replay Interval (seconds)", 0.1, 1.5, 0.4, 0.1)

    # Session State Tracking
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_tickets" not in st.session_state:
        st.session_state.active_tickets = []

    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_clicked = col_btn1.button("▶️ Run Stream", use_container_width=True)
    reset_clicked = col_btn2.button("🔄 Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.sim_running = False
        st.session_state.current_step = 0
        st.session_state.history = []
        st.session_state.active_tickets = []
        st.rerun()

    if start_clicked:
        st.session_state.sim_running = True

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Live Telemetry & Agent", "🏭 Fleet Overview", "📋 Maintenance Alerts Queue"]
    )

    with tab1:
        # Load Scenario Data
        if selected_scenario_path.exists():
            scenario_df = pd.read_csv(selected_scenario_path)
        else:
            st.error("Scenario file not found. Run `python main.py` to generate simulation sets.")
            st.stop()

        # Metrics Row Placeholders
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        m_health = metric_col1.empty()
        m_prob = metric_col2.empty()
        m_anom = metric_col3.empty()
        m_risk = metric_col4.empty()

        # Telemetry Chart Placeholders
        chart_col1, chart_col2 = st.columns(2)
        chart_temp = chart_col1.empty()
        chart_speed = chart_col2.empty()

        # Agent Alert Box Placeholder
        agent_box = st.empty()

        # Simulation Loop
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

                row["anomaly_score_pct"] = float(anom_score_pct[0])
                row["failure_probability_pct"] = float(fail_prob_pct[0])

                health, risk = agent.compute_health_and_risk(
                    row["failure_probability_pct"], row["anomaly_score_pct"]
                )
                row["health_score"] = health
                row["risk_tier"] = risk

                # Agent Triage
                ticket = agent.evaluate_reading(row)
                if ticket:
                    st.session_state.active_tickets.append(ticket)

                st.session_state.history.append(row.to_dict())
                hist_df = pd.DataFrame(st.session_state.history)

                # Update Top KPI Cards
                risk_cls = get_risk_color_class(risk)
                m_health.markdown(
                    f'<div class="metric-card"><div class="metric-label">Health Index</div><div class="metric-value {risk_cls}">{health:.1f}%</div><div>Nominal: 90-100%</div></div>',
                    unsafe_allow_html=True,
                )
                m_prob.markdown(
                    f'<div class="metric-card"><div class="metric-label">Failure Probability</div><div class="metric-value {risk_cls}">{row["failure_probability_pct"]:.1f}%</div><div>LightGBM Predict</div></div>',
                    unsafe_allow_html=True,
                )
                m_anom.markdown(
                    f'<div class="metric-card"><div class="metric-label">Anomaly Score</div><div class="metric-value">{row["anomaly_score_pct"]:.1f}%</div><div>Isolation Forest</div></div>',
                    unsafe_allow_html=True,
                )
                m_risk.markdown(
                    f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value {risk_cls}">{risk}</div><div>Cycle #{row["udi"]}</div></div>',
                    unsafe_allow_html=True,
                )

                # Update Live Charts
                if len(hist_df) > 1:
                    chart_temp.line_chart(
                        hist_df.set_index("udi")[["process_temp_k", "air_temp_k"]],
                        height=250,
                    )
                    chart_speed.line_chart(
                        hist_df.set_index("udi")[["rotational_speed_rpm", "torque_nm"]],
                        height=250,
                    )

                # Render Agent Work Order if Alert Triggered
                if ticket:
                    agent_box.markdown(
                        f"""
                        <div class="alert-box">
                            <div class="ticket-header">🚨 MAINTENANCE WORK ORDER ISSUED — [{ticket.ticket_id}]</div>
                            <p><b>Equipment Asset:</b> {ticket.equipment_id} ({ticket.product_type}-Type) | <b>Urgency:</b> {ticket.urgency_level}</p>
                            <p><b>Diagnosed Failure Mode:</b> <span style="color:#f87171; font-weight:700;">{ticket.diagnosed_failure_mode}</span></p>
                            <p><b>Root Cause Analysis:</b> {ticket.root_cause_explanation}</p>
                            <p><b>Prescriptive Action:</b> <span style="color:#fbbf24; font-weight:600;">{ticket.recommended_action}</span></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                progress_bar.progress((step + 1) / total_steps)
                time.sleep(sim_speed)

            st.session_state.sim_running = False

        elif len(st.session_state.history) > 0:
            # Render Static Current State when paused
            latest = st.session_state.history[-1]
            risk_cls = get_risk_color_class(latest["risk_tier"])
            m_health.markdown(
                f'<div class="metric-card"><div class="metric-label">Health Index</div><div class="metric-value {risk_cls}">{latest["health_score"]:.1f}%</div><div>Nominal: 90-100%</div></div>',
                unsafe_allow_html=True,
            )
            m_prob.markdown(
                f'<div class="metric-card"><div class="metric-label">Failure Probability</div><div class="metric-value {risk_cls}">{latest["failure_probability_pct"]:.1f}%</div><div>LightGBM Predict</div></div>',
                unsafe_allow_html=True,
            )
            m_anom.markdown(
                f'<div class="metric-card"><div class="metric-label">Anomaly Score</div><div class="metric-value">{latest["anomaly_score_pct"]:.1f}%</div><div>Isolation Forest</div></div>',
                unsafe_allow_html=True,
            )
            m_risk.markdown(
                f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value {risk_cls}">{latest["risk_tier"]}</div><div>Cycle #{latest["udi"]}</div></div>',
                unsafe_allow_html=True,
            )

            hist_df = pd.DataFrame(st.session_state.history)
            chart_temp.line_chart(
                hist_df.set_index("udi")[["process_temp_k", "air_temp_k"]],
                height=250,
            )
            chart_speed.line_chart(
                hist_df.set_index("udi")[["rotational_speed_rpm", "torque_nm"]],
                height=250,
            )

            if st.session_state.active_tickets:
                last_ticket = st.session_state.active_tickets[-1]
                agent_box.markdown(
                    f"""
                    <div class="alert-box">
                        <div class="ticket-header">🚨 MAINTENANCE WORK ORDER ISSUED — [{last_ticket.ticket_id}]</div>
                        <p><b>Equipment Asset:</b> {last_ticket.equipment_id} ({last_ticket.product_type}-Type) | <b>Urgency:</b> {last_ticket.urgency_level}</p>
                        <p><b>Diagnosed Failure Mode:</b> <span style="color:#f87171; font-weight:700;">{last_ticket.diagnosed_failure_mode}</span></p>
                        <p><b>Root Cause Analysis:</b> {last_ticket.root_cause_explanation}</p>
                        <p><b>Prescriptive Action:</b> <span style="color:#fbbf24; font-weight:600;">{last_ticket.recommended_action}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Initial prompt
            m_health.markdown(
                '<div class="metric-card"><div class="metric-label">Health Index</div><div class="metric-value status-normal">100.0%</div><div>Ready</div></div>',
                unsafe_allow_html=True,
            )
            m_prob.markdown(
                '<div class="metric-card"><div class="metric-label">Failure Probability</div><div class="metric-value status-normal">0.0%</div><div>Ready</div></div>',
                unsafe_allow_html=True,
            )
            m_anom.markdown(
                '<div class="metric-card"><div class="metric-label">Anomaly Score</div><div class="metric-value">0.0%</div><div>Ready</div></div>',
                unsafe_allow_html=True,
            )
            m_risk.markdown(
                '<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value status-normal">NORMAL</div><div>Ready</div></div>',
                unsafe_allow_html=True,
            )
            st.info("👈 Press **Run Stream** in the sidebar to simulate live IoT telemetry and witness real-time failure prediction and autonomous agent response.")

    with tab2:
        st.subheader("🏭 Fleet Equipment Matrix")
        if EQUIPMENT_STATUS_PATH.exists():
            status_df = pd.read_csv(EQUIPMENT_STATUS_PATH)
            st.dataframe(status_df.head(100), use_container_width=True)
        else:
            st.info("Run `python main.py` to generate complete fleet status matrix.")

    with tab3:
        st.subheader("📋 Maintenance Alerts Log")
        if ALERTS_LOG_PATH.exists():
            alerts_df = pd.read_csv(ALERTS_LOG_PATH)
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.info("No persistent alerts log found. Run `python main.py` to generate alert log.")


if __name__ == "__main__":
    main()

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
import streamlit.components.v1 as components
from dashboard.components.digital_twin_3d import generate_digital_twin_html

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

    display_mode = st.sidebar.radio(
        "Telemetry Filter Mode",
        ["Industrial EMA Smoothed (Recommended)", "Raw Instantaneous Workpiece Readings"],
        index=0,
        help="Industrial EMA applies exponential temporal smoothing to filter workpiece noise, reflecting realistic machine health degradation.",
    )

    # Session State Tracking
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
        ["📊 Live Telemetry & Agent", "🏭 Fleet Overview", "📋 Maintenance Alerts Queue"]
    )

    with tab1:
        # Load Scenario Data
        if selected_scenario_path.exists():
            scenario_df = pd.read_csv(selected_scenario_path)
        else:
            st.error("Scenario file not found. Run `python main.py` to generate simulation sets.")
            st.stop()

        # Section 1: Split Screen - 3D Digital Twin (Left) & Real-Time Intelligence (Right)
        col_twin, col_kpis = st.columns([1.15, 0.85])

        with col_twin:
            st.markdown("##### 🌐 Interactive 3D Machine Spindle Digital Twin")
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
                shap_factors = failure_predictor.get_prediction_shap_contributions(feats_df, top_k=5)[0]

                raw_fail_prob = float(fail_prob_pct[0])
                raw_anom_score = float(anom_score_pct[0])
                raw_health, raw_risk = agent.compute_health_and_risk(raw_fail_prob, raw_anom_score)

                # Check if failure event occurred (actual dataset breakdown or high certainty)
                if row.get("machine_failure", 0) == 1 or raw_fail_prob > 85.0:
                    st.session_state.latched_failure = True

                # Compute EMA temporal smoothing (filters workpiece-to-workpiece cutting noise)
                # Escalate rapidly when risk increases, decay slowly
                alpha_p = 0.35 if raw_fail_prob > st.session_state.ema_prob else 0.15
                st.session_state.ema_prob = alpha_p * raw_fail_prob + (1 - alpha_p) * st.session_state.ema_prob
                st.session_state.ema_anom = 0.25 * raw_anom_score + 0.75 * st.session_state.ema_anom
                st.session_state.ema_health = 0.30 * raw_health + 0.70 * st.session_state.ema_health

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
                    f'<div class="metric-card"><div class="metric-label">Health Index</div><div class="metric-value {risk_cls}">{displayed_health:.1f}%</div><div>{"Smoothed Trend" if "EMA" in display_mode else "Instant Cycle"}</div></div>',
                    unsafe_allow_html=True,
                )
                m_prob.markdown(
                    f'<div class="metric-card"><div class="metric-label">Failure Probability</div><div class="metric-value {risk_cls}">{displayed_prob:.1f}%</div><div>{"LightGBM (EMA)" if "EMA" in display_mode else "LightGBM Instant"}</div></div>',
                    unsafe_allow_html=True,
                )
                m_anom.markdown(
                    f'<div class="metric-card"><div class="metric-label">Anomaly Score</div><div class="metric-value">{displayed_anom:.1f}%</div><div>Isolation Forest</div></div>',
                    unsafe_allow_html=True,
                )
                m_risk.markdown(
                    f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value {risk_cls}">{displayed_risk}</div><div>Cycle #{row["udi"]}</div></div>',
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
                        height=250,
                    )
                    chart_speed.line_chart(
                        hist_df.set_index("udi")[["rotational_speed_rpm", "torque_nm"]],
                        height=250,
                    )
                    # SHAP Contribution Bar Chart
                    shap_df = pd.DataFrame(shap_factors, columns=["Feature", "Contribution"]).set_index("Feature")
                    chart_shap.bar_chart(shap_df, height=220)

                # Render Agent Work Order if Alert Triggered
                if ticket:
                    shap_tag = f"<p><b>TreeSHAP Contributors:</b> <code>{ticket.top_contributing_factors}</code></p>" if ticket.top_contributing_factors else ""
                    agent_box.markdown(
                        f"""
                        <div class="alert-box">
                            <div class="ticket-header">🚨 MAINTENANCE WORK ORDER ISSUED — [{ticket.ticket_id}]</div>
                            <p><b>Equipment Asset:</b> {ticket.equipment_id} ({ticket.product_type}-Type) | <b>Urgency:</b> {ticket.urgency_level}</p>
                            <p><b>Diagnosed Failure Mode:</b> <span style="color:#f87171; font-weight:700;">{ticket.diagnosed_failure_mode}</span></p>
                            <p><b>Root Cause Analysis:</b> {ticket.root_cause_explanation}</p>
                            {shap_tag}
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

            # Render 3D Twin in paused state
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
                height=250,
            )
            chart_speed.line_chart(
                hist_df.set_index("udi")[["rotational_speed_rpm", "torque_nm"]],
                height=250,
            )

            latest_feats = pd.DataFrame([latest])[ALL_MODEL_FEATURES]
            latest_shap = failure_predictor.get_prediction_shap_contributions(latest_feats, top_k=5)[0]
            chart_shap.bar_chart(pd.DataFrame(latest_shap, columns=["Feature", "Contribution"]).set_index("Feature"), height=220)

            if st.session_state.active_tickets:
                last_ticket = st.session_state.active_tickets[-1]
                shap_tag = f"<p><b>TreeSHAP Contributors:</b> <code>{last_ticket.top_contributing_factors}</code></p>" if last_ticket.top_contributing_factors else ""
                agent_box.markdown(
                    f"""
                    <div class="alert-box">
                        <div class="ticket-header">🚨 MAINTENANCE WORK ORDER ISSUED — [{last_ticket.ticket_id}]</div>
                        <p><b>Equipment Asset:</b> {last_ticket.equipment_id} ({last_ticket.product_type}-Type) | <b>Urgency:</b> {last_ticket.urgency_level}</p>
                        <p><b>Diagnosed Failure Mode:</b> <span style="color:#f87171; font-weight:700;">{last_ticket.diagnosed_failure_mode}</span></p>
                        <p><b>Root Cause Analysis:</b> {last_ticket.root_cause_explanation}</p>
                        {shap_tag}
                        <p><b>Prescriptive Action:</b> <span style="color:#fbbf24; font-weight:600;">{last_ticket.recommended_action}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Initial prompt & Ready State
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

            # Initial 3D Digital Twin idling
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

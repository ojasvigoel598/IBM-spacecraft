"""
MissionMind - PRODUCTION Mission Control Dashboard v2
Enhanced per user feedback:
- Real physics telemetry change analysis (not just color)
- Full RAG pipeline visible with citations explaining WHY failure
- IBM watsonx Granite integration explicit (real vs mock, prompts, model ID)
- Scenario comparison, ML detector deep dive, physics rules deep dive

Architecture:
Telemetry -> Physics Rules (slope calc + threshold + human explanation)
         -> ML Detector (5 features, ensemble, scaled values)
         -> RAG (query, retrieved docs with scores, content)
         -> Granite (system prompt, user prompt with evidence, output JSON with citations)
         -> UI (Three.js + charts + reasoning panels)
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from missionmind.physics_rules.rules import check_power_subsystem, check_thermal_subsystem, slope, P_SOLAR_MAX
from missionmind.ml.detect import score_dataframe, load_models
from missionmind.ai.granite_client import generate_explanation, WATSONX_AVAILABLE
from missionmind.ai.rag import get_retriever
from missionmind.ai.prompts import SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_RAG, build_user_prompt, build_rag_user_prompt
from missionmind.simulator.thermal import SIGMA

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ===== DETERMINISTIC PIPELINE CACHE =====
# The physics solve (run_scenario) and the ML scoring are BOTH deterministic
# given (mode, duration): the solver uses no RNG (add_noise=False) and the
# ensemble is a fixed joblib artifact. So the full pipeline is safe to memoise
# per scenario — scrubbing the Time Transport then costs ~0 ms instead of
# re-solving 3600 s + re-scoring 3600 rows (~1.4 s) on every rerun.
# The P3-012 warm-up lockout (t<100 flag suppression) is applied INSIDE the
# cached functions so the cached object is never mutated downstream.
@st.cache_data(show_spinner=False, max_entries=6)
def _scored_scenario(mode: str) -> pd.DataFrame:
    from missionmind.simulator.run_scenarios import run_scenario
    _df = run_scenario(failure_mode=mode, duration_s=3600)
    _sc = score_dataframe(_df)
    _sc.loc[_sc["time_s"] < 100, "anomaly_flag"] = 0
    return _sc


@st.cache_data(show_spinner=False, max_entries=8)
def _scored_csv(csv_name: str) -> pd.DataFrame:
    _df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    _sc = score_dataframe(_df)
    _sc.loc[_sc["time_s"] < 100, "anomaly_flag"] = 0
    return _sc

# ===== SESSION CHECKPOINT: survive restarts =====
# Persist the last-viewed scenario + mission time + wall-clock stamp to a small
# JSON file. On the next launch (reboot, crash, manual restart) the dashboard
# resumes at the last position instead of resetting to t=0, and the header shows
# a "DATA AS OF" stamp so an operator can tell how stale the view is.
CHECKPOINT_PATH = os.path.join(DATA_DIR, "last_session.json")


def _load_checkpoint():
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_checkpoint(scenario, csv_name, time_s):
    try:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "scenario": scenario,
                "csv": csv_name,
                "time_s": int(time_s),
                "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + "Z",
            }, f)
    except Exception:
        pass


_checkpoint = _load_checkpoint()
_last_upd = (_checkpoint or {}).get("updated_at", "—")

st.set_page_config(
    page_title="MissionMind - Mission Control v2",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS — Mission Operations design system
st.markdown("""
<style>
    :root {
        --bg: #05070f; --panel: #0a101f; --panel2: #0d1526;
        --border: #182442; --line: #16213e;
        --text: #e8f4ff; --muted: #7d8db1; --dim: #5b6b8c;
        --accent: #00d4ff; --ok: #2ed573; --warn: #ffa502; --crit: #ff4757;
        --mono: Consolas, 'JetBrains Mono', 'Courier New', monospace;
    }
    .stApp { background: var(--bg); }
    .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1500px; }
    [data-testid="stSidebar"] { background: #0a0f1e; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] hr { border-color: var(--line); }
    [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stSlider { margin-bottom: 4px; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; height: 0; }

    /* Mission header */
    .mm-header { display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap;
                 gap: 10px; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 14px; }
    .mm-title { font-size: 1.5rem; font-weight: 800; color: var(--text); letter-spacing: 0.05em; line-height: 1.1; }
    .mm-title .accent { color: var(--accent); }
    .mm-title .edition { color: var(--dim); font-weight: 600; font-size: 1.0rem; letter-spacing: 0.02em; }
    .mm-sub { color: var(--muted); font-size: 0.8rem; margin-top: 4px; letter-spacing: 0.02em; }
    .mm-utc { font-family: var(--mono); font-size: 0.7rem; color: var(--dim); text-align: right; line-height: 1.5; }

    /* Status badges + chips */
    .mm-badge { font-family: var(--mono); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
                padding: 5px 14px; border-radius: 20px; border: 1px solid; white-space: nowrap; }
    .badge-nominal { color: var(--ok); border-color: #2ed57355; background: #2ed57314; }
    .badge-warning { color: var(--warn); border-color: #ffa50266; background: #ffa50214; }
    .badge-critical { color: var(--crit); border-color: #ff475766; background: #ff475714; animation: mm-blink 1.2s infinite; }
    .badge-init { color: var(--dim); border-color: #5b6b8c55; background: #5b6b8c11; }
    @keyframes mm-blink { 50% { opacity: 0.55; } }
    .mm-chip { display: inline-flex; align-items: center; gap: 7px; font-size: 0.72rem; font-weight: 600;
               letter-spacing: 0.04em; color: #9fb0d0; background: #0c1226; border: 1px solid #1e2a4a;
               border-radius: 6px; padding: 4px 11px; white-space: nowrap; }
    .mm-chip .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
    .dot-ok { background: var(--ok); box-shadow: 0 0 6px #2ed57399; }
    .dot-warn { background: var(--warn); box-shadow: 0 0 6px #ffa50299; }
    .dot-crit { background: var(--crit); box-shadow: 0 0 6px #ff475799; animation: mm-blink 1s infinite; }
    .dot-idle { background: #5b6b8c; }
    .status-strip { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .mm-section { font-size: 0.72rem; font-weight: 800; color: var(--accent); text-transform: uppercase;
                  letter-spacing: 0.16em; margin: 4px 0 8px 0; }

    /* KPI cards */
    .mm-kpi { background: linear-gradient(165deg, #0d1526 0%, #0a101f 100%); border: 1px solid var(--border);
             border-radius: 10px; padding: 10px 14px 8px; height: 100%; }
    .mm-kpi.warn { border-color: #ffa50255; }
    .mm-kpi.crit { border-color: #ff475755; }
    .mm-kpi .label { font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
    .mm-kpi .value { font-family: var(--mono); font-size: 1.5rem; font-weight: 700; color: var(--text); line-height: 1.15; margin-top: 3px; }
    .mm-kpi .unit { font-size: 0.75rem; color: var(--dim); font-weight: 400; }
    .mm-kpi .delta { font-family: var(--mono); font-size: 0.68rem; margin-top: 3px; }
    .delta-pos { color: var(--ok); } .delta-neg { color: var(--crit); } .delta-flat { color: var(--dim); }
    .delta-warn { color: var(--warn); }

    /* Panels + misc */
    .mm-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
    .rag-doc { background: #0d1628; border-left: 3px solid var(--accent); padding: 10px 12px; margin: 8px 0; border-radius: 6px; font-size: 0.85rem; }
    .granite-box { background: linear-gradient(150deg, #12102a 0%, #1a0f2e 100%); border: 1px solid #3a2a6a; border-radius: 12px; padding: 16px 18px; }
    .mm-flag { font-family: var(--mono); font-size: 0.85rem; padding: 8px 14px; border-radius: 8px; border: 1px solid; font-weight: 700; letter-spacing: 0.04em; }
    .flag-anom { color: var(--crit); border-color: #ff475766; background: #ff475711; }
    .flag-nom { color: var(--ok); border-color: #2ed57355; background: #2ed57311; }
    iframe[title="st.iframe"] { border: 1px solid var(--border); border-radius: 10px; }
    .mm-footer { color: var(--dim); font-size: 0.68rem; text-align: center; border-top: 1px solid var(--line); padding-top: 10px; margin-top: 18px; }
    div[data-testid="stMetric"] { background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { font-size: 0.8rem; letter-spacing: 0.02em; }
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-thumb { background: #1c2a4a; border-radius: 5px; }
    @media (prefers-reduced-motion: reduce) { .mm-badge, .dot-crit, .badge-critical { animation: none !important; } }
    button:focus-visible, [role="button"]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
""", unsafe_allow_html=True)

# Mission header
st.markdown(f"""
<div class="mm-header">
  <div>
    <div class="mm-title">🛰️ MISSION<span class="accent">MIND</span> <span class="edition">· Mission Control</span></div>
    <div class="mm-sub">Satellite Mission Operations · Physics-based Reliability · Fault Detection · RAG Diagnostics · RUL Prognostics</div>
  </div>
  <div class="mm-utc">SESSION UTC<br>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z<br><span style="opacity:.55;font-size:.62rem;font-weight:400">DATA AS OF<br>{_last_upd}</span></div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Mission Control")
    # Only telemetry CSVs are valid scenarios; evidence CSVs (fix_plan.csv, change_log.csv, ...) lack the time_s schema
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith('run_') and f.endswith('.csv')) if os.path.exists(DATA_DIR) else []
    if not csv_files:
        st.error("No CSVs found, run simulator/run_scenarios")
        csv_files = ["run_normal.csv","run_solar_failure.csv","run_radiator_failure.csv"]
    default_idx = csv_files.index("run_solar_failure.csv") if "run_solar_failure.csv" in csv_files else 0
    # Resume the exact file a previous session was replaying (checkpoint)
    if (_checkpoint or {}).get("scenario") == "csv" and (_checkpoint or {}).get("csv") in csv_files:
        default_idx = csv_files.index(_checkpoint["csv"])
    selected_csv = st.selectbox("📂 Scenario", csv_files, index=default_idx)
    st.divider()
    st.subheader("⏯️ Playback")
    auto_play = st.checkbox("Auto-play (advance every 2s)", value=False)
    playback_speed = st.slider("Step size (s/tick)", 1, 120, 30, 1)
    if 'frame_idx' not in st.session_state:
        # Resume at the last-viewed mission time after a restart (fresh session)
        st.session_state.frame_idx = int((_checkpoint or {}).get("time_s", 0))
    st.caption("Scrub forward & backward with the Time Transport bar below the status strip.")

    st.divider()
    st.subheader("📚 RAG & Reasoning")
    use_rag = st.checkbox("Enable RAG Retrieval", value=True)
    top_k = st.slider("RAG top-k docs", 1, 5, 3)
    show_prompts = st.checkbox("Show Granite prompts (watsonx)", value=False)
    show_raw = st.checkbox("Show raw telemetry table", value=False)

    st.divider()
    st.subheader("🔌 IBM watsonx.ai")
    api_key_present = bool(os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY"))
    proj_present = bool(os.getenv("WATSONX_PROJECT_ID"))
    st.write(f"SDK installed: **{WATSONX_AVAILABLE}**")
    st.write(f"API Key: **{'✅ present' if api_key_present else '❌ missing (mock fallback)'}**")
    st.write(f"Project ID: **{'✅ present' if proj_present else '❌ missing'}**")
    st.write(f"Model: **ibm/granite-3-2b-instruct**")
    if not api_key_present or not proj_present:
        st.info("Using deterministic mock that still returns valid evidence-based JSON with RAG citations. Set env vars to call real watsonx.")
    else:
        st.success("Real watsonx call will be attempted!")

    st.divider()
    st.caption("Constants are assumptions (see README). Thermal mc_p tuned 5000→2000 for demo speed, radiator final 30%→10% for detectability.")

# === SCENARIO CONTROL: inject failures → Simulator → ML → RAG → Granite → 3D ===
st.markdown('<div class="mm-panel">', unsafe_allow_html=True)
st.markdown("""<div class="mm-section">Scenario Control</div>""", unsafe_allow_html=True)
col_live1, col_live2, col_live3 = st.columns([1,1,1])
with col_live1:
    if st.button("✅ Normal Operation", use_container_width=True):
        st.session_state.live_mode = "none"
        st.session_state.frame_idx = 0
        st.rerun()
with col_live2:
    if st.button("☀️ Solar Array Degradation", type="primary", use_container_width=True):
        st.session_state.live_mode = "solar_degradation"
        st.session_state.frame_idx = 0
        st.rerun()
with col_live3:
    if st.button("🌡️ Radiator Degradation", type="primary", use_container_width=True):
        st.session_state.live_mode = "radiator_degradation"
        st.session_state.frame_idx = 0
        st.rerun()
# If this is a fresh session (e.g. after a reboot) and a checkpoint exists, resume
# the scenario that was being viewed rather than starting from CSV replay.
if "live_mode" not in st.session_state and _checkpoint and (_checkpoint.get("scenario") in ("none", "solar_degradation", "radiator_degradation")):
    st.session_state.live_mode = _checkpoint["scenario"]
live_mode = st.session_state.get("live_mode", "none")
st.caption(f"Selected: **{live_mode}** · live physics simulation via run_scenario() · fault injection starts at t=600s, ramp ends at t=900s")
st.markdown('</div>', unsafe_allow_html=True)

# Determine which data to load: if live_mode set, generate live via run_scenario (real physics, not just CSV replay)
from missionmind.simulator.run_scenarios import run_scenario
live_mode = st.session_state.get("live_mode", None)
if live_mode and live_mode in ("none","solar_degradation","radiator_degradation"):
    # Live generation: real physics simulation, cached per scenario so the
    # mission is solved ONCE and every scrub of the Time Transport is instant.
    st.caption(f"🔬 LIVE SIMULATION MODE: physics solved fresh via run_scenario('{live_mode}') — real equations, not pre-recorded. Cached per scenario; scrubbing replays the same solve (deterministic).")
    df_full = None
    try:
        df_scored = _scored_scenario(live_mode)
    except Exception as _e:
        st.error(f"ML model not trained: {_e}")
        df_scored = run_scenario(failure_mode=live_mode, duration_s=3600)
        df_scored["anomaly_score"] = 0.0
        df_scored["anomaly_flag"] = 0
else:
    # Load data (CSV replay mode)
    csv_path = os.path.join(DATA_DIR, selected_csv)
    if not os.path.exists(csv_path):
        st.error(f"Missing {csv_path}")
        st.stop()
    try:
        df_scored = _scored_csv(selected_csv)
    except Exception as _e:
        st.error(f"ML model not trained: {_e}")
        df_scored = pd.read_csv(csv_path)
        df_scored["anomaly_score"] = 0.0
        df_scored["anomaly_flag"] = 0
    df_full = None

# Load all scenarios for comparison
dfs_all = {}
for fname in ["run_normal.csv","run_solar_failure.csv","run_radiator_failure.csv"]:
    p=os.path.join(DATA_DIR,fname)
    if os.path.exists(p):
        dfs_all[fname]=pd.read_csv(p)

# Playback index
max_idx=len(df_scored)-1
if st.session_state.frame_idx>max_idx:
    st.session_state.frame_idx=max_idx  # P3-005 FIX: clamp instead of reset so the End button actually reaches the last frame
current_idx=int(st.session_state.frame_idx)
if auto_play:
    st.session_state.frame_idx=(st.session_state.frame_idx+playback_speed) % (max_idx+1)
    current_idx=int(st.session_state.frame_idx)

current_row = df_scored.iloc[current_idx]
window_df = df_scored.iloc[max(0,current_idx-120):current_idx+1]

# Persist the current view so a later restart resumes here. Called after the
# playback index is resolved so it always reflects what is actually shown.
# When in CSV replay (live_mode unset) record "csv" so the resume logic
# distinguishes "replay the file" from "regenerate a live scenario".
_save_checkpoint(live_mode or "csv", selected_csv, int(current_row["time_s"]))

# Physics checks
phys_power = check_power_subsystem(window_df)
phys_thermal = check_thermal_subsystem(window_df)
physics_flag=None
physics_conf=0.0
subsystem="unknown"
if phys_power:
    physics_flag=phys_power[0]
    physics_conf=phys_power[1]
    subsystem="power"
elif phys_thermal:
    physics_flag=phys_thermal[0]
    physics_conf=phys_thermal[1]
    subsystem="thermal"

anomaly_score_val=float(window_df["anomaly_score"].iloc[-1]) if "anomaly_score" in window_df.columns else 0.0
anomaly_flag_curr=int(current_row.get("anomaly_flag",0))
# ML cold-start burn-in: the first 100s are the documented transient that train.py excludes from
# evaluation (strict window 100-600). Suppress the display flag so mission start isn't shown as a
# false ANOMALY (failures inject at t>=600s, so no genuine anomaly exists in this window).
BURN_IN_S = 100
burn_in = int(current_row["time_s"]) < BURN_IN_S
anomaly_flag_curr = 0 if burn_in else anomaly_flag_curr

# Nominal values (per spec). Key must be "soc" to match current_values and
# the granite_client mock's contract (nom.get('soc')) — a "battery_soc" key
# here rendered "SOC 0.9 vs None" in the Granite Explanation tab.
nominal = {
    "solar_power_w": 520.0,
    "battery_voltage_v": 28.0,
    "soc": 1.0 if current_row["time_s"]>400 else 0.9,
    "temperature_c": -42.46,  # final normal with tuned mc_p
    "heat_in_w": 60.0,
    "heat_out_w": 60.0,
    "epsilon_A": 0.425,
}
# Current vs nominal deltas
curr_vals = {
    "solar_power_w": float(current_row["solar_power_w"]),
    "battery_voltage_v": float(current_row["battery_voltage_v"]),
    "battery_soc": float(current_row["battery_soc"]),
    "temperature_c": float(current_row["temperature_c"]),
    "heat_in_w": float(current_row["heat_in_w"]),
    "heat_out_w": float(current_row["heat_out_w"]),
}

# Detailed physics explanation generator
def generate_physics_explanation(row, window, nominal, failure_mode, phys_power, phys_thermal):
    md=""
    t=int(row["time_s"])
    md+=f"### 🔬 Real Telemetry Change Analysis at t={t}s (Mode: {failure_mode})\n\n"
    # Solar
    solar=row["solar_power_w"]
    net=solar-400.0
    dSOC_aprox = net/3600/100
    md+=f"**Power Subsystem:**\n"
    md+=f"- Solar: `{solar:.1f}W` vs nominal `{nominal['solar_power_w']:.0f}W` → Δ {solar-nominal['solar_power_w']:.1f}W ({solar/nominal['solar_power_w']*100:.0f}% of nominal)\n"
    md+=f"- Load: constant `400W` → Net power `{net:.1f}W` (positive charges, negative drains)\n"
    md+=f"- dSOC/dt ≈ net/3600/E_cap = `{net:.1f}/3600/100 = {dSOC_aprox:.6f}/s`\n"
    md+=f"- SOC: `{row['battery_soc']:.3f}` (V={row['battery_voltage_v']:.2f}V) vs nominal `{nominal['soc']:.3f}`\n"
    md+=f"- Slope of SOC in last 120s window: `{slope(window['battery_soc'].values, window['time_s'].values):.6f}/s`\n"
    if solar<364:
        md+=f"- **⚠️ Solar drop detected**: mean {window['solar_power_w'].mean():.1f}W < 0.7*Pmax (364W) threshold per [DOC-POWER-002]\n"
    if phys_power:
        md+=f"- **Physics flag**: `{phys_power[0]}` conf {phys_power[1]} → Solar degradation signature\n"
    md+="\n"
    md+=f"**Thermal Subsystem:**\n"
    md+=f"- Heat in: `{row['heat_in_w']:.1f}W` (from P_load*(1-η)=400*0.15, constant) vs nominal {nominal['heat_in_w']}W\n"
    md+=f"- Heat out: `{row['heat_out_w']:.1f}W` vs nominal {nominal['heat_out_w']}W → Δ {row['heat_out_w']-nominal['heat_out_w']:.1f}W\n"
    md+=f"- Net heating: Q_in - Q_out = `{row['heat_in_w']-row['heat_out_w']:.1f}W`\n"
    md+=f"- dT/dt = (Q_in-Q_out)/mc_p = {(row['heat_in_w']-row['heat_out_w'])/2000:.5f} K/s\n"
    md+=f"- Temperature: `{row['temperature_c']:.2f}C` vs nominal `{nominal['temperature_c']:.2f}C` → Δ {row['temperature_c']-nominal['temperature_c']:.2f}C\n"
    md+=f"- Slope temp in window: `{slope(window['temperature_c'].values, window['time_s'].values):.5f} C/s`\n"
    md+=f"- Slope heat_in: `{slope(window['heat_in_w'].values):.5f} W/s` (flat if <1.0)\n"
    md+=f"- Epsilon*A effective: nominal 0.425, current estimate {(row['heat_out_w']/(SIGMA*max(1,(row['temperature_c']+273.15)**4))):.4f}\n"
    if phys_thermal:
        md+=f"- **Physics flag**: `{phys_thermal[0]}` conf {phys_thermal[1]} → Radiator degradation\n"
    md+="\n"
    md+=f"**Why system failed / what changed:**\n"
    if failure_mode=="solar_degradation":
        md+=f"- At t=600-900s, degradation_factor ramped 1.0→0.48 (520W→249.6W). After ramp, net -150W drains battery at -0.000417 SOC/s. Battery hits 0% after ~40min, voltage drops 28→24V. This matches solar array stuck panel signature [DOC-POWER-002].\n"
        md+=f"- Real impact: If SOC<0.2, mission rule [DOC-POWER-PROC-001] says enter safe mode, shed loads to ≤250W.\n"
    elif failure_mode=="radiator_degradation":
        md+=f"- At t=600-900s, epsilon*A ramped 0.425→0.0425 (10% final, eq 124C). Heat rejection impaired: Q_out drops from ~60W to ~28W initially, net heating +32W, dT +0.016C/s. Temp climbs from -42C nominal to {row['temperature_c']:.1f}C, heading to 124C equilibrium. This matches radiator louver stuck [DOC-THERM-002].\n"
        md+=f"- Real impact: If T>60C, electronics risk HIGH per [DOC-MISSION-001], need load reduction to lower Q_in.\n"
    else:
        md+=f"- Normal operation: net +120W charges SOC to 1.0, temperature stable near -42C equilibrium (calculated from 60=εσA(T^4-3^4) → T=223K). No flags.\n"
    return md

try:
    _models_loaded = load_models()
    _model_full, _scaler_full = _models_loaded["full"]
    _score_threshold = float(_model_full.offset_)
except Exception:
    _model_full, _scaler_full, _score_threshold = None, None, -0.561

model_full, scaler_full = _model_full, _scaler_full  # reused in the ML Diagnostics tab

granite_input = {
    "subsystem": subsystem if physics_flag else ("power" if "solar" in selected_csv else "thermal" if "radiator" in selected_csv else "unknown"),
    "anomaly_score": round(abs(anomaly_score_val),3) if anomaly_flag_curr else 0.12,
    "anomaly_score_threshold": round(abs(_score_threshold),3),
    "ml_flag": int(anomaly_flag_curr),
    "physics_flag": physics_flag,
    "physics_confidence": physics_conf,
    "current_values": {
        "battery_voltage_v": round(curr_vals["battery_voltage_v"],2),
        "solar_power_w": round(curr_vals["solar_power_w"],1),
        "soc": round(curr_vals["battery_soc"],3),
        "temperature_c": round(curr_vals["temperature_c"],2),
        "heat_in_w": round(curr_vals["heat_in_w"],1),
        "heat_out_w": round(curr_vals["heat_out_w"],1),
        "epsilon_A": round(float(current_row["heat_out_w"]/(5.67e-8 * max(1, (current_row["temperature_c"]+273.15)**4))),4),
    },
    "nominal_values": nominal,
    "time_s": int(current_row["time_s"]),
    "failure_mode": str(current_row.get("failure_mode","none")),
}

# RAG retrieval
retrieved_docs=[]
if use_rag:
    try:
        retriever=get_retriever()
        retrieved_docs=retriever.query_from_anomaly(granite_input, top_k=top_k)
    except Exception as e:
        st.warning(f"RAG retrieval failed: {e}")

# Granite generation
granite_output=None
try:
    granite_output=generate_explanation(granite_input, use_rag=use_rag, top_k=top_k)
except Exception as e:
    st.error(f"Granite failed: {e}")

# Telemetry JSON for Three.js
telemetry_json = json.dumps({
    "time_s": int(current_row["time_s"]),
    "solar_power_w": float(current_row["solar_power_w"]),
    "battery_soc": float(current_row["battery_soc"]),
    "battery_voltage_v": float(current_row["battery_voltage_v"]),
    "temperature_c": float(current_row["temperature_c"]),
    "heat_in_w": float(current_row["heat_in_w"]),
    "heat_out_w": float(current_row["heat_out_w"]),
    "failure_mode": str(current_row.get("failure_mode","none")),
    "selected_scenario": selected_csv,
    "anomaly_flag": int(anomaly_flag_curr),
    "physics_flag": physics_flag,
    "physics_confidence": physics_conf,
    "anomaly_score": anomaly_score_val,
})

three_js_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin:0; overflow:hidden; background:#000; font-family: monospace; }}
  #c {{ width:100%; height:550px; display:block; }}
  #hud {{ position:absolute; top:10px; left:10px; color:#00d4ff; background:rgba(0,0,0,0.65); padding:10px 14px; border-radius:8px; border:1px solid #0f3460; font-size:11.5px; line-height:1.5; box-shadow:0 0 10px rgba(0,212,255,0.3); }}
  #hud b {{ color:#fff; }}
  .warn {{ color:#ff4757; animation: blink 1s infinite; font-weight:bold; }}
  @keyframes blink {{ 0%{{opacity:1}} 50%{{opacity:0.3}} 100%{{opacity:1}} }}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<!--SATELLITE_GEOMETRY-->
</head>
<body>
<canvas id="c"></canvas>
<div id="hud"></div>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
const telemetry = {telemetry_json};
const canvas=document.getElementById('c');
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x02010a);
const camera=new THREE.PerspectiveCamera(45, canvas.clientWidth/canvas.clientHeight, 0.1, 1000);
camera.position.set(1.9,1.2,2.2);
const renderer=new THREE.WebGLRenderer({{canvas:canvas, antialias:true}});
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.shadowMap.enabled=true;
renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=1.2;
const controls=new OrbitControls(camera, renderer.domElement);
controls.enableDamping=true; controls.dampingFactor=0.05; controls.minDistance=0.5; controls.maxDistance=8;
const ambient=new THREE.AmbientLight(0x404060,0.6); scene.add(ambient);
const sunLight=new THREE.DirectionalLight(0xffffff,2.0); sunLight.position.set(10,5,8); sunLight.castShadow=true; sunLight.shadow.mapSize.set(1024,1024); scene.add(sunLight);
const fillLight=new THREE.DirectionalLight(0x6a8cff,0.4); fillLight.position.set(-5,0,-5); scene.add(fillLight);
const anomalyLight=new THREE.PointLight(0xff3344,0,6); anomalyLight.position.set(0,1.5,0); scene.add(anomalyLight);
const starGeo=new THREE.BufferGeometry(); const starCount=2000; const starPos=new Float32Array(starCount*3);
for(let i=0;i<starCount*3;i++) starPos[i]=(Math.random()-0.5)*45; // inside skybox (r=80)
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos,3));
const starMat=new THREE.PointsMaterial({{color:0xffffff, size:0.4, sizeAttenuation:true}}); const stars=new THREE.Points(starGeo, starMat); scene.add(stars);
const earthGeo=new THREE.SphereGeometry(2,64,64); const earthMat=new THREE.MeshStandardMaterial({{color:0x1a5fb4, emissive:0x0a1a2a, emissiveIntensity:0.2, metalness:0.2, roughness:0.8}}); const earth=new THREE.Mesh(earthGeo, earthMat); earth.position.set(-10,-4,-15); scene.add(earth);
const atmGeo=new THREE.SphereGeometry(2.15,32,32); const atmMat=new THREE.MeshBasicMaterial({{color:0x00aaff, transparent:true, opacity:0.12, side:THREE.BackSide}}); const atmosphere=new THREE.Mesh(atmGeo, atmMat); atmosphere.position.copy(earth.position); scene.add(atmosphere);
const sc=new THREE.Group(); scene.add(sc);
// P3-008 FIX: real IBM satellite CAD (satellite_geometry.js from obj_to_geometry.py) replaces the procedural box model
const satData=SATELLITE_GEOMETRY, satScale=1.0/satData.size;
const sat=new THREE.Group(); sat.rotation.z=-Math.PI/2; sc.add(sat); // lay the 200 mm Y-axis panels horizontal, phased array forward
const satParts=[], satPanels=[], satBus=[];
(function buildSatellite(){{
  for(const part of satData.parts){{
    const g=new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(part.positions,3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(part.normals,3));
    g.setIndex(part.indices);
    const m=new THREE.MeshStandardMaterial({{color:new THREE.Color(part.color), metalness:0.3, roughness:0.5}});
    const mesh=new THREE.Mesh(g,m); mesh.castShadow=true; mesh.receiveShadow=true; mesh.name=part.name;
    mesh.baseColor=new THREE.Color(part.color);
    mesh.scale.setScalar(satScale);
    mesh.position.set(-satData.center[0]*satScale, -satData.center[1]*satScale, -satData.center[2]*satScale);
    sat.add(mesh); satParts.push(mesh);
    if(part.name.indexOf('Body2')===0||part.name.indexOf('Body3')===0) satPanels.push(mesh); // solar arrays
    if(part.name.indexOf('MainBusSquare')===0) satBus.push(mesh);                          // main bus
  }}}})();
window.__sat3d = {{ parts: satParts.length, panels: satPanels.length, bus: satBus.length }}; // P3-008 debug hook
// diagnostic overlays above the CAD
const beaconGeo=new THREE.SphereGeometry(0.05,16,16); const beaconMat=new THREE.MeshStandardMaterial({{color:0xff0000, emissive:0xff0000, emissiveIntensity:0.1}}); const beacon=new THREE.Mesh(beaconGeo, beaconMat); beacon.position.set(0,0.62,0); sc.add(beacon);
const outlineGeo=new THREE.BoxGeometry(1.15,0.42,0.5); const outlineMat=new THREE.MeshBasicMaterial({{color:0xff3344, transparent:true, opacity:0, side:THREE.BackSide}}); const outline=new THREE.Mesh(outlineGeo, outlineMat); sc.add(outline);
function updateVisuals(t){{
  const isSolarFail = t.solar_power_w < 364 || t.failure_mode==='solar_degradation';
  const isRadFail = t.physics_flag==='radiator_degradation' || t.failure_mode==='radiator_degradation' || t.temperature_c>30;
  const isAnomaly = t.anomaly_flag===1 || t.physics_flag!==null;
  const pulse = Math.sin(Date.now()*0.006);
  // real-CAD part-level animation: dim + pulse the solar arrays (Body2/Body3) on solar failure,
  // glow the main bus on radiator failure, rest stays clean
  for(const p of satParts){{
    p.material.color.copy(p.baseColor);
    if(satPanels.indexOf(p)>=0 && isSolarFail){{
      p.material.color.multiplyScalar(0.45);
      p.material.emissive.set(0xff3300); p.material.emissiveIntensity = 0.7+0.35*pulse;
    }} else if(satBus.indexOf(p)>=0 && isRadFail){{
      p.material.emissive.set(0xff3300); p.material.emissiveIntensity = 0.55+0.3*pulse;
    }} else {{
      p.material.emissive.set(0x000000); p.material.emissiveIntensity = 0;
    }}
  }}
  anomalyLight.intensity=isAnomaly? 2.0+Math.sin(Date.now()*0.01)*1.0 : 0;
  anomalyLight.color.copy(isSolarFail? new THREE.Color(0xffaa00) : new THREE.Color(0xff3344));
  beacon.material.emissiveIntensity=isAnomaly? 1.0+Math.sin(Date.now()*0.01)*0.5 : 0.1;
  outline.material.opacity=isAnomaly? 0.08+Math.sin(Date.now()*0.005)*0.06 : 0;
  const soc=t.battery_soc; let battColor;
  if(soc>0.8) battColor=new THREE.Color(0x00ff88); else if(soc>0.5) battColor=new THREE.Color(0xffcc00); else if(soc>0.3) battColor=new THREE.Color(0xff8800); else battColor=new THREE.Color(0xff0044);
  const hud=document.getElementById('hud');
  const _sec=t.time_s, _HH=String(Math.floor(_sec/3600)).padStart(2,'0'), _MM=String(Math.floor(_sec/60)%60).padStart(2,'0'), _SS=String(Math.floor(_sec)%60).padStart(2,'0');
  const _ang=((_sec%1200)/1200*360).toFixed(1);
  hud.innerHTML=`<b>⏱ T+${{_HH}}:${{_MM}}:${{_SS}}</b> · ORBIT ${{_ang}}°<br>☀️ <b>${{t.solar_power_w.toFixed(1)}}W</b> (${{(t.solar_power_w/520*100).toFixed(0)}}%) · 🔋 <b style='color:${{battColor.getStyle()}}'>${{(t.battery_soc*100).toFixed(1)}}%</b><br>🌡️ <b>${{t.temperature_c.toFixed(1)}}°C</b> · V:${{t.battery_voltage_v.toFixed(2)}}V<br>${{t.anomaly_flag?'<span class="warn">⚠ ANOMALY · '+t.anomaly_score.toFixed(3)+'</span><br>':'✅ NOMINAL<br>'}}${{t.physics_flag?'<span class="warn">'+t.physics_flag+' · conf '+t.physics_confidence+'</span>':''}}`;
}}
/*__SCENE_POLISH__*/
let time=0; function animate(){{ requestAnimationFrame(animate); time+=0.01; applyOrbit(); earth.rotation.y+=0.0015; atmosphere.rotation.y+=0.0015; stars.rotation.y+=0.0001; controls.update(); updateVisuals(telemetry); applyHover(); updateLabels(); renderer.render(scene,camera); window.__sat3d.tris = renderer.info.render.triangles; }} animate();
window.addEventListener('resize',()=>{{ const w=canvas.clientWidth, h=600; camera.aspect=w/h; camera.updateProjectionMatrix(); renderer.setSize(w,h,false); }});
</script>
</body>
</html>
"""

# P3-008: inject the real satellite CAD geometry into the embedded viewer
_geom_js = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'components', 'satellite_geometry.js')
if os.path.exists(_geom_js):
    with open(_geom_js, encoding='utf-8') as _fh:
        three_js_html = three_js_html.replace('<!--SATELLITE_GEOMETRY-->',
                                              '<script>\n' + _fh.read() + '\n</script>')
else:
    raise RuntimeError("satellite_geometry.js missing - run: python missionmind/viz/components/obj_to_geometry.py")

# P3-009: scene polish (gradient skybox, spacecraft part labels, hover feedback, rim light).
# Plain JS injected post-f-string so braces need no escaping.
_SCENE_POLISH_JS = """
// ===== P3-009 scene polish: skybox, part labels, hover feedback =====
// 1. subtle gradient skybox (depth cue instead of flat black)
const _skyCanvas = document.createElement('canvas'); _skyCanvas.width = 2; _skyCanvas.height = 512;
const _skyCtx = _skyCanvas.getContext('2d');
const _skyGrad = _skyCtx.createLinearGradient(0, 0, 0, 512);
_skyGrad.addColorStop(0, '#0c1428'); _skyGrad.addColorStop(0.55, '#04060f'); _skyGrad.addColorStop(1, '#060b1c');
_skyCtx.fillStyle = _skyGrad; _skyCtx.fillRect(0, 0, 2, 512);
const _skyTex = new THREE.CanvasTexture(_skyCanvas);
const _sky = new THREE.Mesh(new THREE.SphereGeometry(80, 16, 16),
  new THREE.MeshBasicMaterial({ map: _skyTex, side: THREE.BackSide, depthWrite: false }));
_sky.frustumCulled = false; scene.add(_sky);
// 2. spacecraft part labels (DOM overlay, projected every frame)
const _labelHost = document.createElement('div');
_labelHost.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:1;';
(canvas.parentElement || document.body).appendChild(_labelHost);
const _labels = [];
function _partCenter(mesh) {
  const p = mesh.geometry.attributes.position.array; let x = 0, y = 0, z = 0, n = 0;
  for (let i = 0; i < p.length; i += 3) { x += p[i]; y += p[i + 1]; z += p[i + 2]; n++; }
  return [x / n, y / n, z / n];
}
function _addLabel(anchor, text, color) {
  const el = document.createElement('div');
  el.textContent = text;
  el.style.cssText = 'position:absolute;font:600 9.5px Consolas,monospace;color:' + color +
    ';background:rgba(4,7,15,0.8);border:1px solid #1e2a4a;border-radius:4px;padding:2px 7px;' +
    'letter-spacing:0.08em;transform:translate(-50%,-170%);display:none;white-space:nowrap;';
  _labelHost.appendChild(el);
  _labels.push({ obj: anchor, el: el });
}
const _byGrp = {};
for (const _p of satParts) { const _g = _p.name.split('/')[0]; (_byGrp[_g] = _byGrp[_g] || []).push(_p); }
function _anchor(mesh) {
  const c = _partCenter(mesh);
  const o = new THREE.Object3D();
  o.position.set((c[0] - satData.center[0]) * satScale, (c[1] - satData.center[1]) * satScale, (c[2] - satData.center[2]) * satScale);
  sat.add(o);
  return o;
}
if (_byGrp['Body2']) _addLabel(_anchor(_byGrp['Body2'][0]), 'SOLAR ARRAY L', '#8fd8ff');
if (_byGrp['Body3']) _addLabel(_anchor(_byGrp['Body3'][0]), 'SOLAR ARRAY R', '#8fd8ff');
if (_byGrp['MainBusSquare']) _addLabel(_anchor(_byGrp['MainBusSquare'][0]), 'MAIN BUS', '#e2e8f0');
if (_byGrp['PhasedArrayAntenna']) _addLabel(_anchor(_byGrp['PhasedArrayAntenna'][0]), 'PHASED ARRAY', '#e2e8f0');
function updateLabels() {
  const v = new THREE.Vector3();
  for (const lab of _labels) {
    lab.obj.getWorldPosition(v);
    v.project(camera);
    if (v.z > 1) { lab.el.style.display = 'none'; continue; }
    const x = (v.x * 0.5 + 0.5) * canvas.clientWidth;
    const y = (-v.y * 0.5 + 0.5) * canvas.clientHeight;
    lab.el.style.display = 'block';
    lab.el.style.transform = 'translate(' + x + 'px,' + y + 'px) translate(-50%,-170%)';
  }
}
// 3. hover feedback: cyan highlight + pointer cursor on the part under the cursor
const _ray = new THREE.Raycaster(), _mouse = new THREE.Vector2();
let _hovered = null;
renderer.domElement.addEventListener('pointermove', function (e) {
  const r = renderer.domElement.getBoundingClientRect();
  _mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  _mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  _ray.setFromCamera(_mouse, camera);
  const hits = _ray.intersectObjects(satParts, false);
  if (hits.length) {
    const m = hits[0].object;
    if (_hovered !== m) {
      if (_hovered) { _hovered.material.emissive.set(0x000000); _hovered.material.emissiveIntensity = _hovered.__baseEm || 0; }
      _hovered = m; _hovered.__baseEm = _hovered.material.emissiveIntensity || 0;
      renderer.domElement.style.cursor = 'pointer';
    }
  } else {
    if (_hovered) { _hovered.material.emissive.set(0x000000); _hovered.material.emissiveIntensity = _hovered.__baseEm || 0; _hovered = null; }
    renderer.domElement.style.cursor = 'default';
  }
});
function applyHover() {
  if (_hovered) { _hovered.material.emissive.set(0x00d4ff); _hovered.material.emissiveIntensity = 0.35; }
}
// 4. hemisphere + rim light for material readability
const _hemi = new THREE.HemisphereLight(0x3a4a7a, 0x05070f, 0.55); scene.add(_hemi);
const _rim = new THREE.DirectionalLight(0x2a4a7a, 0.5); _rim.position.set(-4, 2, -6); scene.add(_rim);
// 5. ORBITAL MOTION: Earth at centre, satellite travels a 1200s orbit ring in sync
//    with the mission clock (telemetry.time_s drives both the plots and the 3D).
const _ORB_PERIOD = 1200, _ORB_R = 3.6;
earth.position.set(0, 0, 0); earth.scale.setScalar(0.65);
atmosphere.position.set(0, 0, 0); atmosphere.scale.setScalar(0.65);
const _ringPts = [];
for (let _i = 0; _i <= 128; _i++) {
  const _a = (_i / 128) * Math.PI * 2;
  _ringPts.push(new THREE.Vector3(Math.cos(_a) * _ORB_R, 0, Math.sin(_a) * _ORB_R));
}
const _ringGeo = new THREE.BufferGeometry().setFromPoints(_ringPts);
const _ring = new THREE.Line(_ringGeo, new THREE.LineBasicMaterial({ color: 0x2a4a7a, transparent: true, opacity: 0.45 }));
scene.add(_ring);
const _marker = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 12),
  new THREE.MeshBasicMaterial({ color: 0x00d4ff }));
scene.add(_marker);
function applyOrbit() {
  const _theta = ((telemetry.time_s % _ORB_PERIOD) / _ORB_PERIOD) * Math.PI * 2;
  const _x = Math.cos(_theta) * _ORB_R;
  const _z = Math.sin(_theta) * _ORB_R;
  sc.position.set(_x, 0, _z);
  sc.rotation.y = -_theta + Math.PI / 2;   // nose along the velocity vector
  _marker.position.set(_x, 0, _z);
  // follow the spacecraft: camera + orbit-controls target ride with it
  camera.position.set(_x + Math.cos(_theta) * 2.1, 1.35, _z + Math.sin(_theta) * 2.1);
  controls.target.set(_x, 0, _z);
  controls.update();
  window.__orbit = { deg: (_theta * 180 / Math.PI) % 360, period: _ORB_PERIOD };
}
"""
three_js_html = three_js_html.replace('/*__SCENE_POLISH__*/', _SCENE_POLISH_JS)

# ===== OPS OVERVIEW: system status strip + KPI row =====
risk = granite_output.get("risk", "LOW") if granite_output else "LOW"
t_s = int(current_row["time_s"])
if burn_in:
    sys_state, sys_badge = "INITIALIZING", "badge-init"
elif risk == "HIGH" or physics_flag or anomaly_flag_curr:
    sys_state, sys_badge = "CRITICAL", "badge-critical"
elif risk == "MEDIUM":
    sys_state, sys_badge = "WARNING", "badge-warning"
else:
    sys_state, sys_badge = "NOMINAL", "badge-nominal"

solar_w = float(current_row["solar_power_w"])
net_w = solar_w - 400.0
soc = float(current_row["battery_soc"])
volt = float(current_row["battery_voltage_v"])
temp = float(current_row["temperature_c"])
heat_bal = float(current_row["heat_in_w"]) - float(current_row["heat_out_w"])

# ===== RUL PROGNOSTICS: physics-based time-to-limit (leading indicator that
# can flag degradation BEFORE the ML/physics detectors trip) =====
# Battery: minutes until 0% SOC at the current net power (dSOC/dt = net/3600/100).
if net_w < 0:
    _d_soc = abs(net_w) / 3600.0 / 100.0
    bat_rul_min = (soc / _d_soc) / 60.0 if _d_soc > 0 else float("inf")
else:
    bat_rul_min = float("inf")   # charging -> no depletion risk
# Thermal: minutes until the 60C electronics limit at current dT/dt (mc_p=2000).
_dT_dt = heat_bal / 2000.0
if _dT_dt > 0 and temp < 60.0:
    thm_rul_min = (60.0 - temp) / _dT_dt / 60.0
else:
    thm_rul_min = float("inf")   # stable / cooling
rul_state = "ok"
if bat_rul_min < 30.0 or thm_rul_min < 60.0:
    rul_state = "crit"
elif bat_rul_min < 60.0 or thm_rul_min < 120.0:
    rul_state = "warn"
rul_label = (
    (f"BAT {bat_rul_min:.0f}m" if bat_rul_min < float('inf') else "BAT ∞") + " · " +
    (f"THM {thm_rul_min:.0f}m" if thm_rul_min < float('inf') else "THM ∞")
)
# Prognostics is a LEADING indicator: escalate the ops state even when the
# ML/physics detectors have not tripped yet (e.g. solar ramp 733s vs flag 900s).
if rul_state == "crit":
    if sys_state == "NOMINAL":
        sys_state, sys_badge = "WARNING", "badge-warning"
    elif sys_state == "WARNING":
        sys_state, sys_badge = "CRITICAL", "badge-critical"
elif rul_state == "warn" and sys_state == "NOMINAL":
    sys_state, sys_badge = "WARNING", "badge-warning"

_DOT = {"ok": "dot-ok", "warn": "dot-warn", "crit": "dot-crit", "idle": "dot-idle"}
def _chip(label, state="idle"):
    return f'<span class="mm-chip"><span class="dot {_DOT[state]}"></span>{label}</span>'

def _kpi(label, value, unit, delta_text, delta_cls="delta-flat", state=""):
    return (f'<div class="mm-kpi {state}"><div class="label">{label}</div>'
            f'<div class="value">{value}<span class="unit"> {unit}</span></div>'
            f'<div class="delta {delta_cls}">{delta_text}</div></div>')

power_state = "crit" if physics_flag == "solar_degradation" else ("warn" if solar_w < 364 else "ok")
thermal_state = "crit" if physics_flag == "radiator_degradation" else ("warn" if temp > 30 else "ok")
ml_state = "crit" if anomaly_flag_curr else "ok"
rag_state = "ok" if retrieved_docs else "idle"
granite_state = "ok" if granite_output else "idle"

st.markdown("""<div class="mm-section">System Status</div>""", unsafe_allow_html=True)
st.markdown(f"""
<div class="status-strip">
  <span class="mm-badge {sys_badge}">{sys_state}</span>
  {_chip('POWER', power_state)}
  {_chip('THERMAL', thermal_state)}
  {_chip('ML DETECTOR', ml_state)}
  {_chip('RAG', rag_state)}
  {_chip('GRANITE', granite_state)}
  {_chip('PHYSICS: ' + (physics_flag.upper() if physics_flag else 'CLEAR'), 'crit' if physics_flag else 'ok')}
  {_chip('ML: ' + ('ANOMALY' if anomaly_flag_curr else 'NOMINAL'), 'crit' if anomaly_flag_curr else 'ok')}
  {_chip('RUL: ' + rul_label, rul_state)}
</div>
""", unsafe_allow_html=True)
# Clickable quick-nav: jump the mission clock to the fault window of the
# subsystem you care about (website-form behaviour — a chip opens the moment).
_quick = st.columns(4)
with _quick[0]:
    if st.button("⚡ Solar fault onset", use_container_width=True):
        st.session_state.frame_idx = 802
        st.rerun()
with _quick[1]:
    if st.button("🌡️ Radiator fault onset", use_container_width=True):
        st.session_state.frame_idx = 802
        st.rerun()
with _quick[2]:
    if st.button("🟢 Normal start (t=0)", use_container_width=True):
        st.session_state.frame_idx = 0
        st.rerun()
with _quick[3]:
    if st.button("🔚 End of mission", use_container_width=True):
        st.session_state.frame_idx = max_idx
        st.rerun()

explanation_md = generate_physics_explanation(current_row, window_df, nominal, str(current_row.get("failure_mode","none")), phys_power, phys_thermal)

if physics_flag or anomaly_flag_curr:
    _alert_name = {"solar_degradation": "SOLAR ARRAY DEGRADATION", "radiator_degradation": "RADIATOR DEGRADATION"}.get(physics_flag or "", "ML-DETECTED ANOMALY")
    _alert_risk = "HIGH" if (physics_flag or anomaly_flag_curr) else "ELEVATED"
    _alert_sys = {"power": "EPS", "thermal": "TCS"}.get(subsystem, "—" if subsystem == "unknown" else subsystem)
    _alert_ev = []
    if physics_flag == "solar_degradation":
        _alert_ev.append(f"solar {solar_w:.0f}W < 364W")
    if physics_flag == "radiator_degradation":
        _alert_ev.append(f"heat rejection {heat_bal:+.0f}W")
    if anomaly_flag_curr:
        _alert_ev.append(f"ML score {anomaly_score_val:.3f}")
    st.markdown(f"""
<div class="mm-panel" style="border-color:#ff475766; background:#10060d;">
  <span class="mm-flag flag-anom">⚠ {_alert_name}</span><br>
  <span style="font-family:Consolas,monospace; color:#ff9a9a; font-size:0.8rem; line-height:1.7;">
    Risk: <b>{_alert_risk}</b> · Detected: <b>T+{t_s//60:02d}:{t_s%60:02d}</b> · Subsystem: <b>{_alert_sys}</b><br>
    Evidence: {' · '.join(_alert_ev) if _alert_ev else '—'}
  </span>
</div>
""", unsafe_allow_html=True)
    with st.expander("Detailed reasoning", expanded=False):
        st.markdown(explanation_md)

# ===== TIME TRANSPORT: scrub forward/backward through the mission =====
st.markdown('<div class="mm-panel" style="padding:10px 16px">', unsafe_allow_html=True)
st.markdown("""<div class="mm-section">Mission Time Transport</div>""", unsafe_allow_html=True)
_tc1, _tc2 = st.columns([3, 1.4])
with _tc1:
    _scrub = st.slider("Mission time (s)", 0, int(df_scored["time_s"].iloc[-1]), int(current_row["time_s"]), step=30,
                       help="Drag to jump anywhere in the 1-hour mission — no need to wait for the fault.")
    st.session_state.frame_idx = int(_scrub)
with _tc2:
    st.markdown(f'<div class="mm-kpi"><div class="label">Mission Time</div><div class="value">{int(current_row["time_s"])}<span class="unit"> s</span></div><div class="delta delta-flat">frame {current_idx}/{max_idx}</div></div>', unsafe_allow_html=True)
_tb = st.columns(7)
with _tb[0]:
    if st.button("⏮ Start", use_container_width=True): st.session_state.frame_idx = 0; st.rerun()
with _tb[1]:
    if st.button("◀◀ −5 min", use_container_width=True): st.session_state.frame_idx = max(0, current_idx-300); st.rerun()
with _tb[2]:
    if st.button("◀ −30s", use_container_width=True): st.session_state.frame_idx = max(0, current_idx-30); st.rerun()
with _tb[3]:
    if st.button("▶ +30s", use_container_width=True): st.session_state.frame_idx = min(max_idx, current_idx+30); st.rerun()
with _tb[4]:
    if st.button("▶▶ +5 min", use_container_width=True): st.session_state.frame_idx = min(max_idx, current_idx+300); st.rerun()
with _tb[5]:
    if st.button("⏭ End", use_container_width=True): st.session_state.frame_idx = max_idx; st.rerun()
with _tb[6]:
    _anom_mask = df_scored["anomaly_flag"] == 1
    _first_anom = int(df_scored.loc[_anom_mask, "time_s"].min()) if "anomaly_flag" in df_scored.columns and _anom_mask.any() else 900
    if st.button(f"⚡ Fault onset (t={_first_anom}s)", use_container_width=True): st.session_state.frame_idx = _first_anom; st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ===== RUL PROGNOSTICS row (leading indicator) =====
st.markdown("""<div class="mm-section">Prognostics — Remaining Useful Life</div>""", unsafe_allow_html=True)
_pc = st.columns(3)
with _pc[0]:
    _bat_txt = f"{bat_rul_min:.0f}" if bat_rul_min < float('inf') else "∞"
    _bat_d = "draining · time to 0% SOC" if net_w < 0 else "charging · no depletion risk"
    _bat_cls = "delta-neg" if bat_rul_min < 60 else ("delta-warn" if bat_rul_min < 120 else "delta-pos")
    _bat_state = "crit" if bat_rul_min < 30 else ("warn" if bat_rul_min < 60 else "")
    st.markdown(_kpi("Battery RUL", _bat_txt, "min", _bat_d, _bat_cls, _bat_state), unsafe_allow_html=True)
with _pc[1]:
    _th_txt = f"{thm_rul_min:.0f}" if thm_rul_min < float('inf') else "∞"
    _th_d = "heating toward 60°C limit" if _dT_dt > 0 else "thermal stable"
    _th_cls = "delta-neg" if thm_rul_min < 60 else ("delta-warn" if thm_rul_min < 120 else "delta-pos")
    _th_state = "crit" if thm_rul_min < 60 else ("warn" if thm_rul_min < 120 else "")
    st.markdown(_kpi("Thermal Margin", _th_txt, "min", _th_d, _th_cls, _th_state), unsafe_allow_html=True)
with _pc[2]:
    st.markdown(_kpi("Net Power", f"{net_w:+.0f}", "W", "solar − 400W load", "delta-neg" if net_w<0 else "delta-pos"), unsafe_allow_html=True)
st.caption("RUL = minutes until battery hits 0% SOC at current net power, or until electronics reach 60°C at current heating rate. It is a LEADING indicator: in the solar scenario it starts counting down at ~t=730s, before the ML detector flags (~900s).")

st.markdown("""<div class="mm-section">Key Telemetry</div>""", unsafe_allow_html=True)
k1 = _kpi("Mission Time", f"{t_s}", "s", selected_csv.replace("run_", "").replace(".csv", "").replace("_", " "), "delta-flat")
k2 = _kpi("Solar Power", f"{solar_w:.0f}", "W", f"Δ {solar_w-520:+.0f} W vs 520W nominal", "delta-warn" if solar_w < 364 else "delta-flat", "warn" if solar_w < 364 else "")
k3 = _kpi("Net Power", f"{net_w:+.0f}", "W", "charging" if net_w > 0 else "draining", "delta-pos" if net_w > 0 else "delta-neg")
k4 = _kpi("Battery SOC", f"{soc*100:.1f}", "%", f"Δ {(soc-1)*100:+.1f} pts vs full", "delta-neg" if soc < 0.5 else "delta-pos", "crit" if soc < 0.3 else ("warn" if soc < 0.5 else ""))
k5 = _kpi("Bus Voltage", f"{volt:.2f}", "V", f"Δ {volt-28:+.2f} V vs 28V", "delta-neg" if volt < 26 else "delta-flat")
k6 = _kpi("Temperature", f"{temp:.1f}", "°C", f"Δ {temp+42.46:+.1f} °C vs nominal", "delta-warn" if temp > 30 else "delta-flat", "warn" if temp > 30 else "")
k7 = _kpi("Heat Balance", f"{heat_bal:+.0f}", "W", "Q_in − Q_out (heating if +)", "delta-warn" if heat_bal > 5 else "delta-pos", "warn" if heat_bal > 5 else "")
k8 = _kpi("Anomaly Score", f"{anomaly_score_val:.3f}", "", "lower = more anomalous", "delta-neg" if anomaly_flag_curr else "delta-pos", "crit" if anomaly_flag_curr else "")

_r1 = st.columns(4)
for _c, _k in zip(_r1, [k1, k2, k3, k4]):
    _c.markdown(_k, unsafe_allow_html=True)
_r2 = st.columns(4)
for _c, _k in zip(_r2, [k5, k6, k7, k8]):
    _c.markdown(_k, unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ===== Layout =====
col_3d, col_physics = st.columns([1.2, 1])

with col_3d:
    st.subheader("🛰️ Live Spacecraft - PBR Three.js (Driven by Real Telemetry)")
    components.html(three_js_html, height=580, scrolling=False)
    if auto_play:
        time.sleep(2)
        st.rerun()

with col_physics:
    st.subheader("Live Trend — last 120s")
    if PLOTLY_AVAILABLE:
        _lw = window_df
        _lf = go.Figure()
        _lf.add_trace(go.Scatter(x=_lw["time_s"], y=_lw["solar_power_w"], name="Solar W", line=dict(color="#00d4ff")))
        _lf.add_trace(go.Scatter(x=_lw["time_s"], y=_lw["temperature_c"], name="Temp °C", line=dict(color="#ff6b6b"), yaxis="y2"))
        _lf.add_trace(go.Scatter(x=_lw["time_s"], y=_lw["battery_soc"]*100, name="SOC %", line=dict(color="#2ed573", dash="dot"), yaxis="y2"))
        _lf.update_layout(height=300, template="plotly_dark", margin=dict(l=10,r=10,t=24,b=10),
                          paper_bgcolor="#0a101f", plot_bgcolor="#0a101f",
                          legend=dict(orientation="h", y=1.12, font=dict(size=10)),
                          font=dict(color="#7d8db1", family="Consolas, monospace"),
                          yaxis=dict(gridcolor="#16213e", zeroline=False, title="Solar W"),
                          yaxis2=dict(overlaying="y", side="right", gridcolor="#16213e", zeroline=False, title="°C / %"))
        _lf.update_xaxes(gridcolor="#16213e", zeroline=False)
        st.plotly_chart(_lf, use_container_width=True)
    with st.expander("🔬 Physics Verification — what changed & why", expanded=False):
        st.markdown(explanation_md)
    with st.expander("📊 Current vs Nominal (delta)", expanded=False):
        delta_df = pd.DataFrame([
            {"metric": "Solar Power", "current": f"{curr_vals['solar_power_w']:.1f}W", "nominal": f"{nominal['solar_power_w']:.0f}W", "delta": f"{curr_vals['solar_power_w']-nominal['solar_power_w']:+.1f}W ({curr_vals['solar_power_w']/nominal['solar_power_w']*100:.0f}%)"},
            {"metric": "Battery SOC", "current": f"{curr_vals['battery_soc']*100:.1f}%", "nominal": f"{nominal['soc']*100:.0f}%", "delta": f"{(curr_vals['battery_soc']-nominal['soc'])*100:+.1f}%"},
            {"metric": "Voltage", "current": f"{curr_vals['battery_voltage_v']:.2f}V", "nominal": f"{nominal['battery_voltage_v']:.1f}V", "delta": f"{curr_vals['battery_voltage_v']-nominal['battery_voltage_v']:+.2f}V"},
            {"metric": "Temperature", "current": f"{curr_vals['temperature_c']:.1f}C", "nominal": f"{nominal['temperature_c']:.1f}C", "delta": f"{curr_vals['temperature_c']-nominal['temperature_c']:+.1f}C"},
            {"metric": "Heat In", "current": f"{curr_vals['heat_in_w']:.0f}W", "nominal": f"{nominal['heat_in_w']:.0f}W", "delta": f"{curr_vals['heat_in_w']-nominal['heat_in_w']:+.0f}W"},
            {"metric": "Heat Out", "current": f"{curr_vals['heat_out_w']:.0f}W", "nominal": f"{nominal['heat_out_w']:.0f}W", "delta": f"{curr_vals['heat_out_w']-nominal['heat_out_w']:+.0f}W"},
        ])
        st.dataframe(delta_df, use_container_width=True, hide_index=True)

    # (KPI cards above already cover Time/Solar/Anomaly - no duplicate metric row)
# Tabs for deep dive
st.divider()
tab_live, tab_physics, tab_ml, tab_rag, tab_granite, tab_compare, tab_watsonx, tab_ingest = st.tabs([
    "📈 Telemetry",
    "🔬 Physics Rules",
    "🎛️ ML Diagnostics",
    "📚 RAG & Evidence",
    "🧠 Granite Reasoning",
    "🔀 Scenarios",
    "🔌 Integration",
    "🔴 Live Ingest"
])

with tab_live:
    st.subheader("Telemetry Replay — Real Data Change")
    chart_df = df_scored.iloc[:current_idx+1]
    if PLOTLY_AVAILABLE:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Solar Power & Battery Voltage x10", "Temperature & Heat Flux", "Anomaly Flag & Score"))
        fig.add_trace(go.Scatter(x=chart_df.time_s, y=chart_df.solar_power_w, name="Solar W", line=dict(color="#00d4ff")), row=1,col=1)
        fig.add_trace(go.Scatter(x=chart_df.time_s, y=chart_df.battery_voltage_v*10, name="Voltage x10", line=dict(color="#2ed573")), row=1,col=1)
        fig.add_trace(go.Scatter(x=chart_df.time_s, y=chart_df.temperature_c, name="Temp C", line=dict(color="#ff6b6b")), row=2,col=1)
        fig.add_trace(go.Scatter(x=chart_df.time_s, y=chart_df.heat_in_w, name="Heat In", line=dict(color="#ffa502", dash="dash")), row=2,col=1)
        fig.add_trace(go.Scatter(x=chart_df.time_s, y=chart_df.heat_out_w, name="Heat Out", line=dict(color="#00d4ff", dash="dot")), row=2,col=1)
        fig.add_trace(go.Scatter(x=chart_df.time_s, y=chart_df.anomaly_flag*10, name="Flag x10", line=dict(color="#ff4757")), row=3,col=1)
        fig.add_trace(go.Scatter(x=chart_df.time_s, y=chart_df.anomaly_score, name="Score", line=dict(color="#a55eea")), row=3,col=1)
        for r in [1,2,3]:
            fig.add_vline(x=600, line_dash="dash", line_color="orange", row=r,col=1)
            fig.add_vline(x=900, line_dash="dot", line_color="red", row=r,col=1)
        fig.update_layout(height=700, template="plotly_dark", showlegend=True,
                          paper_bgcolor="#05070f", plot_bgcolor="#0a101f",
                          font=dict(color="#7d8db1", family="Consolas, monospace"))
        fig.update_xaxes(gridcolor="#16213e", zeroline=False)
        fig.update_yaxes(gridcolor="#16213e", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Orange dash = injection start 600s, Red dot = ramp end 900s. Watch solar drop 520→249W, net power -150W, SOC drain, heat_out drop 60→28W, temp rise -42→50C.")
    else:
        st.line_chart(chart_df[["solar_power_w","battery_voltage_v","temperature_c","anomaly_flag"]])

    if show_raw:
        st.dataframe(df_scored.iloc[max(0,current_idx-30):current_idx+5])

with tab_physics:
    st.subheader("Physics Rules Layer — Exact Logic (Spec §6) + Tuned Thresholds")
    st.markdown("""
    **Why tuning?** Original spec thresholds were inconsistent with spec constants:
    - Solar net -150W → dSOC -0.000417/s, threshold -0.0005 too strict → tuned -0.0002 + SOC<0.5
    - Radiator net +32W → dT 0.0064 C/s, threshold 0.01 too strict → tuned 0.003
    Flagged in README and code comments.
    """)
    colA,colB=st.columns(2)
    with colA:
        st.markdown("#### Power Check")
        st.code(f"""
window = last 120s
solar_mean = {window_df['solar_power_w'].mean():.1f}W
threshold = 0.7*Pmax = {0.7*P_SOLAR_MAX:.0f}W
solar_drop = {window_df['solar_power_w'].mean():.1f} < {0.7*P_SOLAR_MAX:.0f} = {window_df['solar_power_w'].mean() < 0.7*P_SOLAR_MAX}

soc_slope = {slope(window_df['battery_soc'].values, window_df['time_s'].values):.6f}/s
soc_mean = {window_df['battery_soc'].mean():.3f}
threshold tuned = -0.0002 (orig -0.0005)
soc_declining = slope < -0.0002 OR (mean<0.5 AND slope<0.0001) OR mean<0.1
              = {slope(window_df['battery_soc'].values, window_df['time_s'].values) < -0.0002 or (window_df['battery_soc'].mean()<0.5)}

Result: {phys_power}
""", language="python")
    with colB:
        st.markdown("#### Thermal Check")
        st.code(f"""
temp_slope = {slope(window_df['temperature_c'].values, window_df['time_s'].values):.5f} C/s
heat_in_slope = {slope(window_df['heat_in_w'].values):.5f} W/s
temp_mean = {window_df['temperature_c'].mean():.2f}C
threshold tuned = 0.003 (orig 0.01)
temp_rising = slope > 0.003 = {slope(window_df['temperature_c'].values, window_df['time_s'].values) > 0.003}
heat_in_stable = abs({slope(window_df['heat_in_w'].values):.3f}) < 1.0 = {abs(slope(window_df['heat_in_w'].values)) < 1.0}

Result: {phys_thermal}

Q_in = P_load*(1-η)=400*0.15=60W constant
Q_out = εσA(T^4 - T_space^4)
At T={current_row['temperature_c']:.1f}C, Q_out={current_row['heat_out_w']:.1f}W
Net = {current_row['heat_in_w']-current_row['heat_out_w']:.1f}W heating
dT/dt = Net/mc_p = {(current_row['heat_in_w']-current_row['heat_out_w'])/2000:.5f} K/s
""", language="python")

    st.markdown("**Equilibrium solving (hand-check before trusting sim):**")
    st.code(f"""
Nominal: 60 = 0.85*5.67e-8*0.5*(T^4 - 81) => T_eq = 223K = -50C (cold bias with A=0.5, spec says low-tens ideal)
Radiator degraded 10%: 60 = 0.85*5.67e-8*0.0425*(T^4) => T_eq=397K=124C (HIGH risk)
At current T, approach to new equilibrium drives anomaly.
""")

with tab_ml:
    st.markdown("""<div class="mm-section">ML Diagnostics · Current Frame</div>""", unsafe_allow_html=True)
    # ---- Main view: score / health / trend / confidence / subsystem ----
    if _model_full is None:
        try:
            _models_loaded = load_models()
            model_full, scaler_full = _models_loaded["full"]
            _score_threshold = float(model_full.offset_)
        except Exception:
            _score_threshold = -0.1
    _offset = _score_threshold
    _scr_vals = window_df["anomaly_score"].values if "anomaly_score" in window_df.columns else np.zeros(3)
    _slope_scr = float(np.polyfit(np.arange(len(_scr_vals)), _scr_vals, 1)[0]) if len(_scr_vals) >= 3 else 0.0
    _health = "ANOMALY" if anomaly_flag_curr else "NOMINAL"
    _trend_txt = "worsening" if _slope_scr < -0.0005 else ("improving" if _slope_scr > 0.0005 else "stable")
    _conf = min(0.99, abs(anomaly_score_val) / max(0.01, abs(_offset))) if anomaly_flag_curr else max(0.0, 1.0 - abs(anomaly_score_val) / max(0.01, abs(_offset)))
    _sub_txt = {"power": "Power (EPS)", "thermal": "Thermal (TCS)"}.get(subsystem, "—")
    _mc = st.columns(5)
    _mc[0].markdown(_kpi("Anomaly Score", f"{anomaly_score_val:.3f}", "", f"threshold {_offset:.3f}", "delta-neg" if anomaly_flag_curr else "delta-flat"), unsafe_allow_html=True)
    _mc[1].markdown(_kpi("Health State", _health, "", "ensemble verdict", "delta-neg" if anomaly_flag_curr else "delta-pos"), unsafe_allow_html=True)
    _mc[2].markdown(_kpi("Trend", _trend_txt, "", f"score slope {_slope_scr:+.4f}/s", "delta-warn" if _trend_txt == "worsening" else "delta-flat"), unsafe_allow_html=True)
    _mc[3].markdown(_kpi("Confidence", f"{_conf:.0%}", "", "score vs threshold", "delta-flat"), unsafe_allow_html=True)
    _mc[4].markdown(_kpi("Subsystem", _sub_txt, "", "attributed source", "delta-warn" if subsystem != "unknown" else "delta-flat"), unsafe_allow_html=True)

    # ---- Model Details: Unsupervised vs Supervised (click to expand) ----
    with st.expander("🔬 Model Details — Unsupervised vs Supervised", expanded=False):
        st.markdown("**Unsupervised** models learn \"normal\" from unlabelled telemetry and flag deviations from it. "
                    "**Supervised** models are trained on labelled failure data — they are shown for reference and are "
                    "not directly comparable to the unsupervised detectors.")
        _rep = {}
        for _p in (os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'comparison_report.json'),
                   os.path.join(os.path.dirname(__file__), '..', 'models', 'comparison_report.json')):
            if os.path.exists(_p):
                try:
                    with open(_p, encoding="utf-8") as _f:
                        _rep = json.load(_f)
                    break
                except Exception:
                    pass
        if _rep:
            _rows = []
            for _name, _scen in _rep.items():
                _group = "Supervised" if "Supervised" in _name else "Unsupervised"
                _hs = [_scen[k] for k in ("solar_failure_holdout", "radiator_failure_holdout") if k in _scen]
                if not _hs:
                    continue
                def _avg(_key):
                    _v = [h.get(_key) for h in _hs if h.get(_key) is not None]
                    return sum(_v)/len(_v) if _v else float("nan")
                _short = _name.split(" (")[0]
                _rows.append({"Model": _short, "Group": _group,
                              "F1": round(_avg("f1"), 3), "Precision": round(_avg("precision"), 3),
                              "Recall": round(_avg("recall"), 3), "ROC-AUC": round(_avg("roc_auc"), 3),
                              "PR-AUC": round(_avg("pr_auc"), 3), "Delay(s)": round(_avg("detection_delay_s"), 1)})
            _dfm = pd.DataFrame(_rows).sort_values(["Group", "Model"])
            st.dataframe(_dfm, use_container_width=True, hide_index=True,
                         column_config={
                             "F1": st.column_config.NumberColumn("F1", help="Harmonic mean of precision & recall", format="%.3f"),
                             "Precision": st.column_config.NumberColumn("Precision", help="Of items flagged anomalous, fraction truly degraded", format="%.3f"),
                             "Recall": st.column_config.NumberColumn("Recall", help="Of genuinely degraded items, fraction detected", format="%.3f"),
                             "ROC-AUC": st.column_config.NumberColumn("ROC-AUC", help="Threshold-independent ranking quality; 1.0 = perfect", format="%.3f"),
                             "PR-AUC": st.column_config.NumberColumn("PR-AUC", help="Precision-recall area; better for imbalanced fault data", format="%.3f"),
                             "Delay(s)": st.column_config.NumberColumn("Delay(s)", help="Mean detection delay after fault injection", format="%.1f"),
                         })
            st.caption("Holdout metrics averaged over solar + radiator failure test sets (missionmind/models/comparison_report.json).")
        else:
            st.info("comparison_report.json not found — run `python -m missionmind.ml.compare` to generate it.")

    # ---- Feature readout (secondary) ----
    with st.expander("📊 Feature Readout — current frame", expanded=False):
        try:
            from missionmind.ml.train import add_derivative_features, build_feature_matrix
            df_feat = add_derivative_features(pd.DataFrame([current_row]))
            X,_ = build_feature_matrix(df_feat)
            X_scaled = scaler_full.transform(X)
            feat_names = ["Battery Voltage", "Solar Power", "Temperature", "dTemp/dt", "dVolt/dt"]
            rows = []
            for nm, raw, z in zip(feat_names, X[0], X_scaled[0]):
                status = "🔴 far outside normal" if abs(z) > 3 else ("🟡 borderline" if abs(z) > 1.5 else "🟢 within normal")
                rows.append({"Feature": nm, "Value": f"{raw:.3f}", "Z-score": f"{z:+.2f}", "Status": status})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Z-score = scaled distance from the trained normal distribution. |z| > 3 is a strong outlier.")
        except Exception as e:
            st.warning(f"Feature readout unavailable: {e}")

    st.caption("Model setup: IsolationForest ensemble (full + power-only + thermal-only), contamination 0.07, 300 estimators — ORed so a single-subsystem anomaly is still flagged.")

with tab_rag:
    st.markdown("""<div class="mm-section">RAG Evidence · Why</div>""", unsafe_allow_html=True)
    # Query
    _q = (f"subsystem={granite_input['subsystem']} · flag={granite_input['physics_flag'] or 'none'} · "
          f"solar={granite_input['current_values']['solar_power_w']}W · temp={granite_input['current_values']['temperature_c']}°C · SOC={granite_input['current_values']['soc']}")
    st.markdown(f"**Query:** `{_q}`")
    # Concise answer
    if granite_output:
        _risk = granite_output.get("risk", "UNKNOWN")
        _rb = {"HIGH": "badge-critical", "MEDIUM": "badge-warning", "LOW": "badge-nominal"}.get(_risk, "badge-init")
        st.markdown(f"**Risk:** <span class='mm-badge {_rb}'>{_risk}</span> · **Confidence:** {granite_output.get('confidence', 0.0):.0%}", unsafe_allow_html=True)
        st.markdown(f"**Probable cause:** {granite_output.get('probable_cause', '—')}")
        st.markdown(f"**Recommended action:** {granite_output.get('recommended_action', '—')}")
        if granite_output.get("evidence_used"):
            st.markdown(f"**Evidence used:** {', '.join(granite_output['evidence_used'])}")
    # Sources
    if retrieved_docs:
        st.markdown("**Sources:**")
        for doc in retrieved_docs:
            _src = doc.get("path", "")
            if isinstance(_src, str):
                _src = os.path.basename(_src.replace("\\", "/"))
            st.markdown(f"<div class='rag-doc'><b>[{doc['id']}] {doc['title']}</b> — relevance {doc.get('score',0):.2f}<br><small>{_src}</small></div>", unsafe_allow_html=True)
    else:
        st.caption("No evidence retrieved at this frame.")
    # Expandable supporting context
    with st.expander("Supporting context — full document excerpts", expanded=False):
        for doc in (retrieved_docs or []):
            st.markdown(f"<div class='rag-doc'><b>[{doc['id']}] {doc['title']}</b><br>{doc['content'][:800]}</div>", unsafe_allow_html=True)
    with st.expander("Knowledge Base", expanded=False):
        kb_dir=os.path.join(os.path.dirname(__file__), '..', 'ai', 'knowledge_base')
        for md_file in sorted(os.listdir(kb_dir)):
            if md_file.endswith('.md'):
                with st.expander(f"📄 {md_file}"):
                    with open(os.path.join(kb_dir,md_file), encoding="utf-8") as f:
                        st.markdown(f.read())

with tab_granite:
    st.subheader("💡 IBM watsonx Granite — Evidence-Based Explanation")
    st.markdown("""
    **Pipeline:** Telemetry → Physics flag + ML flag → RAG retrieves docs → Prompt (system + user + evidence) → Granite → JSON with citations
    """)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("**System Prompt (locked):**")
        st.code(SYSTEM_PROMPT_RAG if use_rag else SYSTEM_PROMPT_BASE, language="text")
        if show_prompts:
            st.markdown("**User Prompt (with evidence):**")
            if use_rag and retrieved_docs:
                user_p = build_rag_user_prompt(granite_input, retrieved_docs)
            else:
                user_p = build_user_prompt(granite_input)
            st.code(user_p[:3000], language="json")
    with col2:
        st.markdown("**Anomaly Input JSON (built from physics+ML):**")
        st.json(granite_input)

    st.divider()
    if granite_output:
        risk=granite_output.get("risk","UNKNOWN")
        badge={"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}.get(risk,"badge-none")
        st.markdown(f"### Risk: <span class='{badge}'>{risk}</span>  —  Confidence: {granite_output.get('confidence',0.85)}", unsafe_allow_html=True)
        st.markdown('<div class="granite-box">', unsafe_allow_html=True)
        st.markdown(f"**Probable Cause:** {granite_output.get('probable_cause','')}")
        st.markdown(f"**Reasoning (must include citations):** {granite_output.get('reasoning','')}")
        st.markdown(f"**Recommended Action (grounded in troubleshooting doc):** {granite_output.get('recommended_action','')}")
        if "evidence_used" in granite_output:
            st.markdown(f"**Evidence Used:** {', '.join(granite_output['evidence_used'])}")
        if "retrieved_docs" in granite_output:
            st.markdown("**Retrieved Docs Meta:**")
            st.json(granite_output["retrieved_docs"])
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("Raw Granite JSON Output (valid schema)"):
            st.json(granite_output)
    else:
        st.info("Granite output will appear when anomaly detected or periodically. Force by setting time >900s in solar/radiator scenario.")

    st.markdown("""
    **Why evidence-based vs generic:** 
    - Generic LLM: "Solar might be degraded"
    - Ours: cites numbers 520→249W (0.48×), net -150W, dSOC -0.000417/s, slope, threshold 364W per [DOC-POWER-002], plus procedure [DOC-POWER-PROC-001] to shed loads ≤250W, plus mission rule SOC<0.3 HIGH risk.
    """)

with tab_compare:
    st.subheader("🔀 Scenario Comparison — Real Data Change Across All Scenarios")
    if PLOTLY_AVAILABLE and dfs_all:
        # Build comparison of solar, soc, temp, anomaly flag across 3 scenarios
        for metric in ["solar_power_w","battery_soc","temperature_c"]:
            fig=go.Figure()
            for fname, df in dfs_all.items():
                fig.add_trace(go.Scatter(x=df["time_s"], y=df[metric], name=fname.replace('.csv',''), mode="lines"))
            fig.add_vline(x=600, line_dash="dash", line_color="orange")
            fig.add_vline(x=900, line_dash="dot", line_color="red")
            fig.update_layout(title=f"{metric} across scenarios (orange=inject start, red=end)", template="plotly_dark", height=350,
                              paper_bgcolor="#05070f", plot_bgcolor="#0a101f",
                              font=dict(color="#7d8db1", family="Consolas, monospace"))
            fig.update_xaxes(gridcolor="#16213e", zeroline=False)
            fig.update_yaxes(gridcolor="#16213e", zeroline=False)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Solar failure: solar drops, SOC drains to 0, temp unchanged. Radiator failure: temp rises -42→50C, heat_out drops 60→~20W, solar & SOC unchanged. Normal: SOC plateaus 1.0, temp stable -42C.")
    else:
        st.write("Plotly not available or data missing")

    # Table comparison at current time across scenarios
    st.markdown("#### Current Time Comparison Across Scenarios")
    if dfs_all:
        rows=[]
        for fname, df in dfs_all.items():
            if current_idx < len(df):
                r=df.iloc[current_idx]
                rows.append({"scenario":fname, "time":r["time_s"], "solar":r["solar_power_w"], "soc":r["battery_soc"], "volt":r["battery_voltage_v"], "temp":r["temperature_c"], "heat_out":r["heat_out_w"]})
        st.dataframe(pd.DataFrame(rows))

with tab_watsonx:
    st.subheader("🔌 IBM watsonx.ai Integration — Code & Status")
    st.markdown(f"""
    **SDK Available:** `{WATSONX_AVAILABLE}`  
    **Model ID:** `ibm/granite-3-2b-instruct` (Granite)  
    **Credentials:** API Key present={api_key_present}, Project ID present={proj_present}  
    **Mode:** {"REAL watsonx call" if api_key_present and proj_present and WATSONX_AVAILABLE else "MOCK fallback (deterministic, evidence-based, always valid JSON)"}  

    **Why mock?** Ensures demo works offline for judges without IBM Cloud account, while code is production-ready for real call.
    """)
    st.code("""
# Real watsonx call (from granite_client.py)
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

creds = Credentials(api_key=os.getenv("WATSONX_APIKEY"), url="https://us-south.ml.cloud.ibm.com")
model = ModelInference(
    model_id="ibm/granite-3-2b-instruct",
    credentials=creds,
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={"decoding_method":"greedy", "max_new_tokens":500, "temperature":0.2}
)
prompt = f"<|system|>\\n{SYSTEM_PROMPT_RAG}\\n<|user|>\\n{user_prompt_with_RAG}\\n<|assistant|>\\n"
response = model.generate_text(prompt=prompt)
# Parse JSON, validate risk/cause/reasoning/action + evidence_used
    """, language="python")
    st.markdown("**Evidence-based output schema (locked for Streamlit rendering, no fragile text parsing):**")
    st.code("""
{
  "risk": "LOW|MEDIUM|HIGH",
  "probable_cause": str,
  "reasoning": str (must cite [DOC-...] and numbers),
  "recommended_action": str (grounded in troubleshooting),
  "evidence_used": [doc ids],
  "confidence": float,
  "retrieved_docs": [...]
}
    """, language="json")

with tab_ingest:
    st.markdown("""<div class="mm-section">Live Ingest · Virtual Edge Node → TCP/MQTT → Live Ensemble Scoring</div>""", unsafe_allow_html=True)
    st.markdown(
        "A virtual ESP32-class device publishes the SAME physics as frame-by-frame telemetry "
        "over a real JSON-lines transport; the production ML ensemble scores each incoming "
        "window. This is the dynamic path — data arrives, scores move, alerts fire. "
        "A physical ESP32/RPi speaking the same wire format drops in unchanged."
    )

    # Session-persistent streaming state: the node + accumulated frames live
    # in session_state, so each rerun ADVANCES the stream (never replays).
    from missionmind.telemetry.edge_node import VirtualEdgeNode
    if "ingest_node" not in st.session_state:
        st.session_state.ingest_node = VirtualEdgeNode(
            failure_mode="solar_degradation", noise=True, adc_bits=12,
            drop_rate=0.02)
    if "ingest_frames" not in st.session_state:
        st.session_state.ingest_frames = []  # list of dicts (frame.to_dataframe_row)
    if "ingest_live" not in st.session_state:
        st.session_state.ingest_live = []    # (time_s, score, flag) history for the trend

    _ic1, _ic2, _ic3, _ic4 = st.columns([1, 1, 1, 1.4])
    with _ic1:
        _ingest_mode = st.selectbox("Edge-node fault mode",
                                    ["none", "solar_degradation", "radiator_degradation"],
                                    index=1)
    with _ic2:
        _n_frames = st.slider("Frames per tick", 5, 120, 30, 5)
    with _ic3:
        _advance = st.button("⏩ Advance stream", use_container_width=True)
    with _ic4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _auto_ingest = st.checkbox("Auto-advance every 1s", value=False)

    node = st.session_state.ingest_node
    if node.failure_mode != _ingest_mode:
        st.session_state.ingest_node = VirtualEdgeNode(
            failure_mode=_ingest_mode, noise=True, adc_bits=12, drop_rate=0.02)
        node = st.session_state.ingest_node
        st.session_state.ingest_frames = []
        st.session_state.ingest_live = []

    # Advance the stream when the button is pressed OR autoplay is on
    # (each rerun generates NEW frames — the genuinely dynamic path).
    if _advance or _auto_ingest:
        for _ in range(_n_frames):
            f = node.step()
            if f is not None:
                st.session_state.ingest_frames.append(f.to_dataframe_row())

    frames = st.session_state.ingest_frames
    if len(frames) < 30:
        st.info(f"Buffering live frames — {len(frames)}/30 before the ensemble can score.")
    else:
        import pandas as _pd
        _df_ing = _pd.DataFrame(frames[-200:])   # rolling window, capped
        try:
            from missionmind.ml.detect import score_dataframe
            _sc = score_dataframe(_df_ing)
            _latest = _sc.iloc[-1]
            _score_v = float(_latest.get("anomaly_score", 0.0))
            _flag_v = int(_latest.get("anomaly_flag", 0))
            _src_v = int(_latest.get("anomaly_source", 0))
            _t_last = int(_latest["time_s"])
            # Same documented burn-in convention as the main dashboard: the
            # start-up thermal transient (t<100) flags in every run, so suppress
            # the display flag there — injection only starts at t=600s.
            if _t_last < 100:
                _flag_v = 0
        except Exception as _e:
            _score_v, _flag_v, _src_v, _t_last = 0.0, 0, 0, 0

        st.session_state.ingest_live.append((_t_last, _score_v, _flag_v))
        st.session_state.ingest_live = st.session_state.ingest_live[-400:]

        _ic = st.columns(4)
        _ic[0].markdown(_kpi("Frames Received", f"{len(frames)}", "", "virtual-edge-01", "delta-flat"), unsafe_allow_html=True)
        _ic[1].markdown(_kpi("Mission Clock", f"{int(_latest['time_s'])}", "s", "streaming", "delta-flat"), unsafe_allow_html=True)
        _ic[2].markdown(_kpi("Anomaly Score", f"{_score_v:.3f}", "", "live window", "delta-neg" if _flag_v else "delta-pos"), unsafe_allow_html=True)
        _ic[3].markdown(_kpi("Flag", "ANOMALY" if _flag_v else "NOMINAL", "", f"source {_src_v}", "delta-neg" if _flag_v else "delta-pos"), unsafe_allow_html=True)

        if PLOTLY_AVAILABLE and len(st.session_state.ingest_live) > 1:
            _tdf = _pd.DataFrame(st.session_state.ingest_live, columns=["t", "score", "flag"])
            _tg = go.Figure()
            _tg.add_trace(go.Scatter(x=_tdf["t"], y=_tdf["score"], name="Anomaly score",
                                     line=dict(color="#a55eea")))
            _tg.add_trace(go.Scatter(x=_tdf["t"], y=_tdf["flag"] * 0.05, name="Flag (scaled)",
                                     line=dict(color="#ff4757", dash="dot")))
            _tg.update_layout(height=260, template="plotly_dark",
                              margin=dict(l=10, r=10, t=24, b=10),
                              paper_bgcolor="#0a101f", plot_bgcolor="#0a101f",
                              font=dict(color="#7d8db1", family="Consolas, monospace"),
                              yaxis=dict(gridcolor="#16213e", zeroline=False),
                              xaxis=dict(gridcolor="#16213e", zeroline=False))
            st.plotly_chart(_tg, use_container_width=True)
        st.caption("Live trend: the ensemble re-scores the rolling window as frames arrive — the score and flag move in real time.")

    st.divider()
    st.markdown("""**Why this matters:** the CSV replay and live-scenario views solve the full mission up front (deterministic, cached).
    This tab is the genuinely dynamic path — every ⏩ Advance computes NEW physics and NEW scores. It is the same code path
    a real ESP32/RPi would drive via `missionmind/telemetry/` (JSON-lines TCP or MQTT).""")

    if _auto_ingest:
        time.sleep(1)
        st.rerun()

with st.expander("\U0001f50d Runtime code trace — which pipeline code ran"):
    try:
        from missionmind.trace import last as trace_last
        evs = trace_last(40)
        if not evs:
            st.caption("No trace events recorded yet — trigger a scenario solve, an alert, or the Live Ingest tab.")
        else:
            lines = []
            for e in evs:
                t = f"T+{int(e.get('mission_t') or 0)//60:02d}:{int(e.get('mission_t') or 0)%60:02d}" \
                    if e.get("mission_t") is not None else "      "
                val = f" = {e['value']}" if e.get("value") is not None else ""
                note = f"  # {e['note']}" if e.get("note") else ""
                lines.append(f"[{e['seq']}] {t}  {e.get('module')}.{e.get('func')}{val}{note}")
            st.code("\n".join(lines), language="text")
        st.caption("Trace buffer is shared with the web console (`/api/trace`) — every event is a real "
                   "call in this session, captured by missionmind/trace.py at the pipeline seams.")
    except Exception as _e:  # noqa: BLE001
        st.caption(f"Trace unavailable: {type(_e).__name__}")

st.divider()
st.markdown("**End-to-end pipeline:** CSV replay → physics slope check → ML ensemble → RAG retrieval → Granite (real or mock) → 3D digital twin + charts + reasoning.")

st.markdown("""<div class="mm-footer">MISSIONMIND · Satellite Mission Operations · synthetic telemetry (constants flagged as assumptions, not flight data) · ML models retrained by <code>python -m missionmind.ml.train</code></div>""", unsafe_allow_html=True)

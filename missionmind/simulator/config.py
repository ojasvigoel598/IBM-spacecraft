"""
MissionMind — Central Config (Fix P3: Hard-coded parameters duplicated)

All physics constants defined once and imported everywhere to avoid drift.
Spec §3, §4, §5 values with both SPEC and DEMO variants.

This addresses E. Critical Errors: hardcoded asserts drift and F. Major: MC_P and epsA fraction tuning.

P5-009 FIX — DEMO_FAST is now toggleable without editing source. Set the
environment variable  MISSIONMIND_PHYSICS_SPEC=1  to use the spec values
(MC_P=5000 J/K, RADIATOR_FINAL_FRACTION=0.30) instead of the demo values.
Default behaviour is unchanged for backwards compatibility with existing
runs and saved models.
"""
import os

# Power Subsystem Spec §3
P_SOLAR_MAX_SPEC = 520.0
P_SOLAR_MAX = 520.0  # W, full illumination (assumption not flight data)
P_LOAD_SPEC = 400.0
P_LOAD = 400.0       # W, constant bus load MVP simplification
E_CAP_WH_SPEC = 100.0
E_CAP_WH = 100.0     # Wh, usable battery capacity (360kJ SI) = 100*3600 J
E_CAP_JOULES = E_CAP_WH * 3600.0
V_MIN_SPEC = 24.0
V_MIN = 24.0         # V at SOC=0
V_MAX_SPEC = 28.0
V_MAX = 28.0         # V at SOC=1
SOC_0 = 0.9
DT_S = 1.0

# Thermal Subsystem Spec §4
MC_P_SPEC = 5000.0   # Spec value J/K
MC_P_DEMO = 2000.0   # Demo fast value for detectability
# P5-009 FIX: read MISSIONMIND_PHYSICS_SPEC env var so an operator can
# flip to spec-faithful physics without editing source. env var wins over
# the literal default, which keeps True for the demo pipeline.
DEMO_FAST = (os.environ.get("MISSIONMIND_PHYSICS_SPEC", "0") != "1")
MC_P = MC_P_DEMO if DEMO_FAST else MC_P_SPEC
ETA = 0.85
EPSILON = 0.85
AREA = 0.5
SIGMA = 5.67e-8
T_SPACE_K = 3.0
T0_C = 25.0
T0_K = T0_C + 273.15
Q_IN_NOMINAL = P_LOAD * (1.0 - ETA)  # 60W

# Failure Injection Spec §5
T_RAMP_START = 600
T_RAMP_END = 900
RAMP_DURATION = T_RAMP_END - T_RAMP_START
SOLAR_FINAL_FACTOR_SPEC = 0.48
SOLAR_FINAL_FACTOR = 0.48

RADIATOR_FINAL_FRACTION_SPEC = 0.30  # Spec: 30% → epsA 0.1275 → eq 28C
RADIATOR_FINAL_FRACTION_DEMO = 0.10  # Demo: 10% → epsA 0.0425 → eq 124C
RADIATOR_FINAL_FRACTION = RADIATOR_FINAL_FRACTION_DEMO if DEMO_FAST else RADIATOR_FINAL_FRACTION_SPEC

EPSILON_A_NOMINAL = EPSILON * AREA  # 0.425
EPSILON_A_FINAL = EPSILON_A_NOMINAL * RADIATOR_FINAL_FRACTION
EPSILON_A_FINAL_SPEC = EPSILON_A_NOMINAL * RADIATOR_FINAL_FRACTION_SPEC
EPSILON_A_FINAL_DEMO = EPSILON_A_NOMINAL * RADIATOR_FINAL_FRACTION_DEMO

# Physics Rules Spec §6
P_SOLAR_MAX_FOR_RULES = P_SOLAR_MAX
SOC_SLOPE_THRESHOLD_SPEC = -0.0005
SOC_SLOPE_THRESHOLD_TUNED = -0.0002
TEMP_SLOPE_THRESHOLD_SPEC = 0.01
TEMP_SLOPE_THRESHOLD_TUNED = 0.003
SOLAR_DROP_THRESHOLD_FACTOR = 0.7  # 0.7*Pmax = 364W
HEAT_IN_STABLE_THRESHOLD = 1.0  # W/s

# ML Spec §7
CONTAMINATION_SPEC = 0.05
CONTAMINATION_TUNED = 0.07
N_ESTIMATORS_SPEC = 200
N_ESTIMATORS_TUNED = 300

print(f"[Config] DEMO_FAST={DEMO_FAST} MC_P {MC_P_SPEC}->{MC_P} RADIATOR_FINAL {RADIATOR_FINAL_FRACTION_SPEC}->{RADIATOR_FINAL_FRACTION}")

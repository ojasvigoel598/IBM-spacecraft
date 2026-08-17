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
MC_P_SPEC = 5000.0   # Spec value J/K (nominal physics)
MC_P_DEMO = 2000.0   # Legacy demo value (was tuned for 1-h detectability)
# P5-009 / P8: physics fidelity selector. DEFAULT = SPEC (physically
# justified nominal values). The demo-tuned values (MC_P 2000 and the
# exaggerated 10% radiator fault) are opt-in via MISSIONMIND_DEMO_FAST=1
# and are labelled as controlled injected faults, not nominal physics.
DEMO_FAST = (os.environ.get("MISSIONMIND_DEMO_FAST", "0") == "1")
if os.environ.get("MISSIONMIND_PHYSICS_SPEC", "0") == "1":
    DEMO_FAST = False   # backwards compatible: force spec explicitly
MC_P = MC_P_SPEC if not DEMO_FAST else MC_P_DEMO
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

RADIATOR_FINAL_FRACTION_SPEC = 0.30  # Spec: 30% -> epsA 0.1275 (realistic injected fault)
RADIATOR_FINAL_FRACTION_DEMO = 0.10  # Exaggerated demo fault (10% -> epsA 0.0425).
# Explicitly a CONTROLLED INJECTED FAULT for 1-hour detectability, NOT a
# nominal-physics parameter. The realistic fault (SPEC 30%) is the default.
RADIATOR_FINAL_FRACTION = (RADIATOR_FINAL_FRACTION_DEMO if DEMO_FAST
                           else RADIATOR_FINAL_FRACTION_SPEC)

EPSILON_A_NOMINAL = EPSILON * AREA  # 0.425
EPSILON_A_FINAL = EPSILON_A_NOMINAL * RADIATOR_FINAL_FRACTION
EPSILON_A_FINAL_SPEC = EPSILON_A_NOMINAL * RADIATOR_FINAL_FRACTION_SPEC
EPSILON_A_FINAL_DEMO = EPSILON_A_NOMINAL * RADIATOR_FINAL_FRACTION_DEMO

# Battery policy (first-order energy-conserving, see power.py)
# A real battery cannot deliver load once depleted: below SOC_SAFE_MODE_ENTER
# the bus sheds non-essential load; at SOC 0 the bus trips (load -> 0) and
# waits for solar recharge above SOC_SAFE_MODE_EXIT (hysteresis).
SOC_SAFE_MODE_ENTER = 0.20   # enter safe mode below this SOC
SOC_SAFE_MODE_EXIT = 0.35    # resume full load above this SOC (hysteresis)
P_LOAD_SAFE = 100.0          # W, safe-mode load (essential bus only)
V_UVLO = 22.0                # V, hardware undervoltage-lockout floor (reference;
                             # the linear model trips the bus at V_MIN=24 V = SOC 0)

# Thermal environment (first-order LEO model, see thermal.py)
# Direct solar, Earth albedo and Earth IR absorbed on the bus; radiative
# rejection on the radiator. First-order engineering fidelity only.
G_SOLAR = 1361.0      # W/m^2 solar constant at 1 AU
ALPHA_S = 0.16        # solar absorptivity of sunlit surfaces (OSR/MLI dominated)
A_SUNLIT = 0.30       # m^2 projected sunlit area
ALBEDO = 0.30         # Earth albedo
F_ALBEDO = 0.15       # view factor to the lit Earth (first-order)
Q_IR_EARTH = 230.0    # W/m^2 Earth IR flux at spacecraft altitude
F_IR = 0.40           # view factor to the Earth (first-order)

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

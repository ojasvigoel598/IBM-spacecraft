"""
MissionMind - Physics Rule-Check Layer
Spec Section 6 - Exact Logic

This layer makes the project more than generic ML: it answers
"is this physically consistent with a known failure mode?"
"""

import numpy as np
import pandas as pd

# P1-003 FIX: Explicit SPEC vs TUNED thresholds to document drift from spec.
# Constants come from config.py (single source of truth, architecture review
# candidate 1) — no hardcoded copies here.
from missionmind.simulator.config import (
    P_SOLAR_MAX,
    SOC_SLOPE_THRESHOLD_SPEC,
    SOC_SLOPE_THRESHOLD_TUNED,
    TEMP_SLOPE_THRESHOLD_SPEC,
    TEMP_SLOPE_THRESHOLD_TUNED,
)

# Log both for audit
print(f"[Physics Rules] Using tuned thresholds: SOC {SOC_SLOPE_THRESHOLD_SPEC}->{SOC_SLOPE_THRESHOLD_TUNED}, TEMP {TEMP_SLOPE_THRESHOLD_SPEC}->{TEMP_SLOPE_THRESHOLD_TUNED} (reason: spec inconsistent with net -150W -> -0.000417 and net +32W -> 0.0064)")

def slope(series_or_values, time_or_none=None):
    """
    Compute slope per second.
    If time provided, linear regression slope.
    If series is array-like, assume 1-sec spacing and use polyfit or simple diff.
    """
    y = np.asarray(series_or_values, dtype=float)
    if len(y) < 2:
        return 0.0
    if time_or_none is not None:
        x = np.asarray(time_or_none, dtype=float)
    else:
        x = np.arange(len(y), dtype=float)

    # polyfit degree 1
    try:
        m, _ = np.polyfit(x, y, 1)
        return float(m)
    except (np.linalg.LinAlgError, ValueError):
        # Numerically degenerate window (constant y or insufficient rank);
        # fall back to plain endpoint slope so the rule still gives a value.
        return float((y[-1] - y[0]) / (x[-1] - x[0] + 1e-9))

def confidence_from(*bools_or_values):
    """
    Simple heuristic: if both conditions strongly met, high confidence.
    Spec says confidence_from(solar_drop, soc_declining) etc.
    We produce a value 0-1 based on magnitude.
    """
    # If inputs are bool, map to base confidences
    # For numeric we can scale: this is placeholder for explainable logic
    # We'll just return average mapped to 0.7-0.95 range when triggered
    conf = 0.85
    # If any explicit numeric confidence passed, average
    nums = [v for v in bools_or_values if isinstance(v, (int,float)) and not isinstance(v,bool)]
    if nums:
        # crude scaling
        conf = float(np.clip(np.mean(np.abs(nums))*2 + 0.6, 0.6, 0.95))
    return round(conf, 2)

def check_power_subsystem(window: pd.DataFrame):
    """
    window = last N seconds of telemetry (e.g. N=120)
    Spec:
      solar_drop = mean solar_power_w < 0.7 * P_solar_max
      soc_declining = slope(battery_soc) < -0.0005
    If both: return 'solar_degradation', confidence
    Else None

    NOTE: Spec constants give dSOC = -150W/3600/100 = -0.000417/s when solar=250W,
    which is slightly less negative than spec threshold -0.0005. Also after battery
    clamps at 0, slope becomes 0. So we tune threshold to -0.0002 and also consider
    SOC mean low as alternative indication. Documented as intentional adjustment
    because spec threshold is inconsistent with spec failure magnitude.
    """
    if window is None or len(window) < 10:
        return None

    solar_mean = window["solar_power_w"].mean()
    solar_drop = solar_mean < 0.7 * P_SOLAR_MAX

    batt_slope = slope(window["battery_soc"].values, window["time_s"].values if "time_s" in window else None)
    soc_mean = window["battery_soc"].mean()
    # Tuned threshold: use TUNED constant, SPEC is -0.0005 but physical -0.000417 requires -0.0002
    soc_declining = batt_slope < SOC_SLOPE_THRESHOLD_TUNED or (soc_mean < 0.5 and batt_slope < 0.0001) or (soc_mean < 0.1)

    t_last = float(window["time_s"].iloc[-1]) if "time_s" in window else None
    if solar_drop and soc_declining:
        conf = 0.6 + 0.2 * (1 - solar_mean/(0.7*P_SOLAR_MAX)) + 0.2 * min(1.0, abs(batt_slope)/0.001 + (0.5 if soc_mean<0.5 else 0))
        conf = float(np.clip(conf, 0.65, 0.95))
        try:
            from missionmind.trace import record
            record("physics_rules", "check_power_subsystem", mission_t=t_last,
                   note=f"solar_drop ({solar_mean:.0f}W) + soc_declining ({batt_slope:.4f}/s) -> solar_degradation",
                   value=conf)
        except Exception:  # noqa: BLE001
            pass
        return ("solar_degradation", round(conf, 2))

    try:
        from missionmind.trace import record
        record("physics_rules", "check_power_subsystem", mission_t=t_last,
               note=f"no power rule hit (solar {solar_mean:.0f}W, soc slope {batt_slope:.4f}/s)",
               value=None)
    except Exception:  # noqa: BLE001
        pass
    return None

def check_thermal_subsystem(window: pd.DataFrame):
    """
    temp_rising = slope(temperature_c) > 0.01 C/s
    heat_in_stable = abs(slope(heat_in_w)) < 1.0 W/s ~flat
    If both => radiator_degradation

    NOTE: With spec constants, Q_in=60W, epsilon*A degraded 0.1275, at 250K Q_out~28W,
    net 32W, dT=0.0064 K/s <0.01 threshold. So threshold too strict. Tune to 0.003
    to match physics. Also consider temp mean rising above nominal -23C benchmark.
    """
    if window is None or len(window) < 10:
        return None

    temp_slope = slope(window["temperature_c"].values, window["time_s"].values if "time_s" in window else None)
    heat_in_slope = slope(window["heat_in_w"].values) if "heat_in_w" in window else 0.0
    temp_mean = window["temperature_c"].mean()

    temp_rising = temp_slope > TEMP_SLOPE_THRESHOLD_TUNED  # tuned from SPEC 0.01
    heat_in_stable = abs(heat_in_slope) < 1.0
    # Additional heuristic: if temp is significantly above nominal equilibrium -23C and rising
    temp_high = temp_mean > -10 and temp_slope > -0.005  # not cooling fast

    t_last = float(window["time_s"].iloc[-1]) if "time_s" in window else None
    if (temp_rising and heat_in_stable) or (temp_high and heat_in_stable and temp_mean > 0):
        conf = 0.6 + 0.3 * min(1.0, temp_slope/0.05 + (0.2 if temp_high else 0))
        conf = float(np.clip(conf, 0.65, 0.95))
        try:
            from missionmind.trace import record
            record("physics_rules", "check_thermal_subsystem", mission_t=t_last,
                   note=f"temp_rising ({temp_slope:.4f}C/s, mean {temp_mean:.1f}C) -> radiator_degradation",
                   value=conf)
        except Exception:  # noqa: BLE001
            pass
        return ("radiator_degradation", round(conf, 2))

    try:
        from missionmind.trace import record
        record("physics_rules", "check_thermal_subsystem", mission_t=t_last,
               note=f"no thermal rule hit (temp slope {temp_slope:.4f}C/s, mean {temp_mean:.1f}C)",
               value=None)
    except Exception:  # noqa: BLE001
        pass
    return None

def check_eclipse(window: pd.DataFrame):
    """P5-ORBIT: is a solar-power dip EXPLAINED by Kepler eclipse physics?

    When the satellite is in eclipse, solar power drops to ~0 W as a matter of
    orbital geometry (not a fault). If the telemetry window shows the satellite
    in eclipse AND solar power is low, this rule returns ('eclipse', conf) —
    the physics EXPLANATION for a solar dip that an unsupervised ML detector
    would otherwise flag as a solar-array fault.

    The adaptive layer uses this to expose ML-vs-physics disagreement:
    ML flag 'solar' + physics 'eclipse' = expected transient, not a fault.
    """
    if window is None or len(window) < 10:
        return None
    if "in_eclipse" not in window.columns or "solar_power_w" not in window.columns:
        return None
    eclipse_frac = float(window["in_eclipse"].mean())
    solar_mean = float(window["solar_power_w"].mean())
    if eclipse_frac > 0.5 and solar_mean < 0.7 * P_SOLAR_MAX:
        t_last = float(window["time_s"].iloc[-1]) if "time_s" in window else None
        conf = round(min(0.95, 0.6 + 0.3 * eclipse_frac), 2)
        try:
            from missionmind.trace import record
            record("physics_rules", "check_eclipse", mission_t=t_last,
                   note=f"in eclipse {eclipse_frac:.0%} with solar {solar_mean:.0f}W -> expected transient",
                   value=conf)
        except Exception:  # noqa: BLE001
            pass
        return ("eclipse", conf)
    return None


def check_all_rules(window: pd.DataFrame):
    """
    Run both subsystem checks, returns dict of findings
    """
    results = {}
    p = check_power_subsystem(window)
    if p:
        results["power"] = {"flag": p[0], "confidence": p[1]}
    else:
        results["power"] = None

    t = check_thermal_subsystem(window)
    if t:
        results["thermal"] = {"flag": t[0], "confidence": t[1]}
    else:
        results["thermal"] = None

    return results

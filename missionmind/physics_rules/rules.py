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
    EPSILON_A_NOMINAL,
    SIGMA,
    T_SPACE_K,
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

    # P9: an eclipse-driven solar drop is EXPECTED physics, not a fault - the
    # eclipse rule (check_eclipse) explains it. Without this guard the power
    # rule would fire 'solar_degradation' on every nominal eclipse pass.
    if "in_eclipse" in window.columns:
        try:
            if float(window["in_eclipse"].astype(float).mean()) > 0.5:
                return None
        except Exception:  # noqa: BLE001
            pass

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

def thermal_rejection_residual(window: pd.DataFrame):
    """Radiator-health diagnostic: measured heat rejection minus the
    Stefan-Boltzmann expectation for a NOMINAL radiator at the measured
    temperature.

        expected_q_out = epsA_nominal * sigma * (T_mean^4 - T_space^4)
        residual        = mean(heat_out_w) - expected_q_out

    This is the physically correct radiator fault signature. A degraded
    radiator reduces eps*A, so it rejects LESS heat at any given temperature
    than the nominal product would - the residual goes strongly negative
    regardless of whether the bus happens to be warming or cooling. The old
    temperature-trend heuristic (temp_rising + heat_in_stable) could not tell
    "bus legitimately warms from load/environment" apart from "radiator
    degraded", so it fired on the environment-coupled nominal run (P10).
    Returns (residual_w, expected_w) or None when heat_out_w is absent.
    """
    if window is None or len(window) < 10:
        return None
    if "heat_out_w" not in window.columns:
        return None
    t_k = window["temperature_c"].astype(float) + 273.15
    # T^4 via multiplication (not `x ** 4`): platform pow() differs in the
    # last ULP across libms; multiplication is IEEE-deterministic everywhere.
    t4 = (t_k * t_k) * (t_k * t_k)
    expected = float((EPSILON_A_NOMINAL * SIGMA
                      * (t4 - (T_SPACE_K * T_SPACE_K) * (T_SPACE_K * T_SPACE_K))).mean())
    measured = float(window["heat_out_w"].astype(float).mean())
    return measured - expected, expected


def check_thermal_subsystem(window: pd.DataFrame):
    """
    radiator_degradation = measured heat rejection is substantially below the
    nominal-radiator Stefan-Boltzmann expectation at the measured temperature
    (thermal_rejection_residual << 0). Confidence scales with the fractional
    deficit. The old SPEC heuristic (temp slope > 0.01 C/s + heat_in flat)
    conflated legitimate warming with degradation, so it false-positived on
    the environment-coupled nominal run; a rising bus with a HEALTHY radiator
    still rejects heat at the nominal rate and produces ~0 residual.
    """
    res = thermal_rejection_residual(window)
    if res is None:
        return None
    residual_w, expected_w = res
    # tolerance: sensor noise on heat_out_w (P2-003 ~ +- a few W) plus a small
    # margin; a degraded radiator (30% epsA) leaves a deficit far beyond this.
    tol = max(8.0, 0.12 * expected_w)
    if residual_w < -tol:
        deficit = min(1.0, -residual_w / max(1.0, expected_w))
        conf = float(np.clip(0.6 + 0.35 * deficit, 0.65, 0.95))
        t_last = float(window["time_s"].iloc[-1]) if "time_s" in window else None
        try:
            from missionmind.trace import record
            record("physics_rules", "check_thermal_subsystem", mission_t=t_last,
                   note=f"heat rejection {residual_w:+.1f}W vs nominal expectation "
                        f"{expected_w:.0f}W (deficit {deficit:.0%}) -> radiator_degradation",
                   value=conf)
        except Exception:  # noqa: BLE001
            pass
        return ("radiator_degradation", round(conf, 2))

    t_last = float(window["time_s"].iloc[-1]) if "time_s" in window else None
    try:
        from missionmind.trace import record
        record("physics_rules", "check_thermal_subsystem", mission_t=t_last,
               note=f"heat rejection {residual_w:+.1f}W vs nominal {expected_w:.0f}W "
                    f"(residual within tolerance) - radiator healthy",
               value=None)
    except Exception:  # noqa: BLE001
        pass
    return None

def eclipse_residual(window: pd.DataFrame) -> dict:
    """Residual of measured solar power vs the physically expected eclipse-
    adjusted value (P_max * sun_exposure, assuming a nominal array).

    This is the quantity that decides whether a solar dip is EXPLAINED by
    orbital geometry or is a genuine fault:
      * in eclipse and |residual| small  -> eclipse explains the dip
      * in eclipse and residual << 0     -> degradation BEYOND the eclipse
        (e.g. penumbra where a degraded array underproduces, or a fault that
        persists when the Sun returns)
      * not in eclipse                   -> the solar channel is unambiguous
    Returns a dict (or None when the columns are absent).
    """
    if window is None or len(window) < 10:
        return None
    need = ("solar_power_w", "in_eclipse", "sun_exposure")
    if not all(c in window.columns for c in need):
        return None
    exposure = window["sun_exposure"].astype(float)
    solar = window["solar_power_w"].astype(float)
    eclipse_frac = float(window["in_eclipse"].astype(float).mean())
    expected = float((P_SOLAR_MAX * exposure).mean())
    measured = float(solar.mean())
    residual = measured - expected
    # tolerance: a nominal array matches the eclipse-adjusted expectation to
    # within sensor noise (P2-003: +-2 W) plus a small margin. A real fault
    # (degradation factor ~0.48) leaves a residual far beyond this.
    tol = max(10.0, 0.15 * abs(expected))
    if eclipse_frac > 0.5:
        status = "eclipse" if residual > -tol else "eclipse_plus_fault"
    else:
        status = "full"
    return {"expected_solar_w": expected, "measured_solar_w": measured,
            "residual_w": residual, "eclipse_frac": eclipse_frac,
            "in_eclipse": eclipse_frac > 0.5, "status": status}


def check_eclipse(window: pd.DataFrame):
    """Eclipse explanation rule (residual-based, P9).

    Returns ('eclipse', conf) ONLY when the measured solar drop is consistent
    with the eclipse-adjusted expectation (residual within tolerance) - i.e.
    the ML flag is genuinely explained by orbital geometry. Returns
    ('eclipse_plus_fault', conf) when the measured power is substantially
    LOWER than the eclipse-adjusted value, so a real fault can never be
    silently erased by an eclipse: the explanation only fires when the
    physics supports it.
    """
    res = eclipse_residual(window)
    if res is None or not res["in_eclipse"]:
        return None
    t_last = float(window["time_s"].iloc[-1]) if "time_s" in window else None
    if res["status"] == "eclipse":
        conf = round(min(0.95, 0.6 + 0.3 * res["eclipse_frac"]), 2)
        try:
            from missionmind.trace import record
            record("physics_rules", "check_eclipse", mission_t=t_last,
                   note=f"in eclipse {res['eclipse_frac']:.0%}, solar {res['measured_solar_w']:.0f}W "
                        f"vs expected {res['expected_solar_w']:.0f}W (residual {res['residual_w']:.1f}W) -> explained",
                   value=conf)
        except Exception:  # noqa: BLE001
            pass
        return ("eclipse", conf)
    # eclipse + measured solar substantially below expectation -> real fault
    conf = round(min(0.97, 0.7 + 0.25 * min(1.0, -res["residual_w"] / max(1.0, res["expected_solar_w"]))), 2)
    try:
        from missionmind.trace import record
        record("physics_rules", "check_eclipse", mission_t=t_last,
               note=f"in eclipse but solar {res['measured_solar_w']:.0f}W << expected "
                    f"{res['expected_solar_w']:.0f}W (residual {res['residual_w']:.1f}W) -> fault NOT explained",
               value=conf)
    except Exception:  # noqa: BLE001
        pass
    return ("eclipse_plus_fault", conf)


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

"""
MissionMind - Power Subsystem Simulator
Spec Section 3 - Exact Model

Constants are reasonable small-satellite starting values, not flight-verified data.
Flagged explicitly as assumptions per checklist guidance.

Model:
- P_solar = P_solar_max * illumination(t) * degradation_factor
- illumination = 1.0 (MVP: constant sun, no eclipse)
- net_power = P_solar - P_load
- dSOC = (net_power * dt / 3600) / E_cap
- SOC clamped [0,1]
- battery_voltage = V_min + (V_max - V_min) * SOC (linear model)
"""

import pandas as pd
import numpy as np

# Centralized config (Fix P3: hard-coded duplication). config.py is the single
# source of truth — if it cannot be imported, fail loudly rather than silently
# binding stale local copies (architecture review candidate 1).
from .config import (
    P_SOLAR_MAX, P_LOAD, E_CAP_WH, E_CAP_JOULES,
    V_MIN, V_MAX, SOC_0, DT_S
)

def illumination(t_s: float) -> float:
    """MVP: constant sun exposure, no eclipse modelling"""
    return 1.0

def compute_power_step(t_s: float, soc: float, degradation_factor: float = 1.0):
    """
    Compute one step of power subsystem.
    Returns (solar_power_w, load_power_w, soc_new, voltage_v, net_power_w)
    """
    solar_w = P_SOLAR_MAX * illumination(t_s) * degradation_factor
    net_w = solar_w - P_LOAD
    d_soc = (net_w * DT_S / 3600.0) / E_CAP_WH
    soc_new = float(np.clip(soc + d_soc, 0.0, 1.0))
    voltage_v = V_MIN + (V_MAX - V_MIN) * soc_new
    return solar_w, P_LOAD, soc_new, voltage_v, net_w

def simulate_power(duration_s: int = 3600, degradation_func=None, soc_init: float = SOC_0):
    """
    Simulate power alone for sanity check.
    degradation_func: callable t -> factor, default 1.0
    Returns DataFrame with columns: time_s, solar_power_w, load_power_w, battery_soc, battery_voltage_v, net_power_w
    """
    if degradation_func is None:
        degradation_func = lambda t: 1.0

    times = []
    solar = []
    load = []
    socs = []
    volts = []
    nets = []

    soc = soc_init
    for t in range(duration_s):
        deg = degradation_func(t)
        s_w, l_w, soc_new, v_v, net_w = compute_power_step(t, soc, deg)
        times.append(t)
        solar.append(s_w)
        load.append(l_w)
        socs.append(soc_new)
        volts.append(v_v)
        nets.append(net_w)
        soc = soc_new

    df = pd.DataFrame({
        "time_s": times,
        "solar_power_w": solar,
        "load_power_w": load,
        "battery_soc": socs,
        "battery_voltage_v": volts,
        "net_power_w": nets,
    })
    return df

if __name__ == "__main__":
    # Sanity check: normal case SOC rises and plateaus near 1.0, voltage near 28V over 3600s
    df = simulate_power(3600)
    print("=== Power Subsystem Sanity Check (NORMAL) ===")
    print(f"Initial SOC: {df['battery_soc'].iloc[0]:.3f}, Final SOC: {df['battery_soc'].iloc[-1]:.3f}")
    print(f"Initial Voltage: {df['battery_voltage_v'].iloc[0]:.2f}V, Final Voltage: {df['battery_voltage_v'].iloc[-1]:.2f}V")
    print(f"Solar: {df['solar_power_w'].iloc[0]}W, Load: {df['load_power_w'].iloc[0]}W, Net: {df['net_power_w'].iloc[0]}W")
    # Expected: SOC should rise to ~1.0
    assert df['battery_soc'].iloc[-1] > 0.95, f"SOC final {df['battery_soc'].iloc[-1]} should be near 1.0"
    assert df['battery_voltage_v'].iloc[-1] > 27.5, f"Voltage final {df['battery_voltage_v'].iloc[-1]} should be near 28.0V"
    print("PASS: SOC rises and plateaus near 1.0, voltage near 28V")

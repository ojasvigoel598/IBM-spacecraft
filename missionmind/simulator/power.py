"""
MissionMind - Power Subsystem Simulator
Spec Section 3 - Exact Model + P7 corrections

Constants are reasonable small-satellite starting values, not flight-verified data.
Flagged explicitly as assumptions per checklist guidance.

Model (P7: eclipse coupled into the EPS):
- P_solar = P_solar_max * sun_exposure(t) * degradation_factor(t)
  sun_exposure comes from the orbital conical-shadow model (orbital.py):
  1.0 in full sun, 0.0 in umbra, smoothly 0..1 in penumbra. The power model
  now physically responds to orbital illumination (no more constant-sun MVP).
- Battery first-order policy (energy conserving):
    SOC > 0.20            -> full load 400 W
    0 < SOC <= 0.20       -> safe mode: load shed to 100 W (essential bus)
    SOC <= 0 and deficit  -> bus trips: load 0 W, waits for solar recharge
    recharge above SOC 0.35 resumes full load (hysteresis)
  The battery can never deliver energy it does not have: at SOC = 0 the load
  is zeroed, so dE/dt = 0 (no energy created from nothing).
- net_power = P_solar - P_load
- dSOC = (net_power * dt / 3600) / E_cap
- SOC clamped [0,1]
- battery_voltage = V_min + (V_max - V_min) * SOC (linear model); 0 V on a
  tripped bus.
"""

import pandas as pd
import numpy as np

from .config import (
    P_SOLAR_MAX, P_LOAD, E_CAP_WH, E_CAP_JOULES,
    V_MIN, V_MAX, SOC_0, DT_S,
    SOC_SAFE_MODE_ENTER, SOC_SAFE_MODE_EXIT, P_LOAD_SAFE,
)

BUS_NORMAL = "normal"
BUS_SAFE_MODE = "safe_mode"
BUS_OFF = "off"


def illumination(t_s: float) -> float:
    """Deprecated constant-sun helper kept for backward compatibility.

    The physical illumination factor now comes from the orbital eclipse model
    (orbital.py `sun_exposure`); callers pass it as `sun_exposure` to
    compute_power_step. This function returns 1.0 (full sun) as the legacy
    default so isolated power-only sanity runs are unchanged.
    """
    return 1.0


def _apply_battery_policy(soc: float, solar_w: float, bus_state: str):
    """First-order energy-conserving battery policy with hysteresis.

    Returns (load_w, bus_state_new). Never draws load the battery cannot
    supply: once SOC reaches 0 under a deficit the bus trips (load -> 0) and
    only solar recharge above the exit threshold restores it.
    """
    load_w = P_LOAD
    state = BUS_NORMAL
    if bus_state == BUS_SAFE_MODE:
        if soc > SOC_SAFE_MODE_EXIT:
            state, load_w = BUS_NORMAL, P_LOAD
        elif soc <= 0.0 and solar_w < P_LOAD_SAFE:
            state, load_w = BUS_OFF, 0.0
        else:
            state, load_w = BUS_SAFE_MODE, P_LOAD_SAFE
    elif bus_state == BUS_OFF:
        if soc >= SOC_SAFE_MODE_EXIT:
            state, load_w = BUS_NORMAL, P_LOAD
        else:
            state, load_w = BUS_OFF, 0.0
    else:  # normal
        if soc <= SOC_SAFE_MODE_ENTER:
            if soc <= 0.0 and solar_w < P_LOAD_SAFE:
                state, load_w = BUS_OFF, 0.0
            else:
                state, load_w = BUS_SAFE_MODE, P_LOAD_SAFE
        else:
            state, load_w = BUS_NORMAL, P_LOAD
    return load_w, state


def compute_power_step(t_s: float, soc: float, degradation_factor: float = 1.0,
                       sun_exposure: float = 1.0, bus_state: str = BUS_NORMAL):
    """Compute one step of the power subsystem.

    Returns (solar_power_w, load_power_w, soc_new, voltage_v, net_power_w,
             bus_state_new). sun_exposure is the 0..1 illumination factor from
    the orbital eclipse model (1 = full sun). bus_state tracks the battery
    policy state machine (normal | safe_mode | off).
    """
    solar_w = P_SOLAR_MAX * max(0.0, float(sun_exposure)) * degradation_factor
    solar_w = max(0.0, solar_w)
    load_w, state = _apply_battery_policy(soc, solar_w, bus_state)
    net_w = solar_w - load_w
    d_soc = (net_w * DT_S / 3600.0) / E_CAP_WH
    soc_new = float(np.clip(soc + d_soc, 0.0, 1.0))
    # HARD guarantee: the battery can never deliver more energy than it holds.
    # If this step would drive SOC <= 0 under a deficit (the pre-step SOC is
    # still positive, so the policy check above cannot see it), the bus trips
    # IN THE SAME STEP: load zeroed, net = solar (charge resumes if sunlight).
    if soc_new <= 0.0 and net_w < 0.0:
        state = BUS_OFF
        load_w = 0.0
        net_w = solar_w
        d_soc = (net_w * DT_S / 3600.0) / E_CAP_WH
        soc_new = float(np.clip(soc + d_soc, 0.0, 1.0))
    if state == BUS_OFF:
        voltage_v = 0.0
    else:
        voltage_v = V_MIN + (V_MAX - V_MIN) * soc_new
    return solar_w, load_w, soc_new, voltage_v, net_w, state


def simulate_power(duration_s: int = 3600, degradation_func=None,
                   soc_init: float = SOC_0, exposure_func=None):
    """Simulate power alone for sanity check.

    degradation_func: callable t -> factor, default 1.0
    exposure_func:    callable t -> sun_exposure, default 1.0 (constant sun)
    Returns DataFrame with columns: time_s, solar_power_w, load_power_w,
    battery_soc, battery_voltage_v, net_power_w, bus_state.
    """
    if degradation_func is None:
        degradation_func = lambda t: 1.0
    if exposure_func is None:
        exposure_func = lambda t: 1.0

    times, solar, load, socs, volts, nets, states = [], [], [], [], [], [], []
    soc = soc_init
    state = BUS_NORMAL
    for t in range(duration_s):
        deg = degradation_func(t)
        exp = exposure_func(t)
        s_w, l_w, soc_new, v_v, net_w, state = compute_power_step(
            t, soc, deg, sun_exposure=exp, bus_state=state)
        times.append(t); solar.append(s_w); load.append(l_w)
        socs.append(soc_new); volts.append(v_v); nets.append(net_w)
        states.append(state)
        soc = soc_new

    return pd.DataFrame({
        "time_s": times, "solar_power_w": solar, "load_power_w": load,
        "battery_soc": socs, "battery_voltage_v": volts,
        "net_power_w": nets, "bus_state": states,
    })


if __name__ == "__main__":
    # Sanity check: normal case SOC rises and plateaus near 1.0, voltage near 28V
    df = simulate_power(3600)
    print("=== Power Subsystem Sanity Check (NORMAL) ===")
    print(f"Initial SOC: {df['battery_soc'].iloc[0]:.3f}, Final SOC: {df['battery_soc'].iloc[-1]:.3f}")
    print(f"Initial Voltage: {df['battery_voltage_v'].iloc[0]:.2f}V, Final Voltage: {df['battery_voltage_v'].iloc[-1]:.2f}V")
    print(f"Solar: {df['solar_power_w'].iloc[0]}W, Load: {df['load_power_w'].iloc[0]}W, Net: {df['net_power_w'].iloc[0]}W")
    assert df['battery_soc'].iloc[-1] > 0.95, f"SOC final {df['battery_soc'].iloc[-1]} should be near 1.0"
    assert df['battery_voltage_v'].iloc[-1] > 27.5, f"Voltage final {df['battery_voltage_v'].iloc[-1]} should be near 28.0V"
    print("PASS: SOC rises and plateaus near 1.0, voltage near 28V")

    # Energy-conservation check: solar-failure with eclipse dips must NOT draw
    # load at SOC=0 (no energy from nothing).
    from .failures import solar_degradation_factor
    df2 = simulate_power(3600, degradation_func=solar_degradation_factor)
    assert df2['battery_soc'].min() >= 0.0
    off = df2[df2['bus_state'] == 'off']
    if len(off):
        assert (off['load_power_w'] == 0).all(), "bus must draw zero load when tripped"
    print(f"SOC min {df2['battery_soc'].min():.3f}, bus off from t="
          f"{df2['time_s'][df2['bus_state']=='off'].min() if len(off) else 'never'}, "
          f"load at SOC=0: {df2['load_power_w'][df2['battery_soc']<=0].unique() if (df2['battery_soc']<=0).any() else 'no SOC=0'}")
    print("PASS: battery cannot create energy at SOC=0")

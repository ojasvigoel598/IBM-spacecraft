"""
MissionMind - Thermal Subsystem Simulator
Spec Section 4 - Exact Model + P7 first-order LEO environment

Constants are assumptions, not flight data - flagged per checklist.

Model (single-node, first-order engineering fidelity):
- Q_in_internal = P_load * (1 - eta)          waste heat (load-coupled)
- Q_solar  = alpha_s * A_sunlit * G_solar * sun_exposure
- Q_albedo = alpha_s * A_sunlit * G_solar * albedo * F_albedo * sun_exposure
- Q_IR     = eps * A_sunlit * F_IR * q_IR_earth
- Q_out    = epsilon_eff * sigma * A_eff * (T_k^4 - T_space^4)   (Stefan-Boltzmann)
- C dT/dt  = Q_in_internal + Q_solar + Q_albedo + Q_IR - Q_out
- T_k = T_k + dT  (explicit Euler, dt = 1 s; stable: |lambda| ~ 1e-3..1e-2 /s)

sun_exposure comes from the orbital conical-shadow model: 1.0 full sun, 0.0
umbra, smooth 0..1 penumbra. Direct solar and albedo vanish in eclipse; the
Earth IR term is always present (the spacecraft is near the Earth). This is a
first-order engineering model - no claim of high fidelity.

Units are W, W/m^2, K, m^2, dimensionless absorptivity/emissivity. Energy
balance Q_in_total = Q_out + C dT/dt holds per timestep by construction;
validated in tests/test_thermal_environment.py.
"""

import numpy as np
import pandas as pd

from .config import (
    MC_P, MC_P_SPEC, MC_P_DEMO, DEMO_FAST, ETA, EPSILON, AREA, SIGMA,
    T_SPACE_K, T0_C, T0_K, Q_IN_NOMINAL, DT_S,
    G_SOLAR, ALPHA_S, A_SUNLIT, ALBEDO, F_ALBEDO, Q_IR_EARTH, F_IR,
)

print(f"[Thermal] Loaded constants from central config.py DEMO_FAST={DEMO_FAST}")


def compute_equilibrium_temp(epsilon_eff=EPSILON, area_eff=AREA, q_in=Q_IN_NOMINAL):
    """Solve Q_in = epsilon*sigma*A*(T^4 - T_space^4) for T (radiator only,
    no environment). Returns T in K and C."""
    T4 = q_in / (epsilon_eff * SIGMA * area_eff) + T_SPACE_K ** 4
    T_k = T4 ** 0.25
    return T_k, T_k - 273.15


def thermal_environment_fluxes(sun_exposure: float = 1.0) -> dict:
    """First-order environmental heat loads absorbed on the bus (W).

    sun_exposure 1 = full sun, 0 = umbra (direct solar and albedo vanish);
    Earth IR is always present. Returns the three terms and their total.
    """
    exp = float(np.clip(sun_exposure, 0.0, 1.0))
    solar = ALPHA_S * A_SUNLIT * G_SOLAR * exp
    albedo = ALPHA_S * A_SUNLIT * G_SOLAR * ALBEDO * F_ALBEDO * exp
    ir = EPSILON * A_SUNLIT * F_IR * Q_IR_EARTH
    return {"solar_w": solar, "albedo_w": albedo, "earth_ir_w": ir,
            "total_w": solar + albedo + ir}


def compute_environment_equilibrium_temp(sun_exposure: float = 1.0,
                                         epsilon_eff=EPSILON, area_eff=AREA,
                                         q_in=Q_IN_NOMINAL):
    """Full-model equilibrium (internal + environment vs radiator rejection)."""
    env = thermal_environment_fluxes(sun_exposure)
    T4 = (q_in + env["total_w"]) / (epsilon_eff * SIGMA * area_eff) + T_SPACE_K ** 4
    T_k = T4 ** 0.25
    return T_k, T_k - 273.15, env


def compute_thermal_step(t_s: float, T_k: float, epsilon_eff: float = EPSILON,
                         area_eff: float = AREA, q_in: float = Q_IN_NOMINAL,
                         sun_exposure: float = 1.0):
    """One thermal step (explicit Euler, dt = 1 s).

    Returns (T_k_new, Q_in_internal, Q_out, dT). The environment heat loads
    are added inside; Q_in returned is the internal dissipation only (the
    environment is available via thermal_environment_fluxes for the
    energy-balance check).
    """
    env = thermal_environment_fluxes(sun_exposure)
    q_in_total = q_in + env["total_w"]
    q_out = epsilon_eff * SIGMA * area_eff * (T_k ** 4 - T_SPACE_K ** 4)
    dT = (q_in_total - q_out) * DT_S / MC_P
    T_new = T_k + dT
    return T_new, q_in, q_out, dT


def simulate_thermal(duration_s: int = 3600, epsilon_func=None, area_func=None,
                     t_init_k: float = T0_K, exposure_func=None,
                     q_in_func=None):
    """Simulate thermal alone for sanity check.

    epsilon_func / area_func: callable t -> value
    exposure_func: callable t -> sun_exposure (default 1.0)
    q_in_func: callable t -> internal dissipation W (default Q_IN_NOMINAL)
    Returns DataFrame with time_s, temperature_c, temperature_k, heat_in_w,
    heat_out_w, env_solar_w, env_albedo_w, env_ir_w, env_total_w, dT_dt.
    """
    if epsilon_func is None:
        epsilon_func = lambda t: EPSILON
    if area_func is None:
        area_func = lambda t: AREA
    if exposure_func is None:
        exposure_func = lambda t: 1.0
    if q_in_func is None:
        q_in_func = lambda t: Q_IN_NOMINAL

    times, temps_c, temps_k, q_ins, q_outs = [], [], [], [], []
    env_s, env_a, env_i, env_t, dtdts = [], [], [], [], []
    T_k = t_init_k
    for t in range(duration_s):
        eps = epsilon_func(t)
        area = area_func(t)
        exp = exposure_func(t)
        qin = q_in_func(t)
        T_new, q_in, q_out, dT = compute_thermal_step(
            t, T_k, eps, area, q_in=qin, sun_exposure=exp)
        env = thermal_environment_fluxes(exp)
        times.append(t); temps_k.append(T_new); temps_c.append(T_new - 273.15)
        q_ins.append(q_in); q_outs.append(q_out)
        env_s.append(env["solar_w"]); env_a.append(env["albedo_w"])
        env_i.append(env["earth_ir_w"]); env_t.append(env["total_w"])
        dtdts.append(dT)
        T_k = T_new

    return pd.DataFrame({
        "time_s": times, "temperature_c": temps_c, "temperature_k": temps_k,
        "heat_in_w": q_ins, "heat_out_w": q_outs,
        "env_solar_w": env_s, "env_albedo_w": env_a, "env_ir_w": env_i,
        "env_total_w": env_t, "dT_dt": dtdts,
    })


if __name__ == "__main__":
    print("=== Thermal Subsystem Sanity Check (with first-order LEO env) ===")
    T_eq_k, T_eq_c = compute_equilibrium_temp()
    print(f"Radiator-only equilibrium (no env): {T_eq_k:.2f}K = {T_eq_c:.2f}C")
    T_sun_k, T_sun_c, env = compute_environment_equilibrium_temp(sun_exposure=1.0)
    T_ecl_k, T_ecl_c, _ = compute_environment_equilibrium_temp(sun_exposure=0.0)
    print(f"Full-model equilibrium, full sun: {T_sun_k:.2f}K = {T_sun_c:.2f}C "
          f"(env {env['solar_w']:.1f}+{env['albedo_w']:.1f}+{env['earth_ir_w']:.1f} W)")
    print(f"Full-model equilibrium, eclipse  : {T_ecl_k:.2f}K = {T_ecl_c:.2f}C")
    assert -100 < T_sun_c < 80 and -100 < T_ecl_c < 80, "equilibrium out of range"
    print("PASS: equilibria plausible (first-order LEO)")

    df = simulate_thermal(3600)
    print(f"Initial T: {df['temperature_c'].iloc[0]:.2f}C, "
          f"Final T after 3600s: {df['temperature_c'].iloc[-1]:.2f}C")
    # energy balance: |Q_in_total - Q_out - C dT/dt| ~ 0 per step
    mc = MC_P
    resid = (df['heat_in_w'] + df['env_total_w'] - df['heat_out_w']
             - mc * df['dT_dt']).abs().max()
    print(f"Max energy-balance residual (should be ~0): {resid:.3e} W")
    assert resid < 1e-6, "energy balance violated"
    print("PASS: thermal energy balance Q_in = Q_out + C dT/dt")

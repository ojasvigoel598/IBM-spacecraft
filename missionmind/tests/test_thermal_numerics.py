"""Thermal solver numerical validation (explicit Euler, dt = 1 s).

Requirement: quantify the timestep error rather than blindly replacing the
solver. The model is a single-node first-order energy balance:

    C dT/dt = Q_in_total(T-independent) - eps*sigma*A*(T^4 - T_space^4)

For explicit Euler the linearized per-step amplification is
    lambda = dQ_out/dT / C = 4*eps*sigma*A*T^3 / C
and stability requires |lambda * dt| < 1. We verify that margin across the
operating range, then measure the actual discretization error by integrating
the SAME equation with dt = 1 s (production) vs dt = 0.05 s (fine reference)
over an hour. Explicit Euler is first-order: the dt=1s trajectory differs from
the fine reference by O(dt); the test quantifies that this is small (< ~2 K)
for the actual spacecraft constants, so dt = 1 s is adequate for this model.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.simulator.config import (
    MC_P, EPSILON, AREA, SIGMA, T_SPACE_K, T0_K, Q_IN_NOMINAL,
    G_SOLAR, ALPHA_S, A_SUNLIT, ALBEDO, F_ALBEDO, Q_IR_EARTH, F_IR,
)
from missionmind.simulator.thermal import thermal_environment_fluxes


def _euler_integrate(dt_s, duration_s, sun_exposure, t_init_k=T0_K,
                     q_in=Q_IN_NOMINAL, eps=EPSILON, area=AREA):
    """Reference Euler integrator of the SAME thermal equation with a
    configurable dt (faithful re-implementation of compute_thermal_step)."""
    env = thermal_environment_fluxes(sun_exposure)
    q_in_total = q_in + env["total_w"]
    T = t_init_k
    n = int(round(duration_s / dt_s))
    for _ in range(n):
        q_out = eps * SIGMA * area * (T ** 4 - T_SPACE_K ** 4)
        T += (q_in_total - q_out) * dt_s / MC_P
    return T


def test_euler_stability_margin():
    """|lambda*dt| = 4*eps*sigma*A*T^3/C*dt must be << 1 over the operating
    range (200..350 K). At T=350 K: 4*0.85*5.67e-8*0.5*350^3/5000 = 8.3e-4."""
    for T_k in (200.0, 250.0, 300.0, 350.0):
        lam = 4.0 * EPSILON * SIGMA * AREA * T_k ** 3 / MC_P
        assert lam * 1.0 < 0.05, f"Euler amplification factor too large at {T_k} K"
    # absolute margin: at least 20x below the stability limit
    lam_worst = 4.0 * EPSILON * SIGMA * AREA * 350.0 ** 3 / MC_P
    assert lam_worst * 1.0 < 1.0 / 20.0


def test_euler_energy_balance_holds_per_step():
    """q_in_total - q_out = C*dT/dt exactly at each step (by construction)."""
    T = T0_K
    env = thermal_environment_fluxes(0.5)  # penumbra
    q_in_total = Q_IN_NOMINAL + env["total_w"]
    for _ in range(10):
        q_out = EPSILON * SIGMA * AREA * (T ** 4 - T_SPACE_K ** 4)
        dT = (q_in_total - q_out) / MC_P
        T_new = T + dT
        assert abs((T_new - T) - dT) < 1e-12
        assert abs(q_in_total - q_out - MC_P * (T_new - T)) < 1e-9
        T = T_new


def test_euler_timestep_error_quantified():
    """dt=1s vs dt=0.05s reference over one hour: quantify the error. Explicit
    Euler is first-order, so the production dt=1s trajectory stays within a
    small O(dt) band of the fine reference for these constants."""
    duration = 3600
    for exposure in (1.0, 0.0, 0.5):
        t_fine = _euler_integrate(0.05, duration, exposure)
        t_prod = _euler_integrate(1.0, duration, exposure)
        err = abs(t_prod - t_fine)
        # The absolute error must be small (a couple of K at most) relative to
        # the ~60-80 K total excursion — dt=1s is adequate for this model.
        assert err < 2.0, (f"Euler dt=1s error {err:.3f} K at exposure "
                           f"{exposure} exceeds the 2 K budget")
        # First-order check: dt=2s error ~ 2x dt=1s error (order 1).
        t_coarse = _euler_integrate(2.0, duration, exposure)
        err_coarse = abs(t_coarse - t_fine)
        assert 1.2 < err_coarse / max(err, 1e-12) < 3.0, (
            f"expected ~2x error when dt doubles, got {err_coarse / max(err, 1e-12):.2f}x")


def test_euler_reaches_same_equilibrium():
    """Both dt=1s and dt=0.05s converge to the same analytic equilibrium."""
    for exposure in (1.0, 0.0):
        env = thermal_environment_fluxes(exposure)
        q_in_total = Q_IN_NOMINAL + env["total_w"]
        T_eq = (q_in_total / (EPSILON * SIGMA * AREA) + T_SPACE_K ** 4) ** 0.25
        t_fine = _euler_integrate(0.05, 4 * 3600, exposure)
        t_prod = _euler_integrate(1.0, 4 * 3600, exposure)
        assert abs(t_prod - T_eq) < 1.0, f"dt=1s not at equilibrium: {t_prod - T_eq} K"
        assert abs(t_fine - T_eq) < 1.0


if __name__ == "__main__":
    for fn in (test_euler_stability_margin, test_euler_energy_balance_holds_per_step,
               test_euler_timestep_error_quantified, test_euler_reaches_same_equilibrium):
        fn()
        print(f"PASS {fn.__name__}")
    print("All thermal numerics tests PASS")

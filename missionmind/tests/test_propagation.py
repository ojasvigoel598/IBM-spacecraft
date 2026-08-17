"""Validation of the numerical propagator (missionmind/simulator/propagation.py).

The runtime baseline is the analytical two-body Kepler solution (exact, zero
drift). These tests prove the numerical extension point is correct so it can
be trusted when perturbations (J2, and later drag/SRP) are enabled:

  * RK4 converges to the analytical Kepler state at the expected order (~4).
  * Energy / angular momentum drift bounds over 10 orbits.
  * The analytical baseline conserves energy exactly (the contrast that
    justifies keeping it as the runtime propagator).
  * J2 secular nodal regression matches the analytic rate
    Omega_dot = -1.5 n J2 (R/a)^2 cos i.
  * The adaptive DOPRI5 stepper is accurate and more efficient than fixed RK4.
  * Multiple LEO altitudes and a non-zero eccentricity behave correctly.
  * Eclipse state from the numerical trajectory matches the analytical one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.simulator.orbital import (
    orbital_period_s, state_vectors_3d, eclipse_geometry, J2, R_EARTH, MU_EARTH,
)
from missionmind.simulator.propagation import (
    state_from_elements, propagate_rk4, propagate_adaptive, elements_from_state,
    eclipsed_at,
)

REF_A = 6.921e6


def _one_orbit_error(dt, a=REF_A, e=0.0, include_j2=False):
    """RK4 vs the analytical solution at the same covered time (n*dt)."""
    T = orbital_period_s(a)
    n = int(round(T / dt))
    t_end = n * dt
    r0 = state_from_elements(a=a, e=e)
    r_end = propagate_rk4(r0, t_end, dt, include_j2=include_j2)
    r_ana = state_vectors_3d(t_end, a=a, e=e)[0]
    return float(np.linalg.norm(r_end[:3] - r_ana))


def test_rk4_converges_to_analytical_kepler():
    err10 = _one_orbit_error(10.0)
    err5 = _one_orbit_error(5.0)
    assert err10 < 0.1, f"dt=10 one-orbit error {err10:.4f} m should be < 0.1 m"
    assert err5 < err10 / 8.0, "halving dt must cut the error by ~16x"
    order = np.log2(err10 / err5)
    assert 3.0 < order < 5.0, f"RK4 order measured {order:.2f}, expected ~4"


def test_rk4_energy_and_angular_momentum_drift():
    T = orbital_period_s(REF_A)
    r0 = state_from_elements(a=REF_A)
    r_end = propagate_rk4(r0, 10.0 * T, 30.0)
    e0 = elements_from_state(r0)
    e1 = elements_from_state(r_end)
    rel_e = abs((e1["specific_energy"] - e0["specific_energy"]) / e0["specific_energy"])
    rel_h = abs((e1["h"] - e0["h"]) / e0["h"])
    assert rel_e < 1e-6, f"RK4 relative energy drift {rel_e:.2e} over 10 orbits"
    assert rel_h < 1e-6, f"RK4 relative h drift {rel_h:.2e} over 10 orbits"


def test_analytical_conserves_energy_exactly():
    """The analytical baseline must beat any integrator: energy and h are
    constant to machine precision along the trajectory."""
    T = orbital_period_s(REF_A)
    e0 = elements_from_state(state_from_elements(a=REF_A))
    for t in (0.0, 0.3 * T, 0.7 * T, 5.0 * T):
        r, v, _, _ = state_vectors_3d(t, a=REF_A)
        s = np.concatenate([r, v])
        el = elements_from_state(s)
        assert abs((el["specific_energy"] - e0["specific_energy"])
                   / e0["specific_energy"]) < 1e-9
        assert abs((el["h"] - e0["h"]) / e0["h"]) < 1e-9


def test_j2_nodal_regression_matches_analytic_rate():
    T = orbital_period_s(REF_A)
    r0 = state_from_elements(a=REF_A)
    r_j2 = propagate_rk4(r0, T, 10.0, include_j2=True)
    om0 = elements_from_state(r0)["raan_deg"]
    om1 = elements_from_state(r_j2)["raan_deg"]
    d_om = (om1 - om0 + 180.0) % 360.0 - 180.0
    n_mean = 2.0 * np.pi / T
    analytic = -1.5 * n_mean * J2 * (R_EARTH / REF_A) ** 2 \
        * np.cos(np.radians(51.6)) * 180.0 / np.pi * T
    assert abs(d_om - analytic) < 0.05 * abs(analytic), \
        f"J2 node {d_om:.4f} deg vs analytic {analytic:.4f} deg"


def test_adaptive_dopri5_accurate_and_efficient():
    T = orbital_period_s(REF_A)
    r0 = state_from_elements(a=REF_A)
    r_ad, steps = propagate_adaptive(r0, T, dt0=10.0, tol=1e-8)
    r_ana = state_vectors_3d(T, a=REF_A)[0]
    err = float(np.linalg.norm(r_ad[:3] - r_ana))
    assert err < 5.0, f"DOPRI5 one-orbit error {err:.3f} m"
    assert steps < 200, f"DOPRI5 used {steps} steps, expected far fewer than RK4's 573"


def test_multiple_altitudes_and_eccentricity():
    for alt_m in (300e3, 550e3, 1200e3):
        a = R_EARTH + alt_m
        err = _one_orbit_error(10.0, a=a)
        assert err < 0.1, f"{alt_m/1e3:.0f} km: one-orbit error {err:.4f} m"
    err_e = _one_orbit_error(10.0, a=REF_A, e=0.1)
    assert err_e < 0.1, f"e=0.1 one-orbit error {err_e:.4f} m"


def test_eclipse_state_consistent_between_numerical_and_analytical():
    """The eclipse entry/exit derived from a numerical trajectory must agree
    with the analytical eclipse geometry: over one orbit, the eclipse
    fractions differ by < 1%."""
    T = orbital_period_s(REF_A)
    r0 = state_from_elements(a=REF_A)
    frac_ana, frac_num, n = 0.0, 0.0, 0
    for t in np.arange(0.0, T, 60.0):
        r_ana = state_vectors_3d(float(t), a=REF_A)[0]
        frac_ana += eclipse_geometry(r_ana)["eclipse_state"] != "full"
        # propagate the same 6-vector to the same time
        r_num = propagate_rk4(r0, float(t), 10.0)[:3]
        frac_num += eclipsed_at(r_num)
        n += 1
    f_ana = frac_ana / n
    f_num = frac_num / n
    assert abs(f_num - f_ana) < 0.01, \
        f"eclipse fraction mismatch: numerical {f_num:.2%} vs analytical {f_ana:.2%}"


if __name__ == "__main__":
    for fn in (test_rk4_converges_to_analytical_kepler,
               test_rk4_energy_and_angular_momentum_drift,
               test_analytical_conserves_energy_exactly,
               test_j2_nodal_regression_matches_analytic_rate,
               test_adaptive_dopri5_accurate_and_efficient,
               test_multiple_altitudes_and_eccentricity,
               test_eclipse_state_consistent_between_numerical_and_analytical):
        fn()
        print(f"PASS {fn.__name__}")
    print("All propagation tests PASS")

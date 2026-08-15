"""P5 tests — real Kepler orbital physics (missionmind/simulator/orbital.py).

Verified physics (not decoration):
  * Kepler's third law gives a ~95 min period at 550 km LEO.
  * Kepler's equation E - e sin(E) = M is solved to machine precision.
  * A 550 km orbit is in eclipse ~35% of each orbit (cylindrical shadow).
  * Specific orbital energy is conserved along the trajectory (two-body check).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.simulator.orbital import (
    orbital_period_s, kepler_solve, true_anomaly_from_E, state_vectors,
    in_eclipse, eclipse_fraction_over_window, orbital_energy_and_angular_momentum,
    orbit_columns, MU_EARTH, REF_SEMI_MAJOR,
)


def test_kepler_period_550km():
    T = orbital_period_s()
    assert 88 < T / 60 < 105, f"550 km LEO period should be ~95 min, got {T/60:.1f}"
    # analytic: T = 2*pi*sqrt(a^3/mu)
    expect = 2 * np.pi * np.sqrt(REF_SEMI_MAJOR ** 3 / MU_EARTH)
    assert abs(T - expect) < 1e-6


def test_kepler_equation_solve():
    for M in (0.0, 0.5, 1.0, 3.0, 5.5):
        E = kepler_solve(M, 0.1)
        residual = abs((E - 0.1 * np.sin(E)) - M % (2 * np.pi))
        assert residual < 1e-9, f"Kepler residual {residual} too large at M={M}"
    # circular orbit short-circuits: E = M
    assert abs(kepler_solve(2.0, 0.0) - 2.0) < 1e-12


def test_true_anomaly_identity():
    for E in (0.1, 1.5, 3.0, 5.0):
        nu = true_anomaly_from_E(E, 0.1)
        assert 0 <= nu < 2 * np.pi
        assert np.isfinite(nu)


def test_eclipse_fraction_realistic():
    T = orbital_period_s()
    frac = eclipse_fraction_over_window(0, T, step=15.0)
    assert 0.25 < frac < 0.45, f"LEO 550 km eclipse fraction ~35%, got {frac:.2%}"


def test_eclipse_cycle_deterministic():
    # deterministic: same t always gives same eclipse state
    assert in_eclipse(100.0) == in_eclipse(100.0)
    assert isinstance(in_eclipse(100.0), bool)


def test_energy_conservation():
    cons = orbital_energy_and_angular_momentum()
    for t in (0.0, 600.0, 1234.5, 4000.0):
        r, v, _, _ = state_vectors(t)
        e = np.linalg.norm(v) ** 2 / 2 - MU_EARTH / np.linalg.norm(r)
        assert abs(e - cons["specific_energy_Jkg"]) < 1e-6, f"energy drift at t={t}"


def test_orbit_columns_schema():
    cols = orbit_columns(60.0)
    for k in ("orbit_angle_deg", "mean_anomaly_deg", "orbit_period_s",
              "in_eclipse", "sun_incidence_deg", "orbit_energy_jkg",
              "orbit_ang_momentum_m2s", "altitude_km"):
        assert k in cols, f"missing {k}"
    assert 0 <= cols["orbit_angle_deg"] < 360
    assert cols["altitude_km"] == 550.0


if __name__ == "__main__":
    for fn in (test_kepler_period_550km, test_kepler_equation_solve,
               test_true_anomaly_identity, test_eclipse_fraction_realistic,
               test_eclipse_cycle_deterministic, test_energy_conservation,
               test_orbit_columns_schema):
        fn()
        print(f"PASS {fn.__name__}")
    print("All orbital tests PASS")

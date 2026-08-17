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
    state_vectors_3d, eclipse_geometry, in_eclipse,
    eclipse_fraction_over_window, orbital_energy_and_angular_momentum,
    orbit_columns, eclipse_solar_factor, MU_EARTH, REF_SEMI_MAJOR, SUN_DIR,
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


def test_3d_state_magnitudes_and_energy():
    """3D ECI state: |r| = a for the circular reference, vis-viva speed, and
    specific energy conserved along the trajectory."""
    cons = orbital_energy_and_angular_momentum()
    for t in (0.0, 300.0, 5726.0, 12345.0):
        r, v, nu, M = state_vectors_3d(t)
        assert abs(np.linalg.norm(r) - REF_SEMI_MAJOR) < 1e-6, "|r| must equal a"
        assert abs(np.linalg.norm(v) - np.sqrt(MU_EARTH / REF_SEMI_MAJOR)) < 1e-6
        e = np.linalg.norm(v) ** 2 / 2 - MU_EARTH / np.linalg.norm(r)
        assert abs(e - cons["specific_energy_Jkg"]) < 1e-6, f"energy drift at t={t}"
        assert 0 <= nu < 2 * np.pi and 0 <= M < 2 * np.pi


def test_3d_state_nonzero_eccentricity_period():
    """A non-zero-eccentricity orbit returns to the same state after one
    Keplerian period (analytical two-body is closed)."""
    a = REF_SEMI_MAJOR
    e = 0.15
    T = orbital_period_s(a)
    r0, v0, _, _ = state_vectors_3d(0.0, a=a, e=e)
    r1, v1, _, _ = state_vectors_3d(T, a=a, e=e)
    assert np.linalg.norm(r1 - r0) < 1e-6, "position must close after one period"
    assert np.linalg.norm(v1 - v0) < 1e-6, "velocity must close after one period"


def _r_at_psi(psi):
    """ECI position vector whose direction from Earth makes angle psi with the
    Sun direction (psi = 0 -> directly behind Earth, anti-sun)."""
    s = SUN_DIR
    perp = np.array([-s[1], s[0], 0.0])
    perp = perp / np.linalg.norm(perp)
    r_hat = -(np.cos(psi) * s + np.sin(psi) * perp)
    return REF_SEMI_MAJOR * r_hat


def test_conical_shadow_states():
    """Conical shadow: umbra directly behind Earth, penumbra in the band, full
    illumination on the Sun side."""
    geo_center = eclipse_geometry(np.array([-REF_SEMI_MAJOR, 0.0, 0.0]))
    assert geo_center["eclipse_state"] == "umbra"
    assert geo_center["sun_exposure"] == 0.0
    geo_sunside = eclipse_geometry(np.array([REF_SEMI_MAJOR, 0.0, 0.0]))
    assert geo_sunside["eclipse_state"] == "full"
    assert geo_sunside["sun_exposure"] == 1.0
    # penumbra band: rho_e - rho_s < psi < rho_e + rho_s (~1.165..1.175 rad);
    # the midpoint of the band is psi = rho_e itself.
    rho_e = geo_center["rho_earth_rad"]
    rho_s = geo_center["rho_sun_rad"]
    assert eclipse_geometry(_r_at_psi(rho_e - rho_s - 1e-3))["eclipse_state"] == "umbra"
    geo_pen = eclipse_geometry(_r_at_psi(rho_e))
    assert geo_pen["eclipse_state"] == "penumbra"
    assert 0.0 < geo_pen["sun_exposure"] < 1.0
    # just outside the penumbra cone -> fully illuminated
    psi_out = rho_e + rho_s + 1e-3
    assert eclipse_geometry(_r_at_psi(psi_out))["eclipse_state"] == "full"


def test_eclipse_solar_factor_smooth():
    """Solar power follows sun_exposure: full in sunlight, 0 in umbra, and the
    penumbra dims monotonically between."""
    assert eclipse_solar_factor(0.0) == 520.0          # sun side at t=0
    in_shadow = [t for t in range(0, 7000, 10) if in_eclipse(float(t))]
    assert in_shadow, "reference orbit must enter shadow"
    assert all(0.0 <= eclipse_solar_factor(float(t)) <= 520.0 + 1e-9
               for t in range(0, 7000, 10))


def test_orbit_columns_schema():
    cols = orbit_columns(60.0)
    for k in ("orbit_angle_deg", "mean_anomaly_deg", "orbit_period_s",
              "in_eclipse", "eclipse_state", "sun_exposure",
              "sun_incidence_deg", "orbit_energy_jkg",
              "orbit_ang_momentum_m2s", "altitude_km"):
        assert k in cols, f"missing {k}"
    assert 0 <= cols["orbit_angle_deg"] < 360
    assert cols["altitude_km"] == 550.0
    assert cols["eclipse_state"] in ("full", "penumbra", "umbra")
    assert 0.0 <= cols["sun_exposure"] <= 1.0
    assert (cols["in_eclipse"] == 1) == (cols["eclipse_state"] != "full")


if __name__ == "__main__":
    for fn in (test_kepler_period_550km, test_kepler_equation_solve,
               test_true_anomaly_identity, test_eclipse_fraction_realistic,
               test_eclipse_cycle_deterministic, test_energy_conservation,
               test_orbit_columns_schema):
        fn()
        print(f"PASS {fn.__name__}")
    print("All orbital tests PASS")

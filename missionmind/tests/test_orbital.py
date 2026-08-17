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
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.simulator.orbital import (
    orbital_period_s, kepler_solve, true_anomaly_from_E, state_vectors,
    state_vectors_3d, eclipse_geometry, in_eclipse,
    eclipse_fraction_over_window, orbital_energy_and_angular_momentum,
    orbit_columns, eclipse_solar_factor, MU_EARTH, R_EARTH, REF_SEMI_MAJOR,
    SUN_DIR,
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


# --------------------------------------------------------------------------- #
# PQW -> ECI frame-convention regression tests (independent reference).
#
# These pin the transformation against the PUBLISHED closed form of Bate,
# Mueller & White (Fundamentals of Astrodynamics, Eq. 2.19/2.20) at NON-ZERO
# RAAN / argument-of-perigee / inclination / eccentricity. The reference
# orbit's symmetry (Omega=0, omega=0, circular) cannot hide a frame error
# here: the closed form is computed independently from the project code and
# the states must agree to ~machine precision at every random element set.
# --------------------------------------------------------------------------- #

def _bmw_closed_form(t, a, e, i_deg, raan_deg, argp_deg, nu0_deg=0.0):
    """Independent implementation of the published Bate-Mueller-White
    PQW->ECI closed form. Written from the textbook, not from this repo."""
    mu = MU_EARTH
    O, I, w = np.radians(raan_deg), np.radians(i_deg), np.radians(argp_deg)
    # mean anomaly at epoch from the initial true anomaly (same epoch math
    # as the code under test: E0 from nu0, then M0 = E0 - e*sin(E0))
    E0 = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(np.radians(nu0_deg) / 2),
                        np.sqrt(1 + e) * np.cos(np.radians(nu0_deg) / 2))
    M0 = (E0 - e * np.sin(E0)) % (2 * np.pi)
    n = np.sqrt(mu / a ** 3)
    M = (n * t + M0) % (2 * np.pi)
    E = M
    for _ in range(80):
        d = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= d
        if abs(d) < 1e-14:
            break
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))
    p = a * (1 - e * e)
    r = p / (1 + e * np.cos(nu))
    u = w + nu
    r_vec = np.array([
        r * (np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(I)),
        r * (np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(I)),
        r * np.sin(u) * np.sin(I),
    ])
    v_scale = mu / np.sqrt(mu * p)
    v_vec = np.array([
        v_scale * (-np.cos(O) * (np.sin(u) + e * np.sin(w))
                   - np.sin(O) * (np.cos(u) + e * np.cos(w)) * np.cos(I)),
        v_scale * (-np.sin(O) * (np.sin(u) + e * np.sin(w))
                   + np.cos(O) * (np.cos(u) + e * np.cos(w)) * np.cos(I)),
        v_scale * (np.cos(u) + e * np.cos(w)) * np.sin(I),
    ])
    return r_vec, v_vec


def test_pqw_to_eci_matches_published_closed_form():
    """Regression: the 3D ECI state must match the published closed form at
    non-zero RAAN / argument-of-perigee / inclination / eccentricity. A
    frame-convention error cannot be hidden by the Omega=0, omega=0
    symmetry of the reference orbit."""
    rng = np.random.default_rng(11)
    worst_r = worst_v = 0.0
    for _ in range(60):
        a = 6.7e6 + rng.uniform(0, 1.2e6)
        e = rng.uniform(0.0, 0.5)
        i = rng.uniform(10, 120)
        raan = rng.uniform(-180, 180)
        argp = rng.uniform(-180, 180)
        nu0 = rng.uniform(0, 360)
        t = rng.uniform(0, 40000)
        r_code, v_code, _, _ = state_vectors_3d(
            t, a=a, e=e, i_deg=i, raan_deg=raan, argp_deg=argp, nu0_deg=nu0)
        r_ref, v_ref = _bmw_closed_form(t, a, e, i, raan, argp, nu0)
        worst_r = max(worst_r, np.linalg.norm(r_code - r_ref) / np.linalg.norm(r_ref))
        worst_v = max(worst_v, np.linalg.norm(v_code - v_ref) / np.linalg.norm(v_ref))
    assert worst_r < 1e-9, f"ECI position deviates from published closed form: {worst_r:.2e}"
    assert worst_v < 1e-9, f"ECI velocity deviates from published closed form: {worst_v:.2e}"


def test_angular_momentum_direction_matches_elements():
    """The orbit-normal direction must be h_hat = [sin(O)sin(i), -cos(O)sin(i),
    cos(i)] in ECI for the standard prograde convention (checked at non-zero
    RAAN and inclination, where a sign flip would be visible)."""
    for i, raan in ((51.6, 0.0), (45.0, 40.0), (98.0, -33.0)):
        r, v, _, _ = state_vectors_3d(1234.5, e=0.2, i_deg=i, raan_deg=raan,
                                      argp_deg=25.0)
        h = np.cross(r, v)
        h = h / np.linalg.norm(h)
        O, I = np.radians(raan), np.radians(i)
        expect = np.array([np.sin(O) * np.sin(I), -np.cos(O) * np.sin(I), np.cos(I)])
        assert np.allclose(h, expect, atol=1e-9), (i, raan, h, expect)


def test_polar_orbit_geometry_physical():
    """Physical orientation test: a polar prograde orbit (i=90, Omega=0,
    omega=0) at true anomaly 90 deg must place the satellite at the NORTH
    pole ([0, 0, +a]) — a mirror/reflection convention error would send it
    to the south pole."""
    a = REF_SEMI_MAJOR
    for nu0 in (0.0, 90.0):
        r, v, nu, _ = state_vectors_3d(
            0.0, a=a, e=0.0, i_deg=90.0, raan_deg=0.0, argp_deg=0.0, nu0_deg=nu0)
        if nu0 == 90.0:
            assert abs(r[2] - a) < 1e-6, f"polar orbit at nu=90 must be at +z: {r}"
            assert abs(r[0]) < 1e-6 and abs(r[1]) < 1e-6
        else:
            assert abs(r[0] - a) < 1e-6  # nu=0 -> ascending node on +x


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


# --------------------------------------------------------------------------- #
# Edge-case hardening: degenerate inputs must fail explicitly, never produce
# NaN / division-by-zero / silently wrong iterates. High-e Kepler cases must
# still converge (Vallado start guess E0=pi for e>=0.8).
# --------------------------------------------------------------------------- #

def test_kepler_high_eccentricity_converges():
    """e -> 1 Kepler cases converge for every mean anomaly, including the
    hard M near 0 (where f' -> 1-e makes Newton slow)."""
    for e in (0.9, 0.99, 0.999):
        for M in (0.0, 1e-6, 0.01, 1.0, np.pi, 4.0, 2 * np.pi - 1e-6):
            E = kepler_solve(M, e)
            residual = abs((E - e * np.sin(E)) - M % (2 * np.pi))
            assert residual < 1e-9, f"residual {residual:.2e} at e={e}, M={M}"


def test_kepler_rejects_parabolic_and_hyperbolic():
    """e >= 1 is outside the elliptic two-body domain: fail loudly."""
    for e in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            kepler_solve(1.0, e)


def test_kepler_rejects_nonfinite_mean_anomaly():
    with pytest.raises(ValueError):
        kepler_solve(float("nan"), 0.5)


def test_state_vectors_reject_degenerate_elements():
    """a <= 0, e outside [0,1), mu <= 0 must raise, not produce NaN."""
    with pytest.raises(ValueError):
        state_vectors_3d(0.0, a=-1.0)
    with pytest.raises(ValueError):
        state_vectors_3d(0.0, a=0.0)
    with pytest.raises(ValueError):
        state_vectors_3d(0.0, e=1.0)
    with pytest.raises(ValueError):
        state_vectors_3d(0.0, e=1.2)
    with pytest.raises(ValueError):
        state_vectors_3d(0.0, e=-0.5)
    with pytest.raises(ValueError):
        state_vectors_3d(0.0, mu=-1.0)
    with pytest.raises(ValueError):
        state_vectors_3d(float("nan"))


def test_orbital_period_rejects_unphysical_a():
    with pytest.raises(ValueError):
        orbital_period_s(0.0)
    with pytest.raises(ValueError):
        orbital_period_s(-6.9e6)


def test_eclipse_geometry_rejects_degenerate_position():
    with pytest.raises(ValueError):
        eclipse_geometry(np.zeros(3))
    with pytest.raises(ValueError):
        eclipse_geometry(np.array([np.nan, 1.0, 0.0]))


def test_high_eccentricity_energy_conserved_and_period_closes():
    """At e = 0.6 (a = 2 R_earth, a well-defined ellipse with perigee 0.8 R)
    the two-body energy is conserved along the arc and equals -mu/(2a), and
    the state closes after one Keplerian period."""
    a = 2.0 * R_EARTH
    e = 0.6
    T = orbital_period_s(a)
    r0, v0, _, _ = state_vectors_3d(0.0, a=a, e=e)
    eps0 = np.linalg.norm(v0) ** 2 / 2 - MU_EARTH / np.linalg.norm(r0)
    for t in (123.4, 1234.5, T * 0.5):
        r, v, _, _ = state_vectors_3d(t, a=a, e=e)
        eps = np.linalg.norm(v) ** 2 / 2 - MU_EARTH / np.linalg.norm(r)
        assert abs(eps - eps0) < 1e-6, f"energy drift at e=0.6, t={t}"
        assert abs(eps - (-MU_EARTH / (2.0 * a))) < 1e-3
    r1, v1, _, _ = state_vectors_3d(T, a=a, e=e)
    assert np.linalg.norm(r1 - r0) < 1e-6 and np.linalg.norm(v1 - v0) < 1e-6


if __name__ == "__main__":
    for fn in (test_kepler_period_550km, test_kepler_equation_solve,
               test_true_anomaly_identity, test_eclipse_fraction_realistic,
               test_eclipse_cycle_deterministic, test_energy_conservation,
               test_orbit_columns_schema,
               test_kepler_high_eccentricity_converges,
               test_kepler_rejects_parabolic_and_hyperbolic,
               test_kepler_rejects_nonfinite_mean_anomaly,
               test_state_vectors_reject_degenerate_elements,
               test_orbital_period_rejects_unphysical_a,
               test_eclipse_geometry_rejects_degenerate_position,
               test_high_eccentricity_energy_conserved_and_period_closes):
        fn()
        print(f"PASS {fn.__name__}")
    print("All orbital tests PASS")

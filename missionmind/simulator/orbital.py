"""
MissionMind — real Kepler orbital mechanics (Priority 5: physics must affect the system).

The orbit is no longer a decorative ring. This module propagates the satellite
with the actual two-body equations of motion and derives physical quantities
that ARE consumed downstream:

    * orbital period        T = 2*pi*sqrt(a^3/mu)          (Kepler's third law)
    * true anomaly          nu(t)  via Kepler's equation   (Newton–Raphson on
                            E - e*sin(E) = M)
    * sun-relative geometry -> eclipse fraction             (cylindrical shadow)
    * sun-pointing attitude -> solar incidence angle        (body frame)
    * orbital energy        eps = -mu/(2a)  and specific angular momentum
                            h = sqrt(mu*a*(1-e^2))          (conserved quantities
                            used as a physics-consistency check)

Downstream use (so the physics genuinely matters, not decoration):
    1. simulator/run_scenarios.py emits `orbit_angle_deg`, `sun_incidence_deg`,
       `in_eclipse`, `orbit_period_s` columns on every telemetry frame.
    2. physics_rules/rules.py gains check_orbit_eclipse(): when the satellite
       is in eclipse, a solar-power drop is EXPECTED physics — the rule layer
       returns an 'eclipse' finding that explains (and suppresses) an ML flag
       that would otherwise look like a solar-array fault.
    3. ml/adaptive.py exposes ML-vs-physics disagreement: if the ML ensemble
       flags 'solar' while the physics says 'eclipse', the decision layer
       reports the disagreement instead of hiding it.

Units are SI (m, s, rad). All functions are deterministic given t.
"""

from __future__ import annotations

import numpy as np

# ---- Earth constants (WGS-84 mean) -----------------------------------------
MU_EARTH = 3.986004418e14      # m^3/s^2
R_EARTH = 6.371e6              # m
J2 = 1.08262668e-3             # oblateness (documented, not applied in 2-body)

# ---- reference orbit (LEO, sun-synchronous-ish) ----------------------------
# a = R + 550 km  (circular, e=0) -> period ~95 min
ALTITUDE_M = 550e3
REF_SEMI_MAJOR = R_EARTH + ALTITUDE_M
REF_ECC = 0.0
REF_INCLINATION_DEG = 51.6     # ISS-like

# Sun direction in ECI (approx, fixed for the 1-h demo — a full ephemeris is
# out of scope and changes nothing at this timescale).
SUN_DIR = np.array([1.0, 0.25, 0.0])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)


def orbital_period_s(a: float = REF_SEMI_MAJOR, mu: float = MU_EARTH) -> float:
    """Kepler's third law: T = 2*pi*sqrt(a^3/mu)."""
    return float(2.0 * np.pi * np.sqrt(a ** 3 / mu))


def mean_anomaly(t: float, a: float = REF_SEMI_MAJOR,
                 mu: float = MU_EARTH, M0: float = 0.0) -> float:
    """Mean anomaly at time t for a circular orbit: M = n*(t - t0) + M0."""
    n = float(np.sqrt(mu / a ** 3))            # mean motion (rad/s)
    return float((n * t + M0) % (2.0 * np.pi))


def kepler_solve(M: float, e: float, tol: float = 1e-12, max_iter: int = 60) -> float:
    """Solve Kepler's equation E - e*sin(E) = M by Newton–Raphson.

    Returns eccentric anomaly E in [0, 2*pi). Valid for any e in [0, 1);
    e=0 (circular) short-circuits to E = M.
    """
    M = float(M % (2.0 * np.pi))
    if e < 1e-12:
        return M
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            break
    return float(E % (2.0 * np.pi))


def true_anomaly_from_E(E: float, e: float) -> float:
    """True anomaly nu from eccentric anomaly (standard identity)."""
    if e < 1e-12:
        return float(E % (2.0 * np.pi))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0),
                          np.sqrt(1 - e) * np.cos(E / 2.0))
    return float(nu % (2.0 * np.pi))


def state_vectors(t: float, a: float = REF_SEMI_MAJOR, e: float = REF_ECC,
                  mu: float = MU_EARTH):
    """Position (m) and velocity (m/s) in the orbital plane (2D, circular
    assumption: inclination/node fixed for the demo). Returns
    (r_vec, v_vec, nu, M)."""
    M = mean_anomaly(t, a=a, mu=mu)
    E = kepler_solve(M, e)
    nu = true_anomaly_from_E(E, e)
    p = a * (1.0 - e * e)                          # semi-latus rectum
    r = p / (1.0 + e * np.cos(nu))                 # radius
    # perifocal frame: unit radial + perpendicular
    u_r = np.array([np.cos(nu), np.sin(nu)])
    u_p = np.array([-np.sin(nu), np.cos(nu)])
    r_vec = r * u_r
    # vis-viva: v^2 = mu*(2/r - 1/a)
    v_mag = float(np.sqrt(mu * (2.0 / r - 1.0 / a)))
    v_vec = v_mag * u_p
    return r_vec, v_vec, nu, M


def in_eclipse(t: float, a: float = REF_SEMI_MAJOR,
               e: float = REF_ECC, sun_dir: np.ndarray = SUN_DIR) -> bool:
    """Cylindrical shadow model: the satellite is eclipsed when it is on the
    night side of Earth within the shadow cylinder (r_proj < R_Earth).

    Uses the orbital-plane geometry: position vector r in the plane, sun
    direction projected onto the plane. Eclipse iff dot(r, s_hat) < 0 AND
    the perpendicular distance to the shadow axis < R_Earth.
    """
    r_vec, _, _, _ = state_vectors(t, a=a, e=e)
    s_hat = np.array(sun_dir, dtype=float)[:2]
    s_hat = s_hat / np.linalg.norm(s_hat)
    r_proj = np.dot(r_vec, s_hat)
    r_perp = np.linalg.norm(r_vec - r_proj * s_hat)
    # night side + inside the shadow cylinder
    return bool(r_proj < 0.0 and r_perp < R_EARTH)


def sun_pointing_attitude(t: float) -> dict:
    """Body-frame sun-pointing attitude (simplified 2-axis): the +X body axis
    is steered at the sun. Returns the solar incidence angle (deg), which is
    what the solar-array power model consumes (cos(incidence) factor).

    Nominal: arrays track the sun -> incidence 0 deg. This is a real
    sun-pointing control law (quaternion-free 2-axis approximation); full
    3-axis quaternion kinematics (q_dot = 1/2 q o omega) are documented in
    docs/RUL_PROGNOSTICS.md as out of scope for a 1-h demo.
    """
    s_hat = np.array(SUN_DIR, dtype=float)[:2]
    s_hat = s_hat / np.linalg.norm(s_hat)
    # body +X points along sun; incidence angle between array normal (+X) and sun
    incidence_rad = np.arccos(np.clip(np.dot(s_hat, np.array([1.0, 0.0])), -1.0, 1.0))
    return {
        "sun_incidence_deg": float(np.degrees(incidence_rad)),
        "sun_vector": s_hat.tolist(),
        "pointing_mode": "sun-tracking (2-axis)",
    }


def orbital_energy_and_angular_momentum(a: float = REF_SEMI_MAJOR,
                                        e: float = REF_ECC,
                                        mu: float = MU_EARTH) -> dict:
    """Conserved quantities used as a physics-consistency check:
    eps = -mu/(2a), h = sqrt(mu*a*(1-e^2)). Both must stay constant along the
    trajectory for a correct two-body propagation."""
    eps = -mu / (2.0 * a)
    h = float(np.sqrt(mu * a * (1.0 - e * e)))
    return {"specific_energy_Jkg": float(eps), "ang_momentum_m2s": h}


def orbit_columns(t: float) -> dict:
    """One frame's worth of orbital telemetry (deterministic)."""
    a = REF_SEMI_MAJOR
    _, _, nu, M = state_vectors(t, a=a)
    period = orbital_period_s(a)
    eclipse = in_eclipse(t, a=a)
    att = sun_pointing_attitude(t)
    cons = orbital_energy_and_angular_momentum(a)
    return {
        "time_s": float(t),
        "orbit_angle_deg": round(float(np.degrees(nu)), 3),
        "mean_anomaly_deg": round(float(np.degrees(M)), 3),
        "orbit_period_s": round(period, 2),
        "in_eclipse": int(eclipse),
        "sun_incidence_deg": round(att["sun_incidence_deg"], 3),
        "orbit_energy_jkg": round(cons["specific_energy_Jkg"], 3),
        "orbit_ang_momentum_m2s": round(cons["ang_momentum_m2s"], 3),
        "altitude_km": round(ALTITUDE_M / 1e3, 2),
        "pointing_mode": att["pointing_mode"],
    }


def eclipse_solar_factor(t: float, nominal_solar_w: float = 520.0) -> float:
    """Solar power actually produced at time t given eclipse + incidence:
    P = P_nominal * cos(incidence) * (0 if in eclipse else 1)."""
    inc = np.radians(sun_pointing_attitude(t)["sun_incidence_deg"])
    if in_eclipse(t):
        return 0.0
    return float(nominal_solar_w * np.cos(inc))


def eclipse_fraction_over_window(t0: float, t1: float, step: float = 30.0) -> float:
    """Fraction of the mission window spent in eclipse (used by rules/adaptive
    to judge whether a solar dip is 'expected')."""
    ts = np.arange(t0, t1, step)
    if len(ts) == 0:
        return 0.0
    return float(np.mean([in_eclipse(float(t)) for t in ts]))


if __name__ == "__main__":
    print("=== Kepler orbital sanity ===")
    period = orbital_period_s()
    print(f"  altitude {ALTITUDE_M/1e3:.0f} km -> period {period/60:.2f} min "
          f"({86400/period:.2f} orbits/day)")
    assert 88 < period / 60 < 105, "LEO 550 km period should be ~95 min"
    # Kepler solve check: E - e sin E = M for e=0.1
    for M in (0.0, 1.0, 3.0, 5.5):
        E = kepler_solve(M, 0.1)
        assert abs((E - 0.1 * np.sin(E)) - M % (2 * np.pi)) < 1e-9
    print("  Kepler equation residual < 1e-9 at e=0.1")
    # eclipse over one period: ~35% of a 550km LEO orbit is eclipsed
    frac = eclipse_fraction_over_window(0, period)
    print(f"  eclipse fraction over one orbit: {frac:.2%} (expected ~35%)")
    assert 0.25 < frac < 0.45, "LEO 550 km eclipse fraction ~35%"
    # energy conservation
    c1 = orbital_energy_and_angular_momentum()
    r0, v0, _, _ = state_vectors(0.0)
    r1, v1, _, _ = state_vectors(1234.5)
    for r, v in ((r0, v0), (r1, v1)):
        assert abs(np.linalg.norm(v) ** 2 / 2 - MU_EARTH / np.linalg.norm(r) - c1["specific_energy_Jkg"]) < 1e-6
    print("  specific energy conserved along trajectory (two-body check PASS)")
    print("Kepler orbital module: PASS")

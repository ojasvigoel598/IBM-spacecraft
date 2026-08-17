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
REF_RAAN_DEG = 0.0             # ascending node on the ECI +X axis
REF_ARG_PERI_DEG = 0.0         # argument of perigee (reference frame only)

# Sun: fixed ECI direction (approx for the 1-h demo — a full ephemeris changes
# nothing at this timescale) plus the physical Sun radius/distance, which the
# conical shadow model needs.
SUN_DIR = np.array([1.0, 0.25, 0.0])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)
R_SUN = 6.957e8                 # m (photosphere radius)
D_SUN = 1.496e11                # m (1 AU, mean Earth-Sun distance)


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


def _pqw_to_eci(i_deg: float, raan_deg: float, argp_deg: float) -> np.ndarray:
    """Perifocal (PQW) to ECI rotation matrix: R = Rz(-Omega) Rx(-i) Rz(-omega)."""
    O = np.radians(raan_deg)
    I = np.radians(i_deg)
    w = np.radians(argp_deg)
    cO, sO, cI, sI, cw, sw = np.cos(O), np.sin(O), np.cos(I), np.sin(I), np.cos(w), np.sin(w)
    return np.array([
        [cO * cw - sO * sw * cI, -cO * sw - sO * cw * cI, sO * sI],
        [sO * cw + cO * sw * cI, -sO * sw + cO * cw * cI, -cO * sI],
        [sw * sI, cw * sI, cI],
    ])


def state_vectors(t: float, a: float = REF_SEMI_MAJOR, e: float = REF_ECC,
                  mu: float = MU_EARTH):
    """Position (m) and velocity (m/s) in the orbital plane (2D). Kept for
    backward compatibility; the 3D ECI state is `state_vectors_3d`."""
    _, _, nu, M = state_vectors_3d(t, a=a, e=e, mu=mu)
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))
    u_r = np.array([np.cos(nu), np.sin(nu)])
    u_p = np.array([-np.sin(nu), np.cos(nu)])
    return r * u_r, np.sqrt(mu * (2.0 / r - 1.0 / a)) * u_p, nu, M


def state_vectors_3d(t: float, a: float = REF_SEMI_MAJOR, e: float = REF_ECC,
                     i_deg: float = REF_INCLINATION_DEG,
                     raan_deg: float = REF_RAAN_DEG,
                     argp_deg: float = REF_ARG_PERI_DEG,
                     nu0_deg: float = 0.0,
                     mu: float = MU_EARTH):
    """Two-body state in ECI (SI units) from classical orbital elements.

    Solves Kepler's equation analytically (M = E - e sin E via Newton-Raphson)
    and rotates the perifocal state to ECI with the standard
    Rz(-Omega) Rx(-i) Rz(-omega) sequence. Returns (r_vec, v_vec, nu, M).
    """
    # mean anomaly at epoch from the given initial true anomaly
    E0 = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(np.radians(nu0_deg) / 2.0),
                          np.sqrt(1.0 + e) * np.cos(np.radians(nu0_deg) / 2.0))
    M0 = float((E0 - e * np.sin(E0)) % (2.0 * np.pi))
    M = mean_anomaly(t, a=a, mu=mu, M0=M0)
    E = kepler_solve(M, e)
    nu = true_anomaly_from_E(E, e)
    p = a * (1.0 - e * e)                          # semi-latus rectum
    r = p / (1.0 + e * np.cos(nu))                 # radius
    # perifocal frame state
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    R = _pqw_to_eci(i_deg, raan_deg, argp_deg)
    return R @ r_pf, R @ v_pf, nu, M


def eclipse_geometry(r_vec: np.ndarray,
                     sun_dir: np.ndarray = SUN_DIR,
                     r_sun: float = R_SUN, d_sun: float = D_SUN,
                     r_earth: float = R_EARTH) -> dict:
    """Conical shadow model (finite Sun, point satellite).

    Let rho_e / rho_s be the angular radii of Earth and the Sun as seen from
    the satellite and psi the angle between the direction to Earth's centre
    (-r_hat) and the direction to the Sun. Then, by simple disk geometry:
        psi < rho_e - rho_s  -> umbra   (Sun fully covered)
        rho_e - rho_s <= psi < rho_e + rho_s -> penumbra (partial)
        psi >= rho_e + rho_s -> full illumination
    The penumbra light factor is the fraction of the Sun's disk NOT covered by
    Earth, from the two-circle overlap integral.
    """
    r_mag = float(np.linalg.norm(r_vec))
    e_hat = -r_vec / r_mag                        # toward Earth's centre
    s_dir = np.asarray(sun_dir, dtype=float)
    s_dir = s_dir / np.linalg.norm(s_dir)
    r_sun_vec = d_sun * s_dir
    s_hat = (r_sun_vec - r_vec) / np.linalg.norm(r_sun_vec - r_vec)  # toward Sun
    rho_e = float(np.arcsin(np.clip(r_earth / r_mag, -1.0, 1.0)))
    rho_s = float(np.arcsin(np.clip(r_sun / d_sun, -1.0, 1.0)))
    psi = float(np.arccos(np.clip(np.dot(e_hat, s_hat), -1.0, 1.0)))

    if psi >= rho_e + rho_s:
        state, exposure = "full", 1.0
    elif psi <= rho_e - rho_s or psi < 1e-12:
        state, exposure = "umbra", 0.0
    else:
        # penumbra: fraction of the Sun's disk covered by Earth's disk
        R1, R2, d = rho_e, rho_s, psi
        a_c = (d * d + R1 * R1 - R2 * R2) / (2.0 * d)
        h2 = R1 * R1 - a_c * a_c
        if h2 < 0.0:
            h2 = 0.0
        overlap = (R1 * R1 * np.arccos(np.clip(a_c / R1, -1.0, 1.0))
                   + R2 * R2 * np.arccos(np.clip((d - a_c) / R2, -1.0, 1.0))
                   - d * np.sqrt(h2))
        occult = float(np.clip(overlap / (np.pi * R2 * R2), 0.0, 1.0))
        state, exposure = "penumbra", 1.0 - occult

    return {
        "eclipse_state": state,       # 'full' | 'penumbra' | 'umbra'
        "sun_exposure": float(exposure),  # 0..1 fraction of the Sun's disk visible
        "psi_rad": psi,
        "rho_earth_rad": rho_e,
        "rho_sun_rad": rho_s,
    }


def in_eclipse(t: float, a: float = REF_SEMI_MAJOR,
               e: float = REF_ECC, sun_dir: np.ndarray = SUN_DIR) -> bool:
    """True when the satellite is in any shadow (umbra or penumbra) under the
    conical shadow model. Deterministic given t."""
    r_vec, _, _, _ = state_vectors_3d(t, a=a, e=e)
    return bool(eclipse_geometry(r_vec, sun_dir)["eclipse_state"] != "full")


def sun_pointing_attitude(t: float) -> dict:
    """Body-frame sun-pointing attitude: the +X body axis (array normal) is
    steered at the Sun, so the solar incidence angle is 0 deg by construction.

    The power model consumes the illumination/exposure factor instead
    (see eclipse_geometry): in full sunlight cos(incidence)=1, and any Earth
    shadow (umbra/penumbra) dims the array through sun_exposure.
    """
    s_hat = np.asarray(SUN_DIR, dtype=float)
    s_hat = s_hat / np.linalg.norm(s_hat)
    return {
        "sun_incidence_deg": 0.0,
        "sun_vector": s_hat.tolist(),
        "pointing_mode": "sun-tracking (3-axis)",
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
    r_vec, _, nu, M = state_vectors_3d(t, a=a)
    period = orbital_period_s(a)
    geo = eclipse_geometry(r_vec)
    att = sun_pointing_attitude(t)
    cons = orbital_energy_and_angular_momentum(a)
    return {
        "time_s": float(t),
        "orbit_angle_deg": round(float(np.degrees(nu)), 3),
        "mean_anomaly_deg": round(float(np.degrees(M)), 3),
        "orbit_period_s": round(period, 2),
        "in_eclipse": int(geo["eclipse_state"] != "full"),
        "eclipse_state": geo["eclipse_state"],
        "sun_exposure": round(geo["sun_exposure"], 4),
        "sun_incidence_deg": round(att["sun_incidence_deg"], 3),
        "orbit_energy_jkg": round(cons["specific_energy_Jkg"], 3),
        "orbit_ang_momentum_m2s": round(cons["ang_momentum_m2s"], 3),
        "altitude_km": round(ALTITUDE_M / 1e3, 2),
        "pointing_mode": att["pointing_mode"],
    }


def eclipse_solar_factor(t: float, nominal_solar_w: float = 520.0) -> float:
    """Solar power actually produced at time t: P = P_nominal * sun_exposure.

    sun_exposure is the fraction of the Sun's disk visible (1 in full sun,
    0 in umbra, smoothly between in penumbra) from the conical shadow model;
    cos(incidence) = 1 under the sun-tracking attitude law."""
    r_vec, _, _, _ = state_vectors_3d(t)
    return float(nominal_solar_w * eclipse_geometry(r_vec)["sun_exposure"])


def eclipse_fraction_over_window(t0: float, t1: float, step: float = 30.0) -> float:
    """Fraction of the mission window spent in eclipse (used by rules/adaptive
    to judge whether a solar dip is 'expected')."""
    ts = np.arange(t0, t1, step)
    if len(ts) == 0:
        return 0.0
    return float(np.mean([in_eclipse(float(t)) for t in ts]))


if __name__ == "__main__":
    print("=== Kepler orbital sanity (analytical two-body + conical shadow) ===")
    period = orbital_period_s()
    print(f"  altitude {ALTITUDE_M/1e3:.0f} km -> period {period/60:.2f} min "
          f"({86400/period:.2f} orbits/day)")
    assert 88 < period / 60 < 105, "LEO 550 km period should be ~95 min"
    for M in (0.0, 1.0, 3.0, 5.5):
        E = kepler_solve(M, 0.1)
        assert abs((E - 0.1 * np.sin(E)) - M % (2 * np.pi)) < 1e-9
    print("  Kepler equation residual < 1e-9 at e=0.1")
    frac = eclipse_fraction_over_window(0, period)
    print(f"  eclipse fraction over one orbit (conical, umbra+penumbra): {frac:.2%}")
    assert 0.20 < frac < 0.45, "LEO 550 km eclipse fraction ~30-40%"
    # energy conservation (3D ECI state)
    c1 = orbital_energy_and_angular_momentum()
    for t in (0.0, 600.0, 1234.5, 4000.0):
        r, v, _, _ = state_vectors_3d(t)
        e = np.linalg.norm(v) ** 2 / 2 - MU_EARTH / np.linalg.norm(r)
        assert abs(e - c1["specific_energy_Jkg"]) < 1e-6, f"energy drift at t={t}"
    print("  specific energy conserved along trajectory (two-body check PASS)")
    # conical shadow edge cases
    r_geo = eclipse_geometry(np.array([-REF_SEMI_MAJOR, 0.0, 0.0]))   # behind Earth
    assert r_geo["eclipse_state"] == "umbra" and r_geo["sun_exposure"] == 0.0
    r_sunside = eclipse_geometry(np.array([REF_SEMI_MAJOR, 0.0, 0.0]))  # sun side
    assert r_sunside["eclipse_state"] == "full" and r_sunside["sun_exposure"] == 1.0
    print("  conical shadow edge cases PASS (umbra behind Earth, full sun side)")
    print("Kepler orbital module: PASS")

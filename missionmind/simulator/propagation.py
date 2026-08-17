"""
MissionMind — numerical orbital propagator (perturbation extension point).

The runtime baseline is the analytical two-body Kepler propagator in
`orbital.py` (exact, deterministic, zero drift — the closed-form solution of
the two-body problem). This module is the numerical side of the hybrid
architecture the docs/ORBITAL_PROPAGATION.md evaluation recommends:

    analytical Kepler (two-body baseline)  ->  run_scenarios / orbit_columns
    numerical RK4 / adaptive DOPRI5        ->  perturbation-enabled physics
                                               (J2 today; drag, SRP, third
                                               body drop in as extra terms)

Everything here is opt-in and validated against the analytical solution in
missionmind/tests/test_propagation.py:

  * RK4 converges to the analytical Kepler state (position error at one orbit,
    convergence order ~4 when the step is halved).
  * Energy / angular momentum drift bounds over 10 orbits.
  * J2 secular nodal regression matches the analytic rate
    Omega_dot = -1.5 * n * J2 * (R/a)^2 * cos(i).
  * The adaptive DOPRI5 stepper meets a tolerance with fewer steps than fixed
    RK4 at the same accuracy.

Units are SI (m, s, rad). State vectors are 6-element [x,y,z,vx,vy,vz] in ECI.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from missionmind.simulator.orbital import MU_EARTH, R_EARTH, J2, state_vectors_3d

# ---------------------------------------------------------------------------
# Force model: two-body + J2 (documented, opt-in). Each term takes the ECI
# position and returns the ECI acceleration. Adding drag/SRP/third-body means
# adding one more function of the same signature.
# ---------------------------------------------------------------------------


def two_body_accel(r: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Central-body acceleration: a = -mu * r / |r|^3."""
    rm = float(np.linalg.norm(r))
    return -mu * r / rm ** 3


def j2_accel(r: np.ndarray, mu: float = MU_EARTH,
             j2: float = J2, r_e: float = R_EARTH) -> np.ndarray:
    """Oblateness (J2) acceleration in ECI (standard form):

    a = -1.5 J2 mu R^2 / r^5 * [ x(1 - 5z^2/r^2), y(1 - 5z^2/r^2),
                                 z(3 - 5z^2/r^2) ]
    """
    rm = float(np.linalg.norm(r))
    x, y, z = r
    r2 = rm * rm
    f = 1.5 * j2 * mu * r_e * r_e / (rm ** 5)
    return np.array([
        f * x * (5.0 * z * z / r2 - 1.0),
        f * y * (5.0 * z * z / r2 - 1.0),
        f * z * (5.0 * z * z / r2 - 3.0),
    ])


def combined_accel(r: np.ndarray, include_j2: bool = False,
                   mu: float = MU_EARTH, j2: float = J2,
                   r_e: float = R_EARTH) -> np.ndarray:
    a = two_body_accel(r, mu)
    if include_j2:
        a = a + j2_accel(r, mu, j2, r_e)
    return a


# ---------------------------------------------------------------------------
# Integrators. Both take the same 6-vector state and an acceleration callable
# `accel(r) -> a`, so switching force models never touches this code.
# ---------------------------------------------------------------------------


def _deriv(state: np.ndarray, accel: Callable) -> np.ndarray:
    return np.concatenate([state[3:], accel(state[:3])])


def rk4_step(state: np.ndarray, dt: float, accel: Callable) -> np.ndarray:
    """One classical RK4 step. Local error O(dt^5), global O(dt^4)."""
    k1 = _deriv(state, accel)
    k2 = _deriv(state + 0.5 * dt * k1, accel)
    k3 = _deriv(state + 0.5 * dt * k2, accel)
    k4 = _deriv(state + dt * k3, accel)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# Dormand-Prince 5(4) tableau (Hairer / scipy RK45 coefficients).
_DOPRI5_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_DOPRI5_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1 / 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3 / 40, 9 / 40, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44 / 45, -56 / 15, 32 / 9, 0.0, 0.0, 0.0, 0.0],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0, 0.0, 0.0],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0.0, 0.0],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0],
])
_DOPRI5_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192,
                       -2187 / 6784, 11 / 84, 0.0])
_DOPRI5_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640,
                       -92097 / 339200, 187 / 2100, 1 / 40])


def dopri5_step(state: np.ndarray, dt: float, accel: Callable) -> Tuple[np.ndarray, float]:
    """One adaptive DOPRI5 step: returns (5th-order solution, error estimate)."""
    n = len(state)
    k = np.zeros((7, n))
    for i in range(7):
        k[i] = _deriv(state + dt * (_DOPRI5_A[i] @ k), accel)
    y5 = state + dt * (_DOPRI5_B5 @ k)
    y4 = state + dt * (_DOPRI5_B4 @ k)
    err = float(np.max(np.abs(y5 - y4)))
    return y5, err


# ---------------------------------------------------------------------------
# High-level propagators
# ---------------------------------------------------------------------------


def propagate_rk4(state0: np.ndarray, t_end: float, dt: float,
                  accel: Optional[Callable] = None,
                  include_j2: bool = False) -> np.ndarray:
    """Fixed-step RK4 propagation of the 6-vector state to t_end.

    dt must divide t_end; the number of steps is exact so the final state is
    at t = t_end, not t_end - remainder.
    """
    if accel is None:
        accel = lambda r: combined_accel(r, include_j2=include_j2)
    state = np.asarray(state0, dtype=float).copy()
    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        state = rk4_step(state, dt, accel)
    return state


def propagate_adaptive(state0: np.ndarray, t_end: float, dt0: float,
                       accel: Optional[Callable] = None,
                       include_j2: bool = False,
                       tol: float = 1e-8, safety: float = 0.9,
                       dt_min: float = 1e-3, dt_max_factor: float = 10.0,
                       ) -> Tuple[np.ndarray, int]:
    """Adaptive DOPRI5 propagation with step-size control.

    Returns (state at t_end, number of accepted steps). The step is rejected
    (and halved) when the embedded 4th/5th-order difference exceeds tol.
    """
    if accel is None:
        accel = lambda r: combined_accel(r, include_j2=include_j2)
    state = np.asarray(state0, dtype=float).copy()
    dt = float(dt0)
    dt_max = dt0 * dt_max_factor
    t = 0.0
    accepted = 0
    while t < t_end - 1e-12:
        dt = min(dt, t_end - t)
        y5, err = dopri5_step(state, dt, accel)
        scale = tol * max(1.0, float(np.max(np.abs(y5))))
        if err < scale:
            state = y5
            t += dt
            accepted += 1
            if err > 0.0:
                dt = min(dt_max, dt * safety * (scale / err) ** 0.2)
            else:
                dt = min(dt_max, dt * 2.0)
        else:
            dt = max(dt_min, dt * 0.5)
    return state, accepted


# ---------------------------------------------------------------------------
# Element machinery for initialising and interpreting numerical states
# ---------------------------------------------------------------------------


def state_from_elements(a: float = 6.921e6, e: float = 0.0,
                        i_deg: float = 51.6, raan_deg: float = 0.0,
                        argp_deg: float = 0.0, nu0_deg: float = 0.0,
                        mu: float = MU_EARTH) -> np.ndarray:
    """6-vector ECI state from classical orbital elements (analytical)."""
    r, v, _, _ = state_vectors_3d(t=0.0, a=a, e=e, i_deg=i_deg,
                                  raan_deg=raan_deg, argp_deg=argp_deg,
                                  nu0_deg=nu0_deg, mu=mu)
    return np.concatenate([np.asarray(r, float), np.asarray(v, float)])


def elements_from_state(state: np.ndarray, mu: float = MU_EARTH) -> dict:
    """Classical elements from an ECI state (r, v) — used to measure drift.

    Returns a, e, i_deg, raan_deg, specific energy, specific angular momentum.
    RAAN is recovered from the node vector n = z_hat x h, so secular nodal
    precession (J2) is directly observable.
    """
    r = state[:3]
    v = state[3:]
    rm = float(np.linalg.norm(r))
    vm = float(np.linalg.norm(v))
    eps = vm * vm / 2.0 - mu / rm                    # specific energy
    h_vec = np.cross(r, v)
    h = float(np.linalg.norm(h_vec))
    a = -mu / (2.0 * eps)
    e_vec = np.cross(v, h_vec) / mu - r / rm
    e = float(np.linalg.norm(e_vec))
    i = float(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)))
    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)
    n_mag = float(np.linalg.norm(n_vec))
    raan = float(np.arctan2(n_vec[1], n_vec[0])) if n_mag > 1e-12 else 0.0
    return {
        "a": a, "e": e, "i_deg": float(np.degrees(i)),
        "raan_deg": float(np.degrees(raan)) % 360.0,
        "specific_energy": eps, "h": h,
    }


def eclipsed_at(state: np.ndarray, sun_dir: np.ndarray = None) -> bool:
    """Quick eclipse check for a numerical state (delegates to the conical
    shadow geometry in orbital.py)."""
    from missionmind.simulator.orbital import eclipse_geometry, SUN_DIR
    if sun_dir is None:
        sun_dir = SUN_DIR
    return eclipse_geometry(state[:3], sun_dir)["eclipse_state"] != "full"


if __name__ == "__main__":
    print("=== numerical propagator vs analytical Kepler ===\n")
    from missionmind.simulator.orbital import orbital_period_s

    a = 6.921e6
    T = orbital_period_s(a)

    def one_orbit_error(dt, e=0.0):
        """RK4 vs the analytical solution at the same covered time (n*dt)."""
        n = int(round(T / dt))
        t_end = n * dt
        r0 = state_from_elements(a=a, e=e)
        r_end = propagate_rk4(r0, t_end, dt)
        r_ana = state_vectors_3d(t_end, a=a, e=e)[0]
        return float(np.linalg.norm(r_end[:3] - r_ana))

    print(f"reference: a={a/1e3:.0f} km, e=0, i=51.6 deg, T={T:.1f} s ({T/60:.1f} min)")
    errs = {}
    for dt in (30.0, 10.0, 5.0, 2.5):
        e = one_orbit_error(dt)
        errs[dt] = e
        print(f"  RK4 dt={dt:4.1f}s: one-orbit error vs analytical {e:8.3f} m")
    if errs[10.0] > 0 and errs[5.0] > 0:
        order = np.log2(errs[10.0] / errs[5.0])
        print(f"  convergence order (dt 10 -> 5): {order:.2f} (expected ~4)")

    r_ad, steps = propagate_adaptive(state_from_elements(a=a), T, dt0=10.0, tol=1e-8)
    r_ana = state_vectors_3d(T, a=a)[0]
    print(f"  DOPRI5 tol=1e-8: one-orbit error vs analytical "
          f"{np.linalg.norm(r_ad[:3] - r_ana):.3f} m in {steps} steps")

    r_j2 = propagate_rk4(state_from_elements(a=a), T, 10.0, include_j2=True)
    om0 = elements_from_state(state_from_elements(a=a))["raan_deg"]
    om1 = elements_from_state(r_j2)["raan_deg"]
    d_om = (om1 - om0 + 180.0) % 360.0 - 180.0      # unwrap to (-180, 180]
    n_mean = 2.0 * np.pi / T
    om_dot_deg_s = -1.5 * n_mean * J2 * (R_EARTH / a) ** 2 \
        * np.cos(np.radians(51.6)) * 180.0 / np.pi
    print(f"\n  J2 nodal regression over one orbit:")
    print(f"    measured  : {d_om:+.4f} deg")
    print(f"    analytic  : {om_dot_deg_s * T:+.4f} deg  (Omega_dot = "
          f"-1.5 n J2 (R/a)^2 cos i = {om_dot_deg_s:.2e} deg/s)")

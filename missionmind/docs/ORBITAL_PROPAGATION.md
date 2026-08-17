# Orbital propagation: method evaluation and decision

**Question investigated:** does a hybrid architecture — an analytical Kepler
solution for the baseline two-body orbit, plus a numerical integrator (RK4 /
adaptive RK) for perturbed orbits — add real benefit to MissionMind, and if
so, where?

The short answer: **yes, and it is the standard architecture**. The analytical
Kepler propagator is the runtime baseline (it already is), and a *validated,
opt-in* numerical propagator is the perturbation extension point. Perturbation
forces (J2, drag, SRP) are deliberately **not** wired into the runtime
telemetry: at this mission's fidelity they change nothing any consumer
decides, and wiring them in would be decoration, not physics.

This document records the reasoning and the measurements. The implementation
lives in `simulator/orbital.py` (analytical two-body + conical shadow) and
`simulator/propagation.py` (numerical RK4 + adaptive DOPRI5 + J2), with the
validation in `tests/test_orbital.py` and `tests/test_propagation.py`.

## 1. Methods compared

| Method | Local err | Long-term energy | Cost per step | Verdict for this project |
|---|---|---|---|---|
| Explicit Euler | O(h²) | grows (unstable for oscillatory) | trivial | rejected |
| Semi-implicit (symplectic) Euler | O(h²) | bounded oscillation | trivial | fine for games, not for "physics" claims |
| RK2 / midpoint | O(h³) | secular drift | low | rejected |
| **Classical RK4** | O(h⁵) | slow secular drift | 4 evals | chosen for the *perturbed* extension point |
| **Adaptive RK (DOPRI5)** | O(h⁵) | same, with error control | 6-7 evals | chosen for accuracy-demanding perturbed runs |
| High-order symplectic (Yoshida/WHFast) | O(h⁶+) | bounded, no drift | high | only justified for decades-long integrations; overkill here |
| **Analytical Kepler (solve M = E − e sin E)** | **exact** | **zero by construction** | ~4 Newton iterations | **chosen as the runtime baseline** |

Sources consulted: the DLR evaluation methodology compares integrated
solutions against the analytic Kepler solution of the two-body problem
(Bredlau, IAC 2025); Atallah et al. 2019 (UCF) notes the perturbed two-body
problem can only be solved numerically, which is exactly why the numerical
side exists as an extension point; classical references on RK4 vs symplectic
long-term energy behaviour (e.g. University of Rochester PHY411 notes on
integrators) explain why non-symplectic methods drift on multi-year
integrations. None of that applies to the 1-hour LEO baseline here, but the
hybrid keeps the door open.

### Why analytical Kepler wins the baseline

For a pure two-body problem the closed-form solution is *not* an
approximation: every propagated state lies exactly on the same conic, so

* orbital energy ε = −μ/(2a) and specific angular momentum h are conserved to
  machine precision **by construction** (measured < 1e-9 relative over many
  orbits — see §3), with no timestep to tune and no drift to budget;
* cost is O(1) and independent of propagation time — ideal for a browser
  visualisation and for the deterministic telemetry columns;
* it is exactly reproducible: same t → same state, which the physics rules
  and the adaptive layer rely on.

### Why the numerical side exists anyway

The moment a perturbing force is added (J2 oblateness, drag, SRP), the
two-body closed form breaks and numerical integration is the only option. The
sensible structure, matching flight software practice (Orekit, GMAT,
poliastro), is therefore:

    analytical Kepler  ->  two-body baseline (runtime, orbit_columns)
    RK4 / DOPRI5       ->  perturbation-enabled physics (opt-in, validated)

`propagation.py` implements that extension point now, and the numbers in §3
prove it is correct, so a future drag-decay / orbit-lifetime scenario can
switch force models without re-deriving or re-testing the integrator.

## 2. What the current implementation did before this change

* `simulator/orbital.py` already propagated the two-body state analytically
  (Kepler's equation via Newton–Raphson, vis-viva velocity, true anomaly) and
  emitted `orbit_angle_deg`, `in_eclipse`, `orbit_period_s`, energy and
  angular-momentum columns consumed by the physics rules and the
  ML-vs-physics `ECLIPSE_EXPLAINED` strategy.
* Its eclipse model was a **2D cylindrical projection**: the Sun direction was
  flattened into the orbital plane and the shadow test was a cylinder-radius
  threshold. Physically motivated but a simplification.
* The Three.js `applyOrbit()` in `viz/app.py` was a **parametric ring**
  (`period 1200 s`, fixed radius), unrelated to the propagated state — the
  visual was decorative, as the README honestly said.

### What changed

1. `orbital.py`: full **3D ECI state** (perifocal → ECI via
   `Rz(−Ω)Rx(−i)Rz(−ω)`) and a **conical shadow** eclipse model: Earth and Sun
   angular radii as seen from the satellite, umbra / penumbra / full states,
   and a `sun_exposure` factor from the two-circle overlap integral. Measured
   eclipse fraction over one 550 km orbit: **36.98%** (physical expectation
   ~35%). Emitted columns stay additive to `run_scenarios`, so trained
   behaviour is unchanged.
2. `propagation.py`: numerical extension point (RK4, DOPRI5, J2), validated
   in §3.
3. `viz/app.py` `applyOrbit()`: now driven by the propagated true anomaly
   (`orbit_angle_deg`) and the eclipse state, with the ring radius as pure
   visual scale. The visual angle *is* the physics angle.

## 3. Measured validation

Reference orbit: a = 6921 km (550 km LEO), e = 0, i = 51.6°, T = 5730.1 s
(95.5 min, 15.08 orbits/day).

**RK4 vs analytical Kepler (one orbit, error at the same covered time):**

| dt (s) | position error (m) |
|---|---|
| 30 | 1.386 |
| 10 | 0.015 |
| 5 | 0.001 |
| 2.5 | ~0.0001 |

Convergence order measured **4.04** (RK4 theory: 4). At dt = 10 s the
one-orbit error is 1.5 cm — far below any threshold in this project.

**Conservation over 10 orbits:**

| quantity | RK4 (dt = 30 s) | analytical Kepler |
|---|---|---|
| specific energy, relative drift | < 1e-6 | < 1e-9 |
| angular momentum, relative drift | < 1e-6 | < 1e-9 |

**Adaptive DOPRI5 (tol 1e-8):** one-orbit error 0.24 m in **68 steps**
(RK4 needs 573 steps at dt = 10 s for 0.015 m). The adaptive stepper is the
right tool when an accuracy budget matters; it is unnecessary for the
baseline.

**J2 oblateness** (the only perturbation with a secular effect at this
fidelity): over one orbit the ascending node regresses **−0.3084° measured
vs −0.3077° analytic** from Ω̇ = −1.5·n·J2·(R/a)²·cos i — 0.2% agreement.
Over the 1-hour demo that is a sub-degree shift of the sun-relative
geometry, which no consumer threshold is sensitive to. Drag at 550 km
(ρ ~ 1e-13 kg/m³, C_D·A/m ~ 0.02 m²/kg) and solar radiation pressure move
the satellite by at most a few metres over an hour against a 6921 km radius;
they change no decision. This is why perturbations stay off in the runtime
telemetry while the numerical machinery is proven and available.

**Eclipse consistency:** eclipse state (umbra/penumbra/full) computed from
RK4-propagated positions agrees with the analytical trajectory to < 1%
eclipse-fraction over a full orbit — entry/exit times track the propagated
geometry, not a visual toggle.

**Altitudes and eccentricity:** one-orbit RK4 error < 0.1 m at 300 km,
550 km and 1200 km, and for e = 0.1; the analytical state closes to machine
precision after one period for e = 0.15.

## 4. Claims discipline

"Physics-based" is claimed only where the mathematics backs it: the runtime
orbital columns come from the analytical two-body solution and the conical
shadow geometry; the numerical propagator is validated against that solution;
the visualisation consumes the propagated angle and eclipse state rather than
a parametric motion. What is *not* claimed: drag, SRP and third-body effects
are modelled (they are not in the runtime), and eclipse is not injected into
the power/battery solve (that is a deliberate, documented decision — the
rules layer already consumes `in_eclipse` to explain solar dips, and wiring
eclipse into the nominal power envelope would invalidate the trained
detectors for zero fidelity gain at this mission length).

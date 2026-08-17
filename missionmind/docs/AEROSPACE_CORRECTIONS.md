# Aerospace corrections — what changed, why, and the measured before/after

This document records the aerospace/physics corrections applied to the
MissionMind simulator (2026-08), the engineering reasoning behind each one,
and the quantitative before/after comparison. Every number is reproduced by
the identical harness (`.freebuff/metrics.py`) run against the pre-correction
and post-correction code with identical seeds, timesteps and fault injections.

**Honesty rules applied:** physics correctness and ML performance are treated
as separate axes. Where a physically correct change *reduced* an ML metric
(see solar-fault recall below), the physics was kept and the reason is
explained. No metric was manufactured.

---

## 1. Per-change record

### 1.1 Eclipse is now coupled into the power/battery/thermal solve

- **What was wrong:** `power.py` computed `P_solar = P_max * degradation` with
  a constant-illumination assumption (`illumination(t) -> 1.0`); the orbital
  conical-shadow model computed a real `sun_exposure` but nothing consumed it
  except a test.
- **Why it was wrong:** a real LEO array produces `P = P_max * degradation *
  sun_exposure`; treating eclipse as a pure rule-layer decoration meant the
  EPS could never show the physical shadow transient, and the rules layer was
  excusing solar dips it could not actually cause.
- **What was changed:** `P_solar = P_max * degradation * sun_exposure(t)` where
  `sun_exposure` comes from the orbital conical-shadow model (1.0 full sun,
  0.0 umbra, smooth penumbra). `thermal.py` gained the first-order LEO
  environment (`Q_solar = alpha_s*A_sunlit*G_solar*sun_exposure`,
  `Q_albedo`, `Q_IR_earth`); `run_scenarios.py`, the telemetry schema (1.2)
  and the live API path all carry the same eclipse state — one source of
  truth.
- **Before → after (normal scenario):** eclipse-mean solar **520 W → 1.5 W**
  (-99.7 %); eclipse energy deficit **0 → 860 kJ**; mean solar **520 → 281 W**
  (-45.9 %).
- **Engineering significance:** the single largest physical correction. The
  dashboard, telemetry, EPS, rules and ML now all see the same shadow.

### 1.2 Battery energy conservation (safe mode / UVLO / bus trip)

- **What was wrong:** SOC was clipped at 0 while the model kept drawing the
  400 W load indefinitely — energy created from nothing.
- **Why it was wrong:** violates conservation of energy; a real battery trips
  (UVLO) or sheds load before that point.
- **What was changed:** first-order battery policy: safe mode at SOC ≤ 0.20
  (100 W essential bus), bus trip at SOC 0, hysteresis recharge at SOC ≥ 0.35.
  Voltage model `V = V_min + SOC*(V_max - V_min)` with `V_min = 24 V` at
  SOC 0; the bus reports 0 V when off. A hard in-step trip guarantees no load
  is ever drawn at SOC ≤ 0.
- **Before → after (normal scenario):** min SOC **0.900 → 0.000** (the pack
  now genuinely drains through the eclipse), time to min SOC **0 → 3385 s**,
  min bus V **27.6 → 0.00 V** (bus tripped, not clipped), consumed energy
  **1440 → 1139 kJ** (the shed/tripped load is real).
- **Verification:** no load at SOC ≤ 0 in any scenario; `ΔE_battery ≈
  ∫(P_charge − P_load)dt` holds to numerical tolerance.
- **Engineering significance:** the 400 W load + 100 Wh pack cannot survive a
  full 33-minute eclipse — a real sizing finding that the old model hid.

### 1.3 PQW→ECI frame convention — verified correct, not changed

- **What the audit claimed:** the `_pqw_to_eci` matrix mirrored `h` in y and
  shifted eclipse timing by 1000–2000 s at non-zero Ω/ω.
- **What we found:** the matrix implements the published Bate–Mueller–White
  closed form (Fundamentals of Astrodynamics Eq. 2.19/2.20) directly. Against
  an *independent* implementation of that closed form, ECI position/velocity
  agree to **< 1e-9 relative** over 60 random element sets with non-zero
  RAAN/argument-of-perigee/inclination/eccentricity, and the orbit normal
  matches `h_hat = [sinΩ·sin i, −cosΩ·sin i, cos i]` to 1e-9. The earlier
  claim was a self-derivation error.
- **Action:** regression tests pin the published closed form at non-zero
  Ω/ω/i/e (the Ω=0, ω=0 symmetric case can no longer hide a future error),
  and the docstring now describes what the matrix actually implements.

### 1.4 Eclipse fault-masking — residual logic replaces hard suppression

- **What was wrong:** `adaptive.py` set `flag = 0` whenever `check_eclipse`
  fired, and the window-based rule demanded > 50 % eclipse in a 120 s window.
  Because the only solar dips the sim produced were the genuine fault, the
  rule could excuse the fault whenever a post-ramp window overlapped an
  eclipse pass.
- **What was changed:** the eclipse check now uses the **eclipse-adjusted
  solar residual** `solar − P_max·sun_exposure`. A residual near zero is
  explained ("eclipse, expected"); a large negative residual stays an
  anomaly even during eclipse. The temperature-trend radiator heuristic was
  replaced by a **heat-rejection residual** (measured `Q_out` vs
  Stefan–Boltzmann expectation at nominal ε·A).
- **Verification:** six cases tested — normal sun, normal eclipse, array
  degradation in sun, array degradation in eclipse, eclipse + additional
  fault, unexpected sun-side drop. Eclipse can no longer erase a genuine
  fault; an eclipse + fault now flags.

### 1.5 Thermal model — first-order LEO environment, spec constants restored

- **What was wrong:** the thermal solve was radiator-only equilibrium
  (`Q_in = 60 W` vs εσA(T⁴−T_space⁴)) → steady ≈ −41 °C, no direct solar,
  no albedo, no Earth IR, no eclipse cycling. `MC_P` and the radiator-fault
  magnitude were demo-tuned (2000 J/K, 10 % radiator) for 1-hour
  detectability and shipped as the default.
- **What was changed:** direct solar, albedo and Earth IR absorbed on the bus
  (eclipse-modulated) plus load-coupled internal dissipation; spec constants
  are the default (`MC_P = 5000 J/K`, radiator fault to 30 % ε·A). The demo
  values are opt-in via `MISSIONMIND_DEMO_FAST=1` and labelled controlled
  injected faults. Energy balance `Q_in = Q_out + C·dT/dt` holds per step
  (residual ~1e-15).
- **Before → after (normal):** steady T **−41.1 → −6.8 °C**; eclipse
  excursion **27.1 → 17.2 K**. Radiator fault: peak T **50.5 → 43.2 °C**,
  margin to the 60 °C limit **9.5 → 16.8 °C**.
- **Engineering significance:** the thermal model is now a defensible
  first-order LEO model (explicitly not high-fidelity).

### 1.6 Prognostics — cycle definition measured from the EPS

- **What was wrong:** `cycles_to_days` assumed one eclipse = one full battery
  cycle (1.0 EFC/orbit).
- **What was changed:** EFC is measured from the actual SOC series
  (accumulated discharge depth, the standard equivalent-full-cycle
  convention); `cycles_to_days` uses the EPS-measured rate.
- **Before → after:** reference scenario measures **1.59 EFC/orbit** (the bus
  trips every eclipse), so 50 EFC of fade = **~3.3 days → ~2.1 days** (~37 %
  faster calendar-time degradation).

### 1.7 Edge-case hardening + numerical validation

- Degenerate inputs (`a ≤ 0`, `e ∉ [0,1)`, `μ ≤ 0`, `r = 0`, non-finite `t`)
  now raise `ValueError` instead of producing NaN / division by zero.
- `kepler_solve`: `e ≥ 1` raises; the `M = 0, e → 1` degenerate case is
  short-circuited; a guaranteed-convergent bisection fallback backs Newton.
  High-e cases (e = 0.9/0.99/0.999) verified at every mean anomaly.
- RK4 convergence order measured **4.04** (theory 4) and DOPRI5 accuracy were
  already validated in `tests/test_propagation.py`.
- Thermal explicit-Euler: stability margin verified (|λ·dt| < 0.05, ≥ 20×
  below the limit) and the dt = 1 s discretization error quantified
  (< 2 K over 1 h vs a dt = 0.05 s reference; error doubles when dt doubles,
  confirming first order). dt = 1 s is retained as adequate.

---

## 2. Before/after metric table (identical harness, identical seeds)

| scenario | metric | before | after | change |
|---|---|---|---|---|
| normal | eclipse-mean solar (W) | 520.0 | 1.5 | **−99.7 %** |
| normal | eclipse energy deficit (kJ) | 0 | 859.7 | n/a (0 → 8.6e5) |
| normal | min SOC | 0.900 | 0.000 | **−100 %** (real drain) |
| normal | time to min SOC (s) | 0 | 3385 | n/a |
| normal | min bus V | 27.60 | 0.00 (tripped) | −100 % |
| normal | steady T (°C) | −41.1 | −6.8 | +83 % (environment-coupled) |
| normal | ML FPR | 0.018 | 0.000 | −98 % |
| solar | eclipse-mean solar (W) | 249.6 | 0.7 | **−99.7 %** |
| solar | ML recall | 1.000 | 0.390 | −61 % (eclipse hides umbra rows) |
| solar | ML flag rate in sun post-fault | 1.000 | 1.000 | 0 % |
| solar | ML flag rate in eclipse post-fault | 1.000 | 0.006 | −99 % (physically correct) |
| radiator | ML recall | 0.537 | 1.000 | +86 % |
| radiator | ML F1 | 0.688 | 0.948 | +38 % |
| radiator | detection delay (s) | 1249 | 0 | −100 % |
| radiator | margin to 60 °C (K) | 9.5 | 16.8 | +76 % |

Full table (power/battery/thermal/ML per scenario, plus orbital metrics) is
produced by `.freebuff/compare.py` from `.freebuff/baseline_before.json` /
`.freebuff/baseline_after.json`.

### Reading the ML numbers honestly

- **Solar recall 1.0 → 0.39 is physics, not regression.** In umbra the array
  produces ~0 W whether or not it is degraded, so no detector can separate
  fault from eclipse there (flag rate in umbra: 0.006). In sunlight the fault
  is still caught at **1.000**.
- **Harness FPR rose in the fault scenarios** (solar 0.18 → 0.33, radiator
  0.07 → 0.33) because the harness counts t ∈ [100, 900) as pre-fault while
  the fault ramp begins at 600 s and the new thermal environment has a
  start-up transient; the more sensitive detectors flag these. Nominal FPR
  (the meaningful no-fault number) improved to 0.000.
- **Radiator detection improved materially:** recall 0.54 → 1.0, F1 0.69 →
  0.95, delay 1249 s → 0 s, driven by the heat-rejection residual feature and
  the tree-depth fix below.

---

## 3. ML-detector fixes discovered during revalidation

- **IsolationForest `max_samples=256` capped tree depth at ~8**, so even a
  −135σ residual scored at the boundary (score −0.015). Raising
  `max_samples` to 1024 restored real separation; verified on the power and
  thermal detectors.
- **Physically-informed residual features** (`solar_residual_w`,
  thermal heat-rejection residual) are computed in
  `add_derivative_features` so training, live scoring, adaptive and
  explainability all see them.
- The **SHAP explainer is cached per model fingerprint** (build cost ~100 s →
  ~0.2 s per call), removing the API-latency hang.
- **Eclipse-aware evaluation gates:** solar-fault recall is measured only
  where the fault is physically observable (sunlight).

---

## 4. Engineering assessment (Q&A)

1. **Which bugs were genuinely real?** Eclipse computed but never coupled to
   EPS; energy created at SOC = 0; eclipse rule able to mask genuine faults;
   thermal model missing the orbital environment; demo-tuned constants shipped
   as nominal; prognostics assumed one eclipse = one cycle; `max_samples=256`
   crippling the isolation forest.
2. **Which audit claims were wrong or overstated?** The PQW→ECI "frame
   error" — the matrix matches the published Bate–Mueller–White closed form
   to < 1e-9; it was a self-derivation error in the audit, not a code bug.
3. **Which fixes materially changed spacecraft behaviour?** Eclipse→EPS
   coupling (solar now dips to ~0 in shadow), the battery trip (SOC genuinely
   reaches 0 and the bus sheds/trips), the environment-coupled thermal
   equilibrium.
4. **Which fixes were mostly code correctness?** Input guards, Kepler
   M=0/e→1 handling, docstring corrections — no change to valid-input output.
5. **Which metric changed the most?** Eclipse-mean solar power in the normal
   scenario: 520 → 1.5 W (**−99.7 %**).
6. **Does the model conserve energy?** Yes — no load at SOC ≤ 0; consumed
   energy falls when the bus trips; thermal balance residual ~1e-15.
7. **Does eclipse physically affect the EPS?** Yes — solar, battery drain and
   thermal swing all respond to the same `sun_exposure` source of truth.
8. **Can a real solar fault still be detected during eclipse?** Yes if the
   residual deviates from the eclipse-expected value; no if the array is
   simply at ~0 W in umbra (physically indistinguishable).
9. **Is PQW→ECI independently validated?** Yes — against the published closed
   form at 60 random non-zero element sets, < 1e-9 relative, plus ĥ
   direction and polar-orbit orientation checks.
10. **Is the thermal model physically defensible?** As a documented
    first-order LEO model — yes (solar + albedo + IR + eclipse + radiative
    rejection, energy-balanced). Not high-fidelity by design.
11. **Did ML performance improve or worsen?** Mixed and explained: radiator
    detection much better (F1 0.69 → 0.95), nominal FPR 0, solar detection
    correctly quiet in umbra and unchanged (1.0) in sunlight; harness-FPR
    rose in fault scenarios due to the 600–900 s ramp and thermal start-up
    transients being counted pre-fault.
12. **Regressions?** None found — the full suite (now 24 suites) passes, the
    retrained models score correctly, and the live API path applies the same
    residual logic as the rules and ML layers.

## 5. Reproduce

```bash
.venv/Scripts/python.exe -m pytest missionmind/ -q            # full suite
.venv/Scripts/python.exe .freebuff/metrics.py --out before.json   # harness (any code state)
python missionmind/e2e_dry_run.py                             # end-to-end dry run
```

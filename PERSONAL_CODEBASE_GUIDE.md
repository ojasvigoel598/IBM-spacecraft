# MissionMind · Personal Codebase Guide

> **This file is for you, not the judges.** It teaches the project to someone with
> reasonable aerospace background who is not yet comfortable with all the Python
> code. It is honest about what the system actually does and where it is
> theatre-versus-real.
>
> If you have not read this for a year and open it tomorrow, you should be able
> to understand (and re-build) the entire project from this file.

---

## Table of Contents

1. Big picture — the 5-sentence overview
2. The data pipeline (one diagram, every stage)
3. Project map (every file in the repo and why it exists)
4. File-by-file walkthrough
5. Line-by-line explanations of important code
6. Trace one telemetry sample through the system
7. Machine learning — explained from zero
8. Physics — every equation, every variable
9. Physics + ML — what each layer knows and how they fuse
10. Walkthrough of a real anomaly from t=0 to alert
11. The 3D digital twin — what is genuinely physics, what is theatre
12. The AI / Granite / RAG layer
13. The two frontends: Streamlit dashboard + React web console
14. The FastAPI backend (real HTTP API with auth)
15. The auth system
16. The adaptive decision layer
17. Configuration, environment variables, paths, ports
18. How to run everything (exact commands)
19. Troubleshooting — known failure modes and fixes
20. Final verified model results (what the numbers mean)
21. Design decisions — alternatives considered, why we chose what we did
22. "What to say if someone asks" — cheat sheet for live demos
23. MissionMind in 5 minutes

---

## 1 · Big picture

**Problem:** A small satellite is in orbit. Telemetry streams in — voltage,
solar current, temperature, etc. — and we want to know, *as fast as possible,*
whether something on the spacecraft is broken or degrading. We don't want false
alarms (they cost operator attention and mission time), and we don't want to
miss real failures (they cost the satellite). We want a system that fuses
**physics sanity-checks** with **machine-learning anomaly detection** and
produces an **operator-facing explanation** that names the likely subsystem
and recommends an action.

**MissionMind does exactly this:**

| Stage | What it does | File(s) |
|---|---|---|
| 1. Generate or read telemetry | Runs a deterministic physics simulator (Kepler orbital mechanics + power + thermal) | `simulator/run_scenarios.py`, `simulator/orbital.py` |
| 2. Compute physics expectations | For each row, predict what temperature and SOC *should* be | `simulator/power.py`, `simulator/thermal.py` |
| 3. Run physics sanity-checks | Flag rows where observed vs predicted diverge beyond thresholds | `physics_rules/rules.py` |
| 4. Run ML anomaly detection | Anomaly score + binary anomaly flag from a 3-model IsolationForest ensemble | `ml/detect.py`, `ml/train.py` |
| 5. Adaptive decision layer | Situation-aware fusion of physics + ML with strategy selection | `ml/adaptive.py` |
| 6. Retrieve relevant docs (RAG) | TF-IDF search across markdown knowledge base | `ai/rag.py` |
| 7. Generate Granite explanation | IBM watsonx LLM (or deterministic mock) produces a JSON recommendation | `ai/granite_client.py`, `ai/prompts.py` |
| 8. Causal narrative | 4-line operator alert: WARN → SUBSYSTEM → EVIDENCE → ACTION | `ml/causal_narrative.py` |
| 9. Render dashboard | Streamlit with live 3D IBM satellite, telemetry charts, alerts, RAG evidence, Granite panel | `viz/app.py` |
| 10. HTTP API + Web console | FastAPI backend with auth, React/Vite frontend | `viz/api_server.py`, `web/` |
| 11. Verify | All of the above have tests in `tests/` that run via `missionmind.e2e_dry_run` |

**One-line pipeline:**

```
telemetry → orbital propagation → physics check → ML ensemble → adaptive decision → RAG evidence → Granite JSON → dashboard/API
```

---

## 2 · The data pipeline (one diagram, every stage)

```
raw telemetry row:                  time_s=600, solar_w=520, V=28.0, T=-42.0, ...
        │
        ▼
orbit_columns(t)                    → orbit_angle_deg, in_eclipse, sun_exposure
        │                            (real Kepler: Newton-Raphson solver)
        ▼
add_derivative_features()          → same row + dT/dt, dV/dt
        │
        ▼
build_feature_matrix()             → numpy matrix shape (n, 5):
                                        col0 V, col1 solar, col2 T, col3 dT/dt, col4 dV/dt
        │
        ▼
StandardScaler (fit on train only) → X_scaled, zero-mean unit-variance per column
        │
        ▼
3 parallel IsolationForests:
   ├─ full   (all 5 features)
   ├─ power  (V, solar, dV/dt)       remember: detect.py uses MIN of the 3 scores
   └─ thermal (T, dT/dt)              for coherence (anomaly_score = lowest)
        │
        ▼
OR-of-flags + MIN-of-scores         → anomaly_flag ∈ {0,1}, anomaly_score ∈ ℝ
        │
        ▼
physics rules (rules.py)            → physics_flag ∈ {None, 'solar_degradation',
                                                       'radiator_degradation'}
        │
        ▼
adaptive decision (adaptive.py)     → situation-aware fusion: strategy selection,
        │                            weighted score, reasoning lines
        ▼
TF-IDF RAG (rag.py)                 → top-3 docs from knowledge_base/*.md
        │
        ▼
Granite (granite_client.py)         → JSON with risk, probable_cause,
                                       reasoning, recommended_action,
                                       evidence_used, confidence
        │
        ▼
causal_narrative.py                 → 4-line operator alert:
        │                            WARN - SUBSYSTEM - EVIDENCE - ACTION
        ▼
Streamlit render (viz/app.py)       → KPI cards, charts, 3D model,
        │                            Granite panel, status strip
        ▼
FastAPI (api_server.py)             → JSON API on port 8100 with auth,
                                       live ingest, RAG alerts, model zoo
        │
        ▼
React console (web/)                → Authenticated mission-control UI
                                        on port 5173
```

---

## 3 · Project map (ACTUAL on-disk structure)

```text
missionmind/
├── __init__.py
├── e2e_dry_run.py              # Runs the full pipeline twice and asserts every stage PASSes
├── open_threejs.py             # Standalone Three.js viewer
├── check_environment.py        # Verifies all deps are importable
├── trace.py                    # Runtime execution trace (which code ran)
│
├── simulator/                  # Physics simulation + real Kepler orbital mechanics
│   ├── config.py               # Single source of truth for every physics constant (DEMO_FAST toggle)
│   ├── orbital.py              # REAL Kepler: Newton-Raphson solver, orbital period, true anomaly,
│   │                           #   eclipse geometry (conical shadow), sun-pointing attitude,
│   │                           #   orbital energy/angular momentum conservation
│   ├── propagation.py          # Numerical extension: RK4 + adaptive DOPRI5 + J2 perturbation
│   ├── power.py                # Battery + solar module (Euler-integrated, eclipse-coupled)
│   ├── thermal.py              # Stefan-Boltzmann radiative balance
│   ├── failures.py             # Fault injection: solar_degradation, radiator_degradation
│   ├── run_scenarios.py        # Coupled simulator → 3 CSVs (calls orbit_columns per row)
│   └── physics_verification.py # Hand-calculated cross-check
│
├── physics_rules/              # Operator-readable anomaly rules
│   ├── rules.py                # check_power_subsystem, check_thermal_subsystem, eclipse-aware
│   └── test_rules.py           # Asserts they fire on the 3 scenarios
│
├── ml/                         # Machine learning
│   ├── train.py                # Trains 3 IsolationForests + scalers, saves to models/
│   ├── detect.py               # Loads models and scores any DataFrame
│   ├── adaptive.py             # Situation-aware decision layer (strategy selection + weighted fusion)
│   ├── causal_narrative.py     # 4-line operator alert generation
│   ├── explainability.py       # SHAP TreeExplainer on production IF model
│   ├── rul_uncertainty.py      # RUL with bootstrap confidence intervals
│   ├── metrics.py              # Basic + advanced metric protocols
│   ├── compare.py              # Comparison matrix across all models
│   ├── cross_fault.py          # Layer ablation (physics / individual / ensemble / adaptive)
│   ├── advanced_models.py      # FCNN, XGBOD, MLP-AE, Hybrid-DIF, PINN zoo
│   ├── pinn_*.py               # PINN research (Raissi strict, torch-autograd, head-to-head)
│   ├── drift.py                # Streaming KS drift test
│   ├── prognostics.py          # Battery RUL (Trend / Similarity / PhysicsInformed)
│   ├── nasa_real_validation.py # Real NASA PCoE .mat validation
│   └── models/                 # Trained artefacts (joblib .pkl files)
│
├── ai/                         # LLM + RAG
│   ├── granite_client.py       # IBM watsonx wrapper + deterministic mock
│   ├── rag.py                  # TF-IDF retriever over markdown
│   ├── prompts.py              # System + user prompt templates
│   └── knowledge_base/         # 4 markdown docs (power, thermal, mission rules, telemetry ref)
│
├── auth/                       # Real multi-user authentication
│   ├── service.py              # Domain logic: create_user, verify_email, sessions, password reset
│   ├── api.py                  # FastAPI router: signup, verify, login, logout, me, reset
│   ├── deps.py                 # FastAPI dependency guards (session cookie → user dict)
│   ├── security.py             # PBKDF2 hashing, token generation, digest storage
│   ├── db.py                   # SQLite schema + thread-local connections
│   ├── ratelimit.py            # In-memory sliding-window rate limiter
│   └── notify.py               # Email notification (dev-echo mode without SMTP)
│
├── telemetry/                  # Live telemetry ingest
│   ├── edge_node.py            # Virtual ESP32-class device (uses orbit_columns for illumination)
│   ├── ingest.py               # JSON-lines TCP server/client + MQTT (paho) + LiveScorer
│   ├── frame.py                # Wire schema (identical to run_scenarios output)
│   └── run_edge_demo.py        # CLI demo: stream 1000 frames through the ensemble
│
├── viz/                        # Streamlit dashboard + Three.js + FastAPI backend
│   ├── app.py                  # ~1400 lines · the entire Streamlit dashboard
│   ├── api_server.py           # FastAPI JSON API: same pipeline + auth + live ingest + model zoo
│   └── components/
│       ├── satellite_geometry.js (generated from OBJ, NOT edited by hand)
│       ├── obj_to_geometry.py  # CATIA/Autodesk OBJ → satellite_geometry.js
│       └── models/             # IBM satellite CAD (OBJ, STL, STEP)
│
├── data/                       # Generated CSVs + real NASA data
│   ├── run_normal.csv          # 3600 rows · deterministic
│   ├── run_solar_failure.csv   # 3600 rows · solar ramp 600→900
│   ├── run_radiator_failure.csv# 3600 rows · ε·A ramp 600→900
│   └── real_nasa/              # B0005/B0006/B0007/B0018 .mat files
│
├── models/                     # Trained artefacts (joblib .pkl files)
│   ├── iforest.joblib, iforest_power.joblib, iforest_thermal.joblib
│   ├── scaler*.joblib, features.txt
│   └── *.json                  # audit matrices, consistency checks
│
└── tests/                      # 30 TDD suites, all pass
    ├── test_physics.py
    ├── test_ml_metrics.py
    ├── test_granite_nominal.py
    ├── test_config_seam.py
    ├── test_prognostics.py
    ├── test_drift.py
    ├── test_auth.py            # 33 auth tests (signup, login, brute-force, injection)
    ├── test_api_server.py      # 12 API surface tests
    ├── test_adaptive.py        # Adaptive decision layer tests
    ├── test_orbital.py         # Kepler solver, orbital period, energy conservation
    ├── test_propagation.py     # RK4/DOPRI5 convergence to analytical Kepler
    └── ...

web/                            # React 19 + Vite + Tailwind v4 + shadcn/ui
    ├── src/App.tsx             # Mission console: KPI grid, SVG charts, time scrubber
    ├── src/auth.tsx            # Auth context provider
    ├── src/components/auth/    # Login/signup/verify screens
    └── src/components/ui/     # shadcn/ui components

scripts/
    ├── build_demo_v2.py        # New demo video builder (edge-tts + MoviePy + FFmpeg)
    ├── brighten_frames.py      # Post-process frames for video visibility
    ├── capture_bright_frames.py# Playwright capture from live dashboard
    ├── start_web_preview.ps1   # Detached FastAPI + Vite launcher
    └── start_dashboard.ps1     # Detached Streamlit launcher

api/
    └── index.py                # Vercel serverless entry point (re-exports FastAPI app)
```

**Dependencies between layers (load order):**

```
simulator/* (no dep, includes orbital.py with Kepler)
    ↓
physics_rules/* (depends on simulator + orbital eclipse state)
    ↓
ml/train.py  →  simulator, then writes models/*.joblib
ml/detect.py →  reads models/*.joblib
ml/adaptive.py →  ml/detect + physics_rules (situation-aware fusion)
ml/causal_narrative.py →  ml/detect + physics_rules + RAG chunks
ml/explainability.py →  ml/detect (SHAP on IF)
    ↓
ai/rag.py, ai/granite_client.py, ai/prompts.py (depend on data layout only)
    ↓
auth/* (standalone auth domain)
    ↓
viz/api_server.py  →  imports ALL of the above + auth + telemetry
viz/app.py  →  imports physics_rules, ml.detect, ai.*, simulator.run_scenarios
web/  →  calls /api/* endpoints on the FastAPI backend
tests/*.py  →  import each layer and assert
e2e_dry_run.py  →  runs everything twice, in this order
```

---

## 4 · File-by-file walkthrough

### `missionmind/simulator/orbital.py` (THE ORBIT IS REAL)

**What it does:** Full Kepler orbital mechanics for LEO at 550 km. This is NOT decorative — it propagates the satellite with real two-body equations:

```python
# Kepler's third law
T = 2π · √(a³ / μ)                    # orbital period (~95.5 min)

# Kepler's equation (Newton-Raphson solver)
E - e·sin(E) = M                       # eccentric anomaly

# True anomaly from eccentric anomaly
ν = 2·arctan2(√(1+e)·sin(E/2), √(1-e)·cos(E/2))

# 3D ECI state vectors
r_ECI = R_PQW_to_ECI · r_PQW          # rotation from perifocal to ECI
```

**Downstream use (physics genuinely matters):**
1. `run_scenarios.py` calls `orbit_columns(t)` per row → `orbit_angle_deg`, `in_eclipse`, `sun_exposure`
2. Eclipse state drives `power.py` (solar drops to ~0 during eclipse)
3. Physics rules use `in_eclipse` to suppress false alarms during orbital night
4. The 3D dashboard reads `orbit_angle_deg` to position the satellite on its real orbit
5. `adaptive.py` uses eclipse fraction to decide if a solar dip is expected geometry or genuine fault

**Kepler solver quality:** Verified to machine precision (`tol=1e-12`, residual < 1e-9). Handles e=0 (circular, short-circuits), e→1 (Vallado starting guess + bisection fallback). Energy conservation checked along the full trajectory.

### `missionmind/simulator/config.py`

**What it does:** Defines EVERY physics constant as a Python constant, then re-exports them. Either spec-faithful values (mc_p=5000 J/K) or demo-fast values (mc_p=2000 J/K). The flag `DEMO_FAST` chooses between them globally.

**Why it exists:** Before this file, `power.py` and `thermal.py` had hardcoded local copies. Now config.py is the single source of truth — a missing config raises a loud ImportError.

### `missionmind/simulator/power.py`

```python
solar_w   = P_SOLAR_MAX * illumination(t) * degradation_factor
net_w     = solar_w - P_LOAD
d_soc     = (net_w * DT_S / 3600.0) / E_CAP_WH
soc_new   = clip(soc + d_soc, 0, 1)
voltage_v = V_MIN + (V_MAX - V_MIN) * soc_new
```

Eclipse-coupled: `illumination(t)` comes from `orbital.py`'s conical shadow model (umbra→0W, penumbra→smooth transition, full sun→520W).

### `missionmind/simulator/thermal.py`

```python
Q_out = EPSILON * SIGMA * AREA * (T_k**4 - T_SPACE_K**4)   # Stefan-Boltzmann
dT    = (Q_IN - Q_OUT) * DT_S / MC_P                        # 1st-order heat balance
T_new = T_k + dT                                            # Euler step
```

### `missionmind/simulator/propagation.py`

**Numerical extension point.** Adds RK4 (fixed-step) and adaptive DOPRI5 propagators on top of the analytical Kepler baseline. Validated: RK4 converges to analytical Kepler at order ~4; DOPRI5 meets tolerance with fewer steps. Also implements J2 perturbation (oblateness) for perturbed orbits.

### `missionmind/ml/adaptive.py` (NEW)

**What it does:** Situation-aware decision layer that sits between the raw ensemble and the operator display. Instead of always using the same fusion logic, it selects a strategy based on what's actually happening:

- **NOMINAL**: no physics or ML flags → report nominal
- **RULE_FIRST**: physics rule fires clearly → trust physics
- **RAMP_LEAD**: ML detects the ramp early, physics hasn't caught up yet → ML leads
- **CONSENSUS**: both physics and ML agree → high confidence
- **ECLIPSE_DISAGREEMENT**: ML flags a solar dip that physics says is just orbital eclipse → suppress, report as expected

Outputs: `strategy`, `adaptive_score` (weighted fusion), `adaptive_flag`, `weights`, `reasoning[]`.

### `missionmind/ml/causal_narrative.py` (NEW)

**What it does:** Generates the 4-line operator alert visible on every anomaly:
```
WARN — SOLAR ARRAY DEGRADATION
Detected T+10:07 · Source POWER
Solar mean 270 W vs 364 W threshold
Battery SOC slope -0.0004/s → Reduce load, enter safe mode
```

### `missionmind/ml/explainability.py` (NEW)

**What it does:** SHAP TreeExplainer on the production IsolationForest. Returns per-feature attribution: which features drove the anomaly score. Cached explainer for performance (TreeExplainer on IF is exact and fast).

### `missionmind/ml/rul_uncertainty.py` (NEW)

**What it does:** Remaining Useful Life with bootstrap confidence intervals. Battery RUL from SOC drain rate, thermal RUL from dT/dt. Returns point estimate + 90% CI bounds. Format: "BAT 83 min (72–94)".

### `missionmind/ml/train.py`

Trains 3 IsolationForests (full + power-specialist + thermal-specialist) on `run_normal.csv`. contamination=0.05 chosen by operator FP tolerance (NOT test-set tuned). Saves 6 `.joblib` files.

### `missionmind/ml/detect.py`

Loads trained models. Produces `anomaly_flag`, `anomaly_score`, `anomaly_source` with **guaranteed coherence**: `flag=1 ⟹ score<0` (MIN of three detectors ensures this by construction).

### `missionmind/physics_rules/rules.py`

Two pure-Python sanity checks, now **eclipse-aware**:
- `check_power_subsystem`: solar < 0.7·P_max AND SOC slope negative
- `check_thermal_subsystem`: dT/dt > 0.003 AND dQ_in/dt < 1
- Eclipse suppression: when `in_eclipse=1`, a solar dip is EXPECTED and the rule returns an eclipse finding instead of a fault

### `missionmind/ai/rag.py`

TF-IDF over `knowledge_base/*.md` (4 files, ~31 chunks). Top-k retrieval by cosine similarity. Metadata-scoped to subsystem (power/thermal/mission). Returns source citations.

### `missionmind/ai/granite_client.py`

IBM watsonx Granite wrapper. If `WATSONX_APIKEY` is set, calls real API. Otherwise deterministic mock (tagged `source="mock"`). `strict=True` mode raises on failure (for `--check` verification).

### `missionmind/auth/` (NEW)

Real multi-user authentication:
- **signup** → email verification → login → session cookie
- PBKDF2-HMAC-SHA256 (310k iterations, per-user salt)
- HttpOnly SameSite=Lax cookies (Secure in production)
- Single-use expiring verification/reset tokens
- Enumeration-safe (same response for existing/unknown emails)
- Rate limiting (login brute-force, per-IP, per-user)
- Server-side authorization (role from DB, never from request)

### `missionmind/telemetry/` (NEW)

Live telemetry ingest:
- `edge_node.py`: Virtual ESP32-class device (12-bit ADC, 2% packet dropout)
- `ingest.py`: JSON-lines TCP server/client + MQTT (paho) + LiveScorer
- Uses `orbit_columns()` for eclipse illumination — the same orbital propagator drives the live stream

### `missionmind/viz/api_server.py` (NEW)

FastAPI JSON API on port 8100. Same pipeline as the Streamlit dashboard but as HTTP endpoints:
- `GET /api/health` — runtime health + Granite state machine
- `GET /api/scenario/{mode}` — full scored telemetry
- `GET /api/alert/{mode}?t=` — physics + ML + RAG alert evidence
- `GET /api/summary/{mode}?t=` — condensed snapshot with RUL + adaptive decision
- `GET /api/live/next?mode=&n=30` — advance virtual edge-node stream
- `GET /api/models` — model zoo self-test
- `GET /api/trace` — runtime execution trace
- `POST /api/auth/*` — signup, verify, login, logout, reset

Security: auth required for all mission endpoints, CORS restricted, 16KB body cap, rate limiting, security headers.

### `missionmind/viz/app.py`

~1400-line Streamlit dashboard. Key sections:
- Sidebar: scenario selector, playback controls, RAG toggle
- KPI strip: Solar, SOC, Voltage, Temperature, Heat I/O, ε·A
- Tabs: Telemetry + 3D · ML Diagnostics · RAG Evidence · Granite · Scenarios · Live Ingest
- Three.js: real IBM satellite CAD, driven by `orbit_angle_deg` from Kepler propagator

### `web/` (NEW)

React 19 + Vite + Tailwind v4 + shadcn/ui mission-control console:
- Login/signup/verify flow
- KPI grid, SVG telemetry charts, time scrubber
- Alert evidence with physics + RAG citations
- Live ingest tab, model diagnostics, runtime code trace
- All endpoints authenticated via session cookie

---

## 5 · Line-by-line explanations of important code

### `orbit_columns()` in `simulator/orbital.py`

```python
def orbit_columns(t: float) -> dict:
    """One frame's worth of orbital telemetry (deterministic)."""
    a = REF_SEMI_MAJOR  # R_EARTH + 550e3 = 6921 km
    r_vec, _, nu, M = state_vectors_3d(t, a=a)
    period = orbital_period_s(a)
    geo = eclipse_geometry(r_vec)
    att = sun_pointing_attitude(t)
    cons = orbital_energy_and_angular_momentum(a)
    return {
        "orbit_angle_deg": round(float(np.degrees(nu)), 3),  # TRUE ANOMALY from Kepler
        "in_eclipse": int(geo["eclipse_state"] != "full"),    # conical shadow model
        "sun_exposure": round(geo["sun_exposure"], 4),        # 0=umbra, 1=full sun
        ...
    }
```

This is called per row in `run_scenarios.py` — every 1-second tick gets the real Kepler angle, eclipse state, and sun exposure. The angle drives both the physics (eclipse → solar drops) and the 3D visualisation (satellite position on orbit ring).

### `add_derivative_features` (`train.py`)

```python
df["d_temp_dt"] = df["temperature_c"].diff().fillna(0) / max(dt_s, 1e-9)
df["d_volt_dt"] = df["battery_voltage_v"].diff().fillna(0) / max(dt_s, 1e-9)
```

Row-by-row difference (ΔT). For 1-second timestep this is dT/dt. `fillna(0)` prevents undefined first-sample value flowing into the scaler.

### `_ensemble_score_and_flag` (`detect.py`)

```python
ensemble_flag  = pred_full | pred_power | pred_thermal        # OR
ensemble_score = np.minimum.reduce([scores_full, scores_power, scores_thermal]) # MIN
```

IsolationForest `decision_function`: higher = more normal. MIN = most pessimistic detector. This guarantees `flag=1 ⟹ score<0`.

### `decide()` in `adaptive.py`

```python
def decide(window):
    # 1. Check physics rules
    # 2. Check ML flags
    # 3. Check eclipse agreement
    # 4. Select strategy (NOMINAL / RULE_FIRST / RAMP_LEAD / CONSENSUS / ECLIPSE_DISAGREEMENT)
    # 5. Weighted fusion of detector scores
    # 6. Generate reasoning lines
```

The adaptive layer is what makes MissionMind more than just "IF + physics rules" — it situationally weights the layers based on what's actually happening.

---

## 6 · Trace one telemetry sample through the system

Pick `time_s=900` in `run_solar_failure.csv`:

| Stage | What's happening | What the data looks like |
|---|---|---|
| `orbit_columns(900)` | Kepler propagator computes true anomaly + eclipse | `orbit_angle_deg=187.3, in_eclipse=0, sun_exposure=1.0` |
| Raw row | 1-second tick from physics simulator | `solar_w=249.6, V=27.1, T=-40.5, heat_in=60, heat_out=55` |
| `add_derivative_features` | Append two columns | `d_temp_dt=-0.0001, d_volt_dt=-0.0073` |
| `StandardScaler` | Subtract train mean, divide std | All columns zero-mean, unit-variance |
| Three IsolationForests | Binary decision + score | `pred_full=True, score=-0.42`; `pred_power=True, score=-0.31`; `pred_thermal=False, score=0.08` |
| `np.minimum.reduce` | Most pessimistic score | ensemble score = **-0.42** |
| Physics rule | Solar mean 270W < 364W threshold | `physics_flag='solar_degradation', conf=0.85` |
| Adaptive decide | Both ML and physics agree → CONSENSUS | `strategy='CONSENSUS', adaptive_flag=1, adaptive_score=-0.38` |
| RAG | Top-3 docs from TF-IDF | Solar degradation procedure, power mission rule |
| Granite | Receives JSON payload → produces recommendation | `risk=HIGH, probable_cause='Solar array degradation', recommended_action='Reduce load, safe mode'` |
| Causal narrative | 4-line operator alert | `WARN — SOLAR ARRAY DEGRADATION · Detected T+15:00 · Source POWER · Solar 270W vs 364W threshold` |
| Dashboard | KPI cards + anomaly chip + 3D panels dim | Operator sees full diagnosis with citations |

---

## 7 · Machine learning — explained from zero

**Isolation Forest** is an unsupervised anomaly detector. Random splits isolate outliers faster than normal points. Path length = anomaly score.

**Three-model ensemble:**
- **full** (5 features): catches correlated multi-subsystem faults
- **power** (V, solar, dV/dt): catches solar-only faults that the full model might dilute
- **thermal** (T, dT/dt): catches radiator faults that the full model might miss

**OR flag + MIN score** = guarantee that flag=1 ⟹ score<0 (no dashboard contradictions).

**Adaptive layer** selects the strategy: when physics and ML disagree (eclipse scenario), it reports the disagreement instead of suppressing it.

**External validation:** AUC = 0.786 ± 0.009 on real NASA PCoE B0005 data (6-seed robust).

---

## 8 · Physics — every equation, every variable

### Orbital mechanics (`simulator/orbital.py`)

```python
T = 2π · √(a³ / μ)                    # Kepler's 3rd law (~95.5 min at 550 km)
E - e·sin(E) = M                       # Kepler's equation (Newton-Raphson)
ν = 2·arctan2(√(1+e)·sin(E/2), √(1-e)·cos(E/2))  # true anomaly
r = a(1-e²) / (1 + e·cos(ν))          # orbital radius
r_ECI = R_PQW_to_ECI · r_PQW          # ECI coordinates
```

Constants: μ=3.986e14 m³/s², R_EARTH=6371 km, altitude=550 km, e=0 (circular), i=51.6°.

### Power subsystem

```
P_solar(t) = P_SOLAR_MAX · sun_exposure(t) · degradation_factor(t)
net_w = P_solar - P_LOAD
ΔSOC = (net_w · Δt) / E_CAP_WH / 3600
V(t) = V_MIN + (V_MAX - V_MIN) · SOC(t)
```

### Thermal subsystem

```
Q_out = ε · σ · A · (T_k⁴ - T_space⁴)      # Stefan-Boltzmann
dT/dt = (Q_in - Q_out) / (m · c_p)          # Euler step
```

---

## 9 · Physics + ML — what each layer knows

| Knowledge | Physics | ML | Adaptive |
|---|---|---|---|
| Expected temperature | ✅ Stefan-Boltzmann | ❌ | selects strategy |
| Solar < 70% = degradation | ✅ threshold | ✅ cluster shape | ✅ weights |
| Eclipse vs fault | ✅ Kepler geometry | ❌ (flags both) | ✅ DISAGREEMENT |
| Cross-subsystem correlation | ❌ | ✅ IF ensemble | ✅ fuses |
| Operator-actionable diagnosis | ❌ | ❌ | ❌ (Granite) |
| Citation to procedure | ❌ | ❌ | ❌ (RAG) |

---

## 10 · Walkthrough of a real anomaly

What an operator sees when solar array degradation begins at t=600 s:

1. **t=0..599 (Normal):** Solar=520W, SOC ramps 0.9→1.0, T=-42°C stable, `anomaly_flag=0`
2. **t=600 (fault injects):** degradation_factor starts ramping 1.0→0.48
3. **t=607 (ML detects):** Within 7 seconds, IF ensemble flags anomaly. Physics hasn't caught up yet → adaptive selects `RAMP_LEAD` strategy
4. **t=900 (ramp ends):** Solar=250W. Physics rule fires → `CONSENSUS` strategy. RAG retrieves docs. Granite produces diagnosis. 3D solar panels dim red.
5. **t=2961 (bus shutdown):** Battery SOC=0, bus trips. But the operator had 39 minutes of warning.

---

## 11 · The 3D digital twin

**What is genuinely physics:**
- `orbit_angle_deg` from real Kepler propagator drives satellite position on orbit ring
- Eclipse state dims the sun light during orbital night
- Solar panel emissive intensity tracks `solar_power_w` (dims on fault)
- Beacon pulses red on `anomaly_flag=1`
- HUD shows real-time telemetry from the Kepler-coupled simulator

**What is visual scale:**
- Orbit ring radius (3.6 scene units) is a visual simplification of the real 6921 km orbit
- Earth texture and starfield are decorative
- Satellite rotation follows velocity vector (nose-along-v convention)

**NOT a closed-loop twin:** no command execution path from UI back to simulator.

---

## 12 · The AI / Granite / RAG layer

**Granite is the EXPLAINER, not the detector.** The actual anomaly decision comes from physics_rules + ml/detect + adaptive. Granite produces the human-readable prose with citations.

**RAG grounding:** TF-IDF over 4 markdown docs (~31 chunks). Metadata-scoped to subsystem. Adversarial-tested (prompt injection, wrong documents, conflicting sources).

**Mock fallback:** When no IBM credentials, deterministic mock returns schema-valid JSON tagged `source="mock"`. Dashboard renders identically.

---

## 13 · The two frontends

### Streamlit dashboard (`viz/app.py`, port 8501)
- Richer: 3D Three.js satellite, Plotly charts, sidebar controls
- Single-file: ~1400 lines
- Session-persistent: resumes at last-viewed scenario/time
- No auth required (local dev)

### React web console (`web/`, port 5173)
- Authenticated: signup → verify → login
- Cleaner UI: shadcn/ui components, SVG charts
- Calls `/api/*` on the FastAPI backend
- Live ingest tab, model diagnostics, runtime trace
- Deployable to Vercel

---

## 14 · The FastAPI backend

Port 8100. Same pipeline as Streamlit but as HTTP JSON endpoints.

```
GET  /api/health                    → runtime health + Granite state
GET  /api/scenarios                 → available scenarios
GET  /api/scenario/{mode}           → full scored telemetry
GET  /api/summary/{mode}?t=         → snapshot with RUL + adaptive decision
GET  /api/alert/{mode}?t=           → physics + ML + RAG + narrative
GET  /api/live/next?mode=&n=30      → advance virtual edge node
GET  /api/models                    → model zoo self-test
GET  /api/trace                     → runtime execution trace
POST /api/auth/signup|login|logout  → authentication
```

All mission endpoints require a verified session cookie. Admin endpoints require admin role.

---

## 15 · The auth system

- **PBKDF2-HMAC-SHA256** (310k iterations, per-user salt)
- **HttpOnly SameSite=Lax cookies** (Secure in production)
- **Single-use expiring tokens** for verification and password reset
- **Enumeration-safe** (same generic response for existing/unknown emails)
- **Rate limited** (login brute-force: 5 attempts/min per IP; API: 240/min per user)
- **Server-side authorization** (role from DB, never from request)
- **33 tests** covering authn/authz, token replay, injection, brute-force, CORS

---

## 16 · The adaptive decision layer

Selects a strategy per telemetry window:

| Strategy | When | Behavior |
|---|---|---|
| NOMINAL | No flags anywhere | Report nominal |
| RULE_FIRST | Physics rule fires, ML may or may not | Trust physics threshold |
| RAMP_LEAD | ML flags early, physics hasn't caught up | ML leads detection |
| CONSENSUS | Both physics and ML agree | High confidence |
| ECLIPSE_DISAGREEMENT | ML flags solar dip that physics says is eclipse | Suppress, report as expected |

The adaptive layer ensures that eclipse geometry (real Kepler) suppresses false alarms that a pure ML system would raise.

---

## 17 · Configuration

**Key environment variables:**

| Variable | Effect |
|---|---|
| `MISSIONMIND_DEMO_FAST=1` | Use demo-tuned constants (mc_p=2000) |
| `MISSIONMIND_PHYSICS_SPEC=1` | Use spec-faithful constants (mc_p=5000) |
| `WATSONX_APIKEY` | Enable real Granite (otherwise mock) |
| `WATSONX_PROJECT_ID` | IBM watsonx project |
| `MISSIONMIND_ADMIN_EMAIL` | Bootstrap admin account |
| `MISSIONMIND_ADMIN_PASSWORD` | Bootstrap admin password |
| `MISSIONMIND_ENV=production` | Secure cookies, no token exposure |
| `MISSIONMIND_ALLOWED_ORIGINS` | CORS allowlist |

**Ports:**
- 8501: Streamlit dashboard
- 8100: FastAPI backend
- 5173: Vite React console

---

## 18 · How to run everything

```bash
# First-time setup
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python -m missionmind.simulator.run_scenarios   # 3 CSVs
python -m missionmind.ml.train                  # trains ensemble

# Dashboard
streamlit run missionmind/viz/app.py            # port 8501

# FastAPI + Web console
python -m uvicorn missionmind.viz.api_server:app --port 8100
cd web && npm install && npm run dev -- --port 5173

# Tests
python -m pytest missionmind/tests/ -q          # 30 suites

# E2E verification
python -m missionmind.e2e_dry_run               # runs twice, exits 0 only if both PASS

# Build demo video
python scripts/capture_bright_frames.py         # capture from live dashboard
python scripts/brighten_frames.py               # brighten for video
python scripts/build_demo_v2.py                 # assemble final MP4
```

---

## 19 · Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Use `.venv/Scripts/python.exe -m ...` explicitly |
| Dashboard 3D is blank | Run `python missionmind/viz/components/obj_to_geometry.py ...` |
| Granite says "model unavailable" | Set `WATSONX_APIKEY` env var (mock runs without it) |
| `e2e_dry_run` fails | Run each step manually to find the failing one |
| UnicodeEncodeError on Windows | `set PYTHONIOENCODING=utf-8` |

---

## 20 · Final verified model results

| Test | Result |
|---|---|
| `e2e_dry_run` | PASS × 2 iterations |
| `test_physics` | SOC 1.000, V 28V, T_eq 223K |
| `physics_rules.test_rules` | 44/44 post-injection, 0 pre-injection FP |
| `test_orbital` | Kepler residual < 1e-9, energy conserved, eclipse geometry correct |
| `test_propagation` | RK4 converges to analytical Kepler at order ~4 |
| `test_auth` | 33/33 pass (signup, login, brute-force, injection, CORS) |
| `test_api_server` | 12/12 pass (health, scenarios, alerts, live, trace) |
| `test_adaptive` | 6/6 pass (NOMINAL, SOLAR, RADIATOR, strategy selection) |
| NASA PCoE B0005 | AUC = 0.786 ± 0.009 (6 seeds) |

---

## 21 · Design decisions

| Decision | Why |
|---|---|
| Three-model IF ensemble | Subsystem-specialised IFs catch faults the full model dilutes |
| Real Kepler propagator | Eclipse state must be physically accurate to suppress false alarms |
| Adaptive decision layer | Situation-aware fusion beats static weighting |
| TF-IDF RAG | 31 chunks from 4 docs — embeddings add no value at this scale |
| Mock-first Granite | Dashboard never breaks when IBM credentials are absent |
| MIN-of-scores ensemble | Guarantees flag=1 ⟹ score<0 (no contradictions) |
| contamination=0.05 | Operator's FP tolerance, not test-set tuned |
| FastAPI + Streamlit | Streamlit for rich 3D demo; FastAPI for programmatic access + auth |

---

## 22 · "What to say if someone asks"

**"What does your AI do?"**
> "MissionMind detects spacecraft faults in 7 seconds using physics + ML, explains the cause with RAG-grounded evidence, and recommends corrective actions via Granite. The AI is the explainer; the detection is physics + ensemble."

**"Where is the real physics?"**
> "The orbital propagator solves Kepler's equation via Newton-Raphson. The eclipse geometry uses a conical shadow model. Power integrates SOC with eclipse-coupled solar input. Thermal uses Stefan-Boltzmann. Every equation is in `simulator/`."

**"What is the digital twin?"**
> "The 3D model is the actual IBM satellite CAD. Solar panels dim on fault. The satellite orbits using real Kepler angles. Eclipse dims the sun. It's not a closed-loop twin — no command execution path yet."

**"How do you know it works?"**
> "30 test suites pass. E2E dry run passes twice. AUC = 0.786 on real NASA battery data. Eclipse-aware rules suppress false alarms during orbital night."

---

## 23 · MissionMind in 5 minutes

1. **Problem:** Small satellite faults need fast diagnosis with few false alarms.
2. **Architecture:** Kepler orbital mechanics → physics rules → 3-model IF ensemble → adaptive decision → RAG evidence → Granite explanation → 4-line operator alert.
3. **Data:** Deterministic simulator (3 scenarios × 3600 rows) + real NASA PCoE battery cells.
4. **Physics:** Stefan-Boltzmann thermal, Euler-integrated SOC, Kepler orbital propagation with eclipse geometry.
5. **ML:** Three IsolationForests (full/power/thermal), OR flag + MIN score, adaptive fusion.
6. **AI:** IBM Granite explainer (or mock), TF-IDF RAG over 4 engineering docs.
7. **Digital twin:** Real IBM satellite CAD, Kepler-driven orbit, fault-responsive parts.
8. **Frontends:** Streamlit dashboard (rich 3D) + React console (authenticated) + FastAPI backend.
9. **Validation:** 30 test suites, e2e dry run, NASA PCoE external benchmark, eclipse-aware false alarm suppression.
10. **Key innovation:** Physics + ML + adaptive fusion with guaranteed score-flag coherence.

**One line:** "MissionMind is a Kepler-propagated, physics-aware anomaly detection + Granite-grounded diagnosis system for small-satellite telemetry, validated against real NASA data, with 7-second detection and 39-minute advance warning."

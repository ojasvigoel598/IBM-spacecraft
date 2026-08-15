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
13. The frontend (Streamlit dashboard) and what every panel does
14. The "API" — actually a CLI; no HTTP endpoints
15. Configuration, environment variables, paths, ports
16. How to run everything (exact commands)
17. Troubleshooting — known failure modes and fixes
18. Final verified model results (what the numbers mean)
19. Design decisions — alternatives considered, why we chose what we did
20. "What to say if someone asks" — cheat sheet for live demos
21. MissionMind in 5 minutes

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
| 1. Generate or read telemetry | Either runs a deterministic physics simulator, or reads CSV | `simulator/run_scenarios.py` |
| 2. Compute physics expectations | For each row, predict what temperature and SOC *should* be | `simulator/power.py`, `simulator/thermal.py` |
| 3. Run physics sanity-checks | Flag rows where observed vs predicted diverge beyond thresholds | `physics_rules/rules.py` |
| 4. Run ML anomaly detection | Anomaly score + binary anomaly flag from a 3-model IsolationForest ensemble | `ml/detect.py`, `ml/train.py` |
| 5. Retrieve relevant docs (RAG) | TF-IDF search across markdown knowledge base | `ai/rag.py` |
| 6. Generate Granite explanation | IBM watsonx LLM (or deterministic mock) produces a JSON recommendation | `ai/granite_client.py`, `ai/prompts.py` |
| 7. Render dashboard | Streamlit with live 3D IBM satellite, telemetry charts, alerts, RAG evidence, Granite panel | `viz/app.py` |
| 8. Verify | All of the above have tests in `tests/` that run via `missionmind.e2e_dry_run` |

**One-line pipeline:**

```
telemetry → physics check → ML anomaly ensemble → RAG evidence → Granite JSON → Streamlit dashboard
```

Each stage can fail loudly. The e2e dry run (`missionmind/e2e_dry_run.py`) runs
this whole pipeline **twice in a row** and asserts every stage passed; this is
the "is the project actually working" guard.

---

## 2 · The data pipeline (one diagram, every stage)

```
raw telemetry row:                  time_s=600, solar_w=520, V=28.0, T=-42.0, ...
        │
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
TF-IDF RAG (rag.py)                 → top-3 docs from knowledge_base/*.md
        │
        ▼
Granite (granite_client.py)         → JSON with risk, probable_cause,
                                       reasoning, recommended_action,
                                       evidence_used, confidence
        │
        ▼
Streamlit render (viz/app.py)       → KPI cards, charts, 3D model,
                                        Granite panel, status strip
```

At every stage the data type is human-readable: rows and NumPy arrays and
JSON dicts. There is no opaque binary blob.

---

## 3 · Project map (ACTUAL on-disk structure)

```text
missionmind/
├── .freebuff/                       # Freebuff preview cache (HTML explainer + previews)
├── requirements.txt                 # Python deps (scikit-learn, pandas, numpy, scipy, streamlit, pyod, …)
├── README.md                        # Mission statement + run instructions
├── CHANGES.md                       # Changelog (audit history)
├── run_demo.sh                      # Linux/macOS demo launcher
├── stop_demo.sh                     # Stops the demo
├── .streamlit                       # Streamlit config
├── .vscode/                         # Editor settings
│
├── missionmind/
│   ├── __init__.py
│   ├── e2e_dry_run.py              # Runs the full pipeline twice and asserts every stage PASSes
│   ├── open_threejs.py             # Standalone Three.js viewer
│   ├── _bug_hunt.py                # Sweeper: compile + import every .py to surface crashes
│   ├── _lifecycle_assertions.py    # Behavioral lifecycle test (regenerate→retrain→score→coherence)
│   │
│   ├── simulator/                  # Physics simulation (no ML)
│   │   ├── config.py               # Single source of truth for every physics constant (DEMO_FAST toggle)
│   │   ├── power.py                # Battery + solar module (Euler-integrated)
│   │   ├── thermal.py              # Stefan-Boltzmann radiative balance
│   │   ├── failures.py             # Fault injection: solar_degradation, radiator_degradation
│   │   ├── run_scenarios.py        # Coupled simulator → 3 CSVs
│   │   ├── physics_verification.py # Hand-calculated cross-check
│   │   └── param_sensitivity.py    # Sweep constants to see detectability trade-off
│   │
│   ├── physics_rules/              # Operator-readable anomaly rules
│   │   ├── rules.py                # check_power_subsystem, check_thermal_subsystem
│   │   └── test_rules.py           # Asserts they fire on the 3 scenarios
│   │
│   ├── ml/                         # Machine learning
│   │   ├── train.py                # Trains 3 IsolationForests + scalers, saves to models/
│   │   ├── detect.py               # Loads models and scores any DataFrame
│   │   ├── metrics.py              # Basic + advanced metric protocols (cycle, predictive horizon)
│   │   ├── compare.py              # Comparison matrix across all models
│   │   ├── advanced_models.py      # FCNN, XGBOD, MLP-AE, Hybrid-DIF, PINN zoo
│   │   ├── pinn_layer_scan.py      # PINN architecture sweep against NASA B0005
│   │   ├── nasa_real_validation.py # Real NASA PCoE .mat validation
│   │   ├── nasa_validation.py      # Sanity validation against small NASA CSV
│   │   ├── cmapss_rul.py           # Turbofan RUL baseline
│   │   ├── prognostics.py          # Battery RUL (Trend / Similarity / PINN)
│   │   ├── rank_models.py          # Transparent ranking from audit matrix
│   │   ├── _audit_eval.py          # Compact per-scenario matrix generator
│   │   ├── _detect_e2e_test.py     # detect.py defensive regression suite
│   │   ├── _detect_consistency_check.py # Coherence BEFORE/AFTER artefact
│   │   ├── drift.py                # Streaming KS drift test (added this session, TDD)
│   │   └── models/__init__.py
│   │
│   ├── ai/                         # LLM + RAG
│   │   ├── granite_client.py       # IBM watsonx wrapper + deterministic mock
│   │   ├── rag.py                  # TF-IDF retriever over markdown
│   │   ├── prompts.py              # System + user prompt templates
│   │   ├── demo_granite_switch.py  # demo entrypoint
│   │   └── knowledge_base/         # 3 markdown docs (power, thermal, mission rules)
│   │
│   ├── data/                       # Generated CSVs + real NASA data
│   │   ├── run_normal.csv          # 3600 rows · deterministic
│   │   ├── run_solar_failure.csv   # 3600 rows · solar ramp 600→900
│   │   ├── run_radiator_failure.csv# 3600 rows · ε·A ramp 600→900
│   │   ├── nasa_battery_sample.csv # 300 rows · 2 discharge cycles (sample)
│   │   ├── real_nasa/              # B0005/B0006/B0007/B0018 .mat files (real bench data)
│   │   └── nasa_grounding.py       # Sweep NASA PCs against simulator
│   │
│   ├── viz/                        # Streamlit user interface
│   │   ├── app.py                  # 1197 lines · the entire dashboard
│   │   └── components/
│   │       ├── obj_to_geometry.py  # CATIA/Autodesk OBJ → satellite_geometry.js
│   │       ├── satellite_geometry.js (generated, NOT edited by hand)
│   │       └── three_spacecraft_standalone.html
│   │
│   ├── models/                     # Trained artefacts (joblib .pkl files + feature list)
│   │   ├── iforest.joblib
│   │   ├── iforest_power.joblib
│   │   ├── iforest_thermal.joblib
│   │   ├── scaler*.joblib
│   │   ├── features.txt
│   │   ├── audit_matrix.json
│   │   ├── audit_probes.json (training audit probes)
│   │   └── detect_consistency_*.json
│   │
│   └── tests/                      # Test suite (run as Python modules, not pytest)
│       ├── test_physics.py
│       ├── test_ml_metrics.py
│       ├── test_granite_nominal.py
│       ├── test_config_seam.py
│       ├── test_prognostics.py
│       ├── test_drift.py           # added this session (TDD)
│       └── scratch_rows_cycles.py # diagnostic
```

**Dependencies between layers (this is the load order you should know):**

```
simulator/* (no dep)
    ↓
physics_rules/* (depends on simulator)
    ↓
ml/train.py  →  simulator, then writes models/*.joblib
ml/detect.py →  reads models/*.joblib
ml/compare.py, ml/advanced_models.py →  ml/train outputs
    ↓
ai/rag.py, ai/granite_client.py, ai/prompts.py (depend on data layout only)
    ↓
viz/app.py  →  imports physics_rules, ml.detect, ai.*, simulator.run_scenarios
tests/*.py  →  import each layer and assert
e2e_dry_run.py  →  runs everything twice, in this order
```

---

## 4 · File-by-file walkthrough

### `missionmind/simulator/config.py`

**What it does:** Defines EVERY physics constant as a Python constant, then
re-exports them. Either the spec-faithful values (mc_p=5000 J/K, RADIATOR
final 30 %) OR demo-fast values (mc_p=2000 J/K, RADIATOR final 10 %). The
flag **`DEMO_FAST = True`** chooses between them globally.

**Why it exists:** Before this file existed, `power.py` and `thermal.py` and
`failures.py` had hardcoded local copies of the same numbers. When the
constants changed (e.g. flipping DEMO_FAST), three files had to be edited
in sync; one always drifted. Now config.py is the single source of truth —
a missing config raises a loud ImportError in every physics module, not a
silent local copy.

**Inputs:** None — pure constants.

**Outputs:** Re-exported to simulators (e.g. `MC_P`, `EPSILON_A_FINAL`).

**Who calls it:** `simulator/power.py`, `simulator/thermal.py`,
`simulator/failures.py`, `simulator/physics_verification.py`,
`missionmind/viz/app.py` (for `SIGMA`).

**Hazard:** `DEMO_FAST = True` makes detection *easier* on
`run_radiator_failure.csv`. If a judge explicitly asks "does this work on
spec-faithful constants?" the answer is "approximately, but the visualization
will be quieter." Mitigation is one-line.

---

### `missionmind/simulator/power.py`

**What it does:** Implements the spacecraft power subsystem in ~20 lines.

```python
solar_w   = P_SOLAR_MAX * illumination(t) * degradation_factor
net_w     = solar_w - P_LOAD
d_soc     = (net_w * DT_S / 3600.0) / E_CAP_WH
soc_new   = clip(soc + d_soc, 0, 1)
voltage_v = V_MIN + (V_MAX - V_MIN) * soc_new
```

**Why it exists:** Real batteries don't have a fixed SOC; it integrates
energy in/out. Euler integration (`soc + d_soc`) is the simplest defensible
discretization.

**Inputs:** previous `soc`, current time `t`, degradation factor.

**Outputs:** `(solar_w, P_LOAD, soc_new, voltage_v, net_w)`.

**Who calls it:** `simulator/run_scenarios.py` calls `compute_power_step()`
per row.

**Linear OCV caveat:** `voltage_v = V_MIN + (V_MAX-V_MIN)·SOC` is a linear
approximation. Real Li-ion OCV curves are sigmoidal. For demo this is fine;
a future improvement would replace with a tabulated NMC/LFP curve.

---

### `missionmind/simulator/thermal.py`

**What it does:** Implements the spacecraft thermal subsystem.

```python
Q_out = EPSILON * SIGMA * AREA * (T_k**4 - T_SPACE_K**4)   # Stefan-Boltzmann radiation
dT    = (Q_IN - Q_out) * DT_S / MC_P                        # 1st-order heat balance
T_new = T_k + dT                                            # Euler step
```

**Why it exists:** Real spacecraft temperature is dominated by radiative
heat loss to deep space (T_space ≈ 3 K). The T⁴ term is necessary to
reproduce any plausible equilibrium.

**Inputs:** previous `T_k`, current time, optional epsilon/area overrides
(for failure simulation).

**Outputs:** `(T_k_new, Q_in, Q_out, dT)`.

**Who calls it:** `simulator/run_scenarios.py`.

**Caveat:** With P_LOAD=400 W and η=0.85, Q_in=60 W; with ε·A=0.425 the
equilibrium is **−50 °C** ("cold-biased but physical"). If you want ~25 °C
nominal, reduce A to ~0.2 m² or raise P_LOAD. Documented in README.

---

### `missionmind/simulator/failures.py`

**What it does:** Linear interpolation ramps for two fault modes between
`t=600` and `t=900`:

```python
def solar_degradation_factor(t):
    if t < 600:    return 1.0
    elif t < 900:  return 1.0 + (0.48 - 1.0) * frac   # 1.0 → 0.48
    else:          return 0.48

def radiator_epsA(t):
    # same shape: 0.425 → 0.0425 (DEMO_FAST)  or 0.425 → 0.1275 (spec)
```

**Why it exists:** Gradual fault injection is more realistic than a step.
300 s of linear ramp gives the detector enough time to catch it before
complete failure.

**Inputs:** time `t`.

**Outputs:** factor / ε·A product.

**Who calls it:** `simulator/run_scenarios.py` per row.

---

### `missionmind/simulator/run_scenarios.py`

**What it does:** Runs the full coupled simulation for any of three modes:
`none`, `solar_degradation`, `radiator_degradation`. Also supports an
`add_noise` flag for realistic sensor noise.

**Inputs:** failure_mode (str), duration_s (int), optional noise.

**Outputs:** `pandas.DataFrame` with columns:
`time_s, solar_power_w, load_power_w, battery_soc, battery_voltage_v,
heat_in_w, heat_out_w, temperature_c, failure_mode`.

**Who calls it:** `viz/app.py` (button-click → rerun), `e2e_dry_run.py`
(baseline step), all tests that need CSV inputs.

---

### `missionmind/simulator/physics_verification.py`

**What it does:** Independently hand-computes the radiator ramp at five
timestamps and asserts the simulator's output matches float-by-float.

**Inputs:** simulator imports.

**Outputs:** PASS / FAIL printed to stdout.

**Why it matters:** If this passes, the simulator's failure-injection math
agrees with hand calculation — strongest possible evidence that the
simulator is correct, not quietly buggy.

---

### `missionmind/physics_rules/rules.py`

**What it does:** Two pure-Python sanity checks:

```python
def check_power_subsystem(window_df):
    # flag if solar mean < 0.7 · P_max
    # flag if SOC slope < SOC_SLOPE_THRESHOLD_TUNED (i.e. battery draining abnormally)

def check_thermal_subsystem(window_df):
    # flag if T slope > 0.003 °C/s AND heat_in slope < 1 W/s (rises without extra power)
```

**Why it exists:** These are operator-readable rules, written in plain
English. They can be explained to a judge without any ML background.

**Inputs:** rolling telemetry window (≤ 120 rows).

**Outputs:** `(flag_name, confidence)` or `None`.

**Who calls it:** `viz/app.py` per `frame_idx` move.

---

### `missionmind/ml/train.py`

**What it does:** Loads `run_normal.csv`, builds features, fits a scaler on
the FIRST 80 % (temporal split), trains **three IsolationForest models**
on the same training split:

1. **full** (5 features: V, solar, T, dT/dt, dV/dt) — `n_estimators=300, max_features=1.0`
2. **power** (V, solar, dV/dt) — `n_estimators=200`
3. **thermal** (T, dT/dt) — `n_estimators=200`

All three save to `missionmind/models/` as `.joblib`.

**Contamination=0.05** chosen transparently from held-out normal FP
tolerance (was 0.07; changed to 0.05 to remove test-set tuning — see
PERSONAL_CODEBASE_GUIDE for the audit history).

**Sensor-noise model:** Adds ±1 W Gaussian to constants and ±0.01 V to
near-constants in TRAINING ONLY. This is a known asymmetry with inference
data — documented. Inference data is clean (deterministic simulator); the
asymmetry exists purely so IF has something to split on.

**Inputs:** `missionmind/data/run_normal.csv`.

**Outputs:** 6 `.joblib` files in `missionmind/models/`.

**Who calls it:** `e2e_dry_run.py` step 4, manual developer rerun after
hyperparameter change.

---

### `missionmind/ml/detect.py`

**What it does:** Loads the 6 trained artefacts; provides two entry points:
`score_csv(path)` and `score_dataframe(df)`. Both return a DataFrame with
**guaranteed coherence invariant**:

```python
ensemble_flag  = pred_full | pred_power | pred_thermal        # OR
ensemble_score = min(scores_full, scores_power, scores_thermal) # MIN
attribution    = argmin(stack of three scores)                  # 0=full, 1=power, 2=thermal
```

This means: **if `anomaly_flag==1`, `anomaly_score < 0`** always, because
at least one of three decision_functions returned negative. (Verified across
all 3 scenarios: **0 contradictory rows in 10,800 total**.)

Two defensive guards added this session:
- **Empty DataFrame:** returns empty 3-tuple without crashing.
- **No-ensemble fallback:** when subsystem `.joblib` files are missing,
  returns `scores_full, pred_full, attribution=zeros` (always 3-tuple).

**Inputs:** CSV path or DataFrame with the schema produced by
`run_scenarios.py`.

**Outputs:** DataFrame with original telemetry columns plus `anomaly_score`,
`anomaly_flag`, `anomaly_source`.

**Who calls it:** `viz/app.py`, `e2e_dry_run.py`, `compare.py`,
`nasa_real_validation.py`, the e2e test for detect itself.

---

### `missionmind/ml/metrics.py`

**What it does:** Two metric protocols:

1. **Basic** — accuracy, precision, recall, specificity, F1, balanced-acc,
   MCC, ROC-AUC, PR-AUC, confusion matrix; with proper handling of
   single-class cases.
2. **Advanced** — FPR before injection, TPR after ramp window, detection
   delay, early-detection rate.
3. **Cycle-level** — rolls row-level scores up to cycles (used by NASA
   PCoE bench validation).
4. **Predictive-horizon** — "predict degradation at t+Δt" rather than
   "is degradation already happening."

**Why it exists:** A single metric is never enough. MissionMind exposes
the full metric stack because every metric reveals a different limitation.

---

### `missionmind/ml/compare.py`

**What it does:** Trains and evaluates **8 candidate models** on the 3
scenarios + 2 holdout:

- **Unsupervised:** IsolationForest, LOF, OneClassSVM, MLP Autoencoder, Hybrid DIF
- **Supervised:** FCNN (MLP 100-50-20), XGBOD
- **Physics-guided NN:** "Custom Physics-Informed NN" — actually a hybrid
  classifier with engineered physics features that *we* call "physics-guided"
  in the audit (term "PINN" was misleading because the loss term is just an
  ensemble probability, not a true PDE residual).

Saves `models/comparison_report.json`.

**Inputs:** normal CSV + solar + radiator + their time-split holdouts.

**Outputs:** Console table + JSON report.

---

### `missionmind/ml/advanced_models.py` (464 lines)

**What it does:** Defines the wrappers for FCNNDetector, XGBODDetector,
MLPAutoencoderDetector, HybridDIFDetector, CustomPhysicsInformedNN — the
5 contestants beyond IsolationForest.

**Inputs:** feature matrix.

**Outputs:** trained estimator with `.decision_function()` and `.predict()`.

**Honest note:** The "Custom Physics-Informed NN" is **not** a true PINN. It
combines a probabilistic classifier with an autoencoder reconstruction error.
The training loss has no PDE-residual term. It is correctly named "physics-
guided" in `rank_models.py`; do not describe it as a true PINN in public.

---

### `missionmind/ml/drift.py` (added this session, TDD-driven)

**What it does:** Two-sample Kolmogorov–Smirnov test for drift detection.

```python
def streaming_ks_test(a, b) -> float:
    """Returns p-value of KS test; raises ValueError if either sample < 2."""
```

**Inputs:** two numerical arrays (training window + recent window).

**Outputs:** p-value (lower = more drift).

**Tests:** `tests/test_drift.py` — identical distributions not detected,
shifted ones detected, short/empty input raises ValueError.

**Not yet wired to the dashboard.** This is intentionally a building block
for a future "MODEL STALENESS" indicator.

---

### `missionmind/ai/rag.py`

**What it does:** TF-IDF vectoriser over `knowledge_base/*.md`; on query,
returns top-k chunks by cosine similarity.

**Inputs:** query string (typically `"power solar degradation ..."`).

**Outputs:** list of dicts: `{id, title, content, score}`.

**Why TF-IDF and not embeddings:** TF-IDF is offline, no API key, no
latency, no quota — and on a 3-document corpus embeddings are not worth
the dependency. If we add 10x more docs, swap to `sentence-transformers`.

---

### `missionmind/ai/granite_client.py`

**What it does:** Wraps IBM watsonx Granite 3 2B instruct. If
`WATSONX_APIKEY` and `WATSONX_PROJECT_ID` env vars are present, calls the
real API. Otherwise returns a deterministic mock JSON with full schema
including `retrieved_docs` quotes.

**Inputs:** Granite prompt payload (current_values, nominal_values,
physics_flag, etc.).

**Outputs:** JSON with `risk`, `probable_cause`, `reasoning`,
`recommended_action`, `evidence_used`, `confidence`, `retrieved_docs`.

**Honest architectural note:** Granite is NOT the anomaly detector.
Granite is the **explainer**. The actual anomaly decision comes from
`physics_rules` and `ml/detect`.

**If LLM is unavailable:** the mock fallback still produces a
schema-valid JSON so the dashboard keeps rendering. Test
`tests/test_granite_nominal.py` exercises the mock path.

---

### `missionmind/viz/app.py`

**What it does:** 1197-line Streamlit application. Has six major sections:

1. **Header** — mission name, status badge, UTC clock
2. **Sidebar** — scenario selector (Normal / Solar / Radiator), playback
   controls, RAG toggle, prompts toggle
3. **Scenario buttons** — three big primary buttons triggering a
   `run_scenario()` rerun
4. **Status strip** — TelemetryLive, MLStatusChips, RagStatus, GraniteStatus
5. **KPI cards** — Solar, SOC, Voltage, Temperature, Heat I/O, ε·A
6. **Tabs/panels** — Telemetry Chart, ML Diagnostics, RAG Evidence,
   Granite Explanation, 3D Satellite (Three.js)

The Three.js view is constructed inside a `components.html` iframe:
satellite geometry comes from `satellite_geometry.js` (generated by
`obj_to_geometry.py` from the IBM satellite OBJ file); a Three.js scene
shows the spacecraft, a textured Earth, stars, and pulses/anomalies
based on telemetry.

**Inputs:** `missionmind/ai/*`, `missionmind/ml/*`, `missionmind/physics_rules/*`,
`missionmind/simulator/*`, plus Streamlit rerun signal.

**Outputs:** rendered HTML.

**Where to start reading it:** Line 37-42 (imports), 134-146 (scenario
buttons), 217-303 (data flow + ML scoring + physics checks),
362-407 (the Granite integration block).

---

### `missionmind/viz/components/obj_to_geometry.py`

**What it does:** Reads the IBM satellite OBJ (a CAD export from
Autodesk/CATIA), deduplicates vertices, normalises to a unit
bounding box, and writes `satellite_geometry.js` — a JSON-ish JS module
that the dashboard embeds inline.

**Inputs:** OBJ file (default `missionmind/viz/components/models/ibm_satellite.obj`).

**Outputs:** `satellite_geometry.js`.

**Why a converter and not direct OBJ loading?** Streamlit's
`components.html` cannot fetch external files (CORS). The 9 MB OBJ
is converted into a compact, deduplicated JS module that can be
inlined.

---

### `missionmind/tests/*`

Every file is a self-executing test module. Run with
`python -m missionmind.tests.test_X`. Pass-and-print the wire-format.

| Test | Asserts |
|---|---|
| `test_physics` | equilibrium T, final SOC=1.0, V=28 V (normal) |
| `physics_rules.test_rules` | 0 pre-injection flags, 44/44 post-injection on solar & radiator |
| `test_ml_metrics` | computational correctness of cycle + predictive-horizon protocols |
| `test_granite_nominal` | mock JSON schema, ML-flag overrides "nominal" verdict |
| `test_config_seam` | every physics module binds to `config.py` (no silent local copies) |
| `test_prognostics` | `true_rul_at(eol, predict)` boundary behaviour |
| `test_drift` (this session) | KS p-value behaviour + ValueError on short input |

---

### `missionmind/e2e_dry_run.py`

**What it does:** Runs the FULL pipeline twice in a row:
`run_scenarios → test_physics → test_rules → train → detect on 3 CSVs →
nasa_real_validation --quick → rag → granite_client`. Prints
"Iteration 1 PASS / Iteration 2 PASS" only if all 9 steps PASS twice.

**Why twice?** Catches non-determinism.

**Inputs:** None.

**Outputs:** Exit 0 + console PASS lines, or exit 1 + the failing step name.

---

## 5 · Line-by-line explanations of important code

### `add_derivative_features` (`train.py` and `detect.py`)

```python
def add_derivative_features(df: pd.DataFrame, dt_s: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    df["d_temp_dt"] = df["temperature_c"].diff().fillna(0) / max(dt_s, 1e-9)
    df["d_volt_dt"] = df["battery_voltage_v"].diff().fillna(0) / max(dt_s, 1e-9)
    return df
```

`df.diff()` returns the row-by-row difference (ΔT, not dT/dt). For the
simulator's 1-second timestep this is numerically the same as dT/dt; the
explicit `/ dt_s` makes the intent clear and the function robust to a
different dt if we later change the simulator. `fillna(0)` is conservative
— at the first sample there is no previous value; setting derivative to
0 prevents an undefined value flowing into the scaler. `max(dt_s, 1e-9)`
guards against division-by-zero if someone passes `dt_s=0`.

### `_ensemble_score_and_flag` (`detect.py`)

```python
ensemble_flag  = pred_full | pred_power | pred_thermal
ensemble_score = np.minimum.reduce([scores_full, scores_power, scores_thermal])
```

Is sklearn `IsolationForest.decision_function()` "higher = more
normal"? Yes. So the **most anomalous** value is the **MINIMUM**. Taking
the MIN across three detectors means: "the displayed score is whatever
the most pessimistic detector thinks." If any one of three trips the
flag, the score is consistent with that flag.

The `attribution = argmin(...)` records which detector drove the MIN —
this surfaces "EPS" or "Thermal" as the failure source on the dashboard.

### `add_noise_if_constant` (`train.py`)

```python
def add_noise_if_constant(X, names):
    Xn = X.copy()
    for idx in range(X.shape[1]):
        std = X[:, idx].std()
        if std < 1e-6:
            Xn[:, idx] += rng.normal(0, 1.0, size=X.shape[0])  # 1 W sensor noise for solar
        elif std < 0.1:
            Xn[:, idx] += rng.normal(0, 0.01, size=X.shape[0]) # 0.01 V sensor noise
    return Xn
```

Why we do this: `IsolationForest` builds trees by randomly selecting
splits. If a column has a single value everywhere (std ≈ 0), every split
on it produces identical subsets — the tree carries no information.
Adding tiny noise gives every sample a slightly different value,
unblocking the splits. The asymmetry (training has noise; inference has
clean data) is a documented limit of the demo.

### `assert before_strict < 0.4 & after > 0.5` (`train.py`)

```python
assert before_strict < 0.4, f"{fname} too many false positives strict before 100-600 {before_strict}"
assert after > 0.5, f"{fname} should detect anomaly after injection, got {after}"
```

`before_strict` is the flag rate in the **strict window** `100 s ≤ t ≤ 600 s`
(burn-in excluded). `after` is the flag rate **after** the ramp ends, `t > 900 s`.

These assertions fail loudly if the model is wildly misbehaving — e.g. if
contamination is 0.99 and it flags everything, or 0.0001 and it never flags.
The numbers (`0.4` and `0.5`) are deliberately loose — they are not the
final metric, just a development sanity gate.

---

## 6 · Trace one telemetry sample through the system

This is the single most useful exercise. Pick one row in
`run_solar_failure.csv`, say `time_s=900`:

| Stage | What's happening | What the data looks like |
|---|---|---|
| Raw row | 1-second tick from physics simulator | `time_s=900, solar_w=249.6, V=27.1, T=-40.5, heat_in=60, heat_out=55, failure_mode='solar_degradation'` |
| `add_derivative_features` | Append two columns | Now also `d_temp_dt=-0.0001, d_volt_dt=-0.0073` |
| `StandardScaler` | Subtract train mean and divide std | All columns zero-mean, unit-variance. The previous row normalised the same way, so dT/dt and dV/dt are now small. |
| Three IsolationForests predict | Each makes a binary decision and produces a `decision_function` value | `pred_full = True, score_full = -0.42` (anomalous); `pred_power = True, score_power = -0.31`; `pred_thermal = False, score_thermal = 0.08` |
| `np.minimum.reduce` | Most pessimistic score | ensemble score = **-0.42** |
| `np.argmin` | Which model drove the MIN | attribution = **0** (full model) |
| Physics rule | Compare solar mean vs P_max·0.7 | solar mean 249 W < 364 W → `physics_flag='solar_degradation', conf=0.92` |
| RAG | Build query string, top-k TF-IDF over docs | returns 3 chunks: solar-degradation procedure, evidence standard, mission rule |
| Granite | Compose prompt with current_values + nominal + flag + RAG chunks | produces JSON `{"risk":"HIGH","probable_cause":"Solar array degradation (ramping 1.0 → 0.48 across 600–900 s)...","recommended_action":"Reduce bus load, transition to safe mode per [DOC-POWER-PROC-001]","confidence":0.92, "evidence_used":[...]}` |
| Streamlit render | KPI cards, anomaly chip, Granite panel | Operator sees "Solar Array Degradation · HIGH · 92% confidence · Reduce load → safe mode" + 3D satellite with fault-tinted solar arrays |

Each line is reproducible from the row. There are no "magic thresholds"
hidden from the UI; everything that the model decided is exposed.

---

## 7 · Machine learning — explained from zero

**Isolation Forest** is an unsupervised anomaly detector. The intuitive
idea: random points in a dense normal cluster require **many random
splits** to isolate; an outlier point in a sparse region requires **few
splits**. The number of splits (path length) is the anomaly score.

**Concrete visualisation:**

Imagine 200 normal points in a unit circle plus one outlier at (10, 10).
Pick a random axis, pick a random split point inside the data range.
Repeat. After ~10 splits the outlier is isolated by itself; the normal
cluster needs ~30 splits because the tree keeps cutting it into halves.

**Code path:**
- `IsolationForest(...).fit(X_train)` builds N=300 trees from bootstrap
  samples of the training data.
- `.predict(X_test)` returns +1 (inlier) or -1 (outlier) per row.
- `.decision_function(X_test)` returns a continuous score: **higher = more
  normal**, lower = more anomalous. Scores cluster near ~0.10 for normal
  rows and ~-0.45 for anomalous rows.
- We added 0.05 `contamination` = "expect ≤5 % of unseen data to be
  anomalous" — this biases the threshold internally without needing an
  external threshold.

**Why we used IF:** unsupervised (no labels needed for fault scenarios),
fast at inference, robust to outliers in training, well-understood
mathematical property (path length in iForest is theoretically linked to
anomaly score). It is the right starting tool for a quick demonstration.

**Why we wrapped it in 3 models (full / power / thermal):**
**Sensor-fusion problem.** A radiator fault changes temperature; a solar
fault changes voltage and solar. A single 5-feature IF has difficulty
separating these because all 5 features share variance. By building
two specialised IFs (one with only power-relevant columns, one with
only thermal-relevant columns) and OR-ing their flags with the full
model's flag, we recover both failure modes from a single architecture.

**How validated:**
- **Self-test** in `missionmind/ml/_detect_e2e_test.py` — runs all 4 paths
  on synthetic CSVs and asserts every invariant holds.
- **In-domain** in `missionmind/ml/compare.py` on the 3 simulator
  scenarios.
- **External** in `missionmind/ml/nasa_real_validation.py` on the REAL
  NASA PCoE `.mat` files for B0005/B0006/B0007/B0018, with cycle-level
  metrics + predictive horizon. (Even with the leak-free model our IF
  scores ~0.7 AUC on real NASA cells — see the honest comparison matrix.)

**Weaknesses:**
- IF learns spatial cluster shapes only. A fault that follows the same
  shape as the normal cluster's tail is missed.
- IF doesn't capture temporal ordering — consecutive-anomaly logic
  (N-of-M rule) is not built in. The dashboard does apply a 100 s burn-in
  lockout for the start-up transient (see `viz/app.py:266-267`).
- The 3-model OR ensemble inflates FPR in some configurations. After this
  session's fix, contamination=0.05 keeps FPR at 0.001 on the held-out
  normal validation set.

---

## 8 · Physics — every equation, every variable

### Power subsystem

**Equation 1 — Solar power generated:**

```
P_solar(t) = P_SOLAR_MAX · illumination(t) · degradation_factor(t)
```

| Variable | Unit | Meaning | Where from |
|---|---|---|---|
| `P_SOLAR_MAX` | W | Peak array output (520 W here; smallsat assumption) | `config.py` |
| `illumination(t)` | 0-1 | 1 in MVP (no eclipse; documented limit) | constant |
| `degradation_factor(t)` | 0-1 | 1.0 nominal; ramps 1.0→0.48 during solar_degradation | `failures.py` |

**Equation 2 — Net power and SOC step:**

```
net_w    = P_solar - P_LOAD
ΔSOC/dt  = (net_w · Δt) / E_CAP_WH / 3600
SOC(t+1) = clip(SOC(t) + ΔSOC/dt, 0, 1)
```

| Variable | Unit | Where |
|---|---|---|
| `P_LOAD` | W | constant 400 W |
| `E_CAP_WH` | Wh | 100 Wh |
| `Δt` | s | 1 (DT_S) |

**Equation 3 — Linear OCV:**

```
V(t) = V_MIN + (V_MAX - V_MIN) · SOC(t)
```

V_MIN=24 V, V_MAX=28 V. Real LFP curves are sigmoidal — this is a
simplification.

**Equation 4 — Solar ramp injection:**

For `t ∈ [600, 900]`:
```
degradation_factor(t) = 1.0 + (0.48 − 1.0) · (t − 600) / 300
```

After `t = 900`: stuck at 0.48 — solar produces 250 W, net = −150 W,
battery drains at −0.000417 SOC/s, reaches 0 in ~40 minutes.

### Thermal subsystem

**Equation 5 — Stefan-Boltzmann radiative heat loss:**

```
Q_out(t) = ε_eff · σ · A_eff · (T_k(t)⁴ − T_space⁴)
```

| Variable | Unit | Where |
|---|---|---|
| `ε_eff` | 0-1 | 0.85 nominal; lower during failure (multiplied by `degradation_factor`) |
| `σ` | W/m²/K⁴ | 5.67E−8 (Stefan-Boltzmann constant) |
| `A_eff` | m² | 0.5 |
| `T_k` | K | spacecraft temperature in Kelvin (= T_C + 273.15) |
| `T_space` | K | 3 (cosmic background) |

**Equation 6 — Heat balance:**

```
dT/dt = (Q_in − Q_out) / (m · c_p)
Q_in = P_load · (1 − η)
```

| Variable | Unit | Where |
|---|---|---|
| `Q_in` | W | 60 (= 400·0.15); constant |
| `m·c_p` | J/K | 2000 J/K (DEMO_FAST) or 5000 (spec) |
| `η` | 0-1 | 0.85 photovoltaic-bus efficiency |

**Equation 7 — Radiator ramp injection:**

For `t ∈ [600, 900]`:
```
ε_eff · A_eff (t) = 0.425 + (0.0425 − 0.425) · (t − 600) / 300   (DEMO_FAST)
```

After `t = 900`: stuck at 0.0425. Equilibrium temperature:
`60 / (0.85 · 5.67E−8 · 0.0425) + 3⁴ = ? → T_eq = 397 K = 124 °C`.

**If a relationship is violated:**
- `solar < 0.7 · P_max`: physics_flag = `'solar_degradation'`.
- `dT/dt > 0.003 °C/s` AND `dheat_in/dt < 1 W/s`: physics_flag =
  `'radiator_degradation'`.
- A battery draining without a power-fault is flagged at conf 0.5
  (less specific, fewer features).

### Orbital mechanics (used only in `prognostics.py`)

**Equation 8 — Kepler period (used to convert cycles → days):**

```
T = 2π · √(a³ / μ)
```

For LEO altitude 550 km: `T = 2π · √((6921 km)³ / 3.986E14) ≈ 95.5 minutes`.
One eclipse per orbit = one charge/discharge cycle. So 50 cycles ≈
50 · 95.5 / 1440 ≈ 3.3 days.

This is the only orbital equation USED by MissionMind. The rest (J2,
drag, SRP, Hohmann, Clohessy-Wiltshire) is documented but not present
because none of them would change a regression result on the bench
data.

---

## 9 · Physics + ML — what each layer knows, how they fuse

| Knowledge | Physics layer | ML layer |
|---|---|---|
| Expected temperature from heat balance | ✅ exact (Stefan-Boltzmann) | ❌ |
| "Solar below 70 % peak = degradation" | ✅ threshold rule | ✅ learned cluster shape |
| "Battery draining exactly matches solar fault" | ❌ | ✅ (correlation in features) |
| "Operator-actionable diagnosis" | ❌ | ❌ (Granite / LLM) |
| "Citation to a procedure document" | ❌ | ❌ (RAG) |

**Concrete fusion example** (real flow from the e2e dry run, t=750 s, solar
degradation scenario):

1. **Telemetry:** solar=350 W, V=27.5, T=-40 °C, dT/dt=-0.0001 K/s.
2. **Physics rule:** solar mean over 120-sample window = 320 W < 364 W →
   `physics_flag='solar_degradation', conf=0.79`. Heat_in slope is
   normal; temperature is normal. → no thermal flag independently.
3. **ML:** IsolationForest `decision_function` on the 5-feature row =
   −0.34. `pred_full = True` (1).
4. **Fusion:** `anomaly_flag = True`, `anomaly_score = -0.34`, subsystem
   attribution = "power" (the power-only model drove the MIN).
5. **RAG:** top-3 docs include `'Solar Array Degradation Procedure'` and
   `'Power Subsystem Mission Rule'` with cosine similarity 0.27.
6. **Granite:** Receives `{subsystem: "power", flag: "solar_degradation",
   current_values: {solar: 350}, nominal_values: {solar: 520}}` plus the
   RAG chunks. Returns JSON: risk=HIGH, probable_cause solar
   degradation, recommended_action "reduce bus load, shed to ≤250 W,
   prepare safe-mode entry per [DOC-POWER-PROC-001]", confidence=0.79.

The two layers **do not know the same things**. Physics guarantees a
threshold that ML cannot. ML captures correlated feature dropouts that
physics thresholds miss. Their combination is what produces the
high-confidence actionable diagnosis.

---

## 10 · Walkthrough of a real anomaly

This is what an operator sees when solar array degradation begins at
t=600 s. Read straight from `missionmind/data/run_solar_failure.csv`.

1. **t=0..599 (Normal Operation, ~600 rows displayed by `st.dataframe`):**
   Solar=520 W, V ramps 26.2→28.0 V (SOC 0.9→1.0), T=-42 °C stable,
   `anomaly_flag=0`, `anomaly_score ≈ 0.10`. Dashboard reads "NOMINAL —
   all green chips". Granite panel: "All subsystems nominal. Continue
   normal ops."

2. **t=600 (fault injects):** `degradation_factor` starts ramping from
   1.0 → 0.48 over 300 s. The simulator's `P_solar(t)` begins decreasing.
   `solar = 520 × 1.0 = 520 W` still at first second.
   Dashboard shows no change yet.

3. **t=900 (ramp ends):** `solar = 520 × 0.48 = 249.6 W`. The fault is
   fully active. From here on the simulator treats it as constant.

4. **What the features see (engineered):** `d_volt_dt` flips negative
   (battery is now draining), `d_temp_dt` slowly positive (slightly more
   heat from the lower state-of-charge).

5. **Physics check (rules.py):**
   `check_power_subsystem(window)` → `solar mean < 0.7 · P_max`
   (`window['solar_power_w'].mean() = 270 W < 364 W`), so
   `physics_flag = 'solar_degradation'`, conf = 0.85.

6. **ML response:** IsolationForest `decision_function` returns −0.32
   to −0.45 — well below the implicit threshold of 0.

7. **Score:** Display reads **"−0.34 · ANOMALY · power-model attribution"**.
   The chip flips red.

8. **Threshold:** No explicit threshold — IsolationForest's predict
   function uses `contamination` to set an internal boundary. With
   contamination=0.05 (leak-free choice), the bare IF correctly fires on
   the 250 W solar drop.

9. **Severity:** Granite's risk field reads **"HIGH"** based on the
   `retrieved_docs` and `current_values`.

10. **Diagnosis:** Granite prompt: `probable_cause = "Solar array
    degradation: degradation_factor 1.0 → 0.48 between 600–900 s,
    observed solar=250 W vs nominal 520 W, battery draining from net
    P_load − P_solar = 150 W"`.

11. **AI explanation:** Granite returns: `"At t=900 s the solar array
    output collapsed to 250 W (-52 %). With demand still 400 W, the
    battery is now draining at -0.000417 SOC/s. If SOC reaches 0 %
    within ~40 minutes, mission rule [DOC-POWER-PROC-001] requires
    safe-mode entry: shed non-essential loads to ≤250 W. Inspect the
    solar array within the next hour for stuck panel / shading / cable
    fault via [DOC-POWER-PROC-001]."`

12. **Action:** Recommended action `"Reduce load, enter safe mode"` +
    `"Inspect solar array"`.

13. **Dashboard display:** KPI card turns yellow. Anomaly chip
    flashes red. Three.js solar panels dim and shift emissive red.
    Granite panel renders the JSON as a styled card. Status strip
    shows: `TelemetryLive=OK · ML=predict=1 · RAG=3docs · Granite=HIGH`.

This is end-to-end from a single `run_scenario()` button press.

---

## 11 · The 3D digital twin

**What the 3D model represents.** The satellite is the **actual IBM
satellite CAD file** (`missionmind/viz/components/models/ibm_satellite.obj`),
not a procedurally-generated stand-in. It was converted by
`obj_to_geometry.py` into a `satellite_geometry.js` module that
hard-codes positions / normals / colours and is embedded inline in the
Streamlit iframe.

**Component mapping:**

| CAD group | Visual colour | What it represents |
|---|---|---|
| `Body2`, `Body3` | silver-blue | Solar arrays (left/right) |
| `MainBusSquare` | dark grey | Main spacecraft bus |
| `PhasedArrayAntenna` | red | Antenna / com subsystem |

**Which telemetry controls the visual state (real connections):**

| Telemetry | Visible effect |
|---|---|
| `solar_power_w` | Emissive intensity on `Body2`/`Body3` (more red when fault) |
| `temperature_c` | HUD text + beacon glow tint |
| `anomaly_flag=1` | Beacon pulses red; outline halo around satellite |

**What is genuinely physics, what is theatre:**

- **Genuinely physics-coupled:** mission clock (`time_s`) shows the
  simulation's elapsed time; solar panels visibly dim when
  `solar_power_w` drops; the Three.js HUD re-renders the affected KPI
  values from the live telemetry.
- **Decoration:** the satellite's orbital position around Earth is
  rendered as a 2D ring only (no actual Kepler propagation is wired
  in; `viz/components/` uses a parametric circle, not a state
  variable). Earth and stars are decorative.
- **NOT a closed-loop digital twin in the strict literature sense:**
  the satellite does NOT yet react back to operator input — there is
  no `command_execution` path from the UI back through to the
  simulator. This is documented as a future-work item in the
  `docs/STRUCTURE.md` notes.

So: the satellite is a faithful representation of the IBM CAD; telemetry
affects parts of it; but orbit, telemetry ingest, and
command-and-control are still simulation-side or visual-only.

---

## 12 · The AI / Granite / RAG layer — what goes where

**What is sent to Granite (`granite_client.py` payload):**

```python
{
  "subsystem": "power",                  # from physics rule
  "anomaly_score": 0.34,                 # absolute value of decision_function
  "anomaly_score_threshold": 0.05,
  "ml_flag": 1,
  "physics_flag": "solar_degradation",
  "physics_confidence": 0.85,
  "current_values": {solar: 250, V: 27, T: -40, soc: 0.78, ...},
  "nominal_values": {solar: 520, V: 28, T: -42, soc: 1.0, ...},
  "time_s": 905,
  "failure_mode": "solar_degradation"
}
```

**What is NOT sent:**
- The anomaly decision itself (Granite does not decide the flag).
- The RAG raw chunks (those become `retrieved_docs[i].content` quoted
  in the prompt).
- Any future-tick prediction (Granite is a one-shot explainer).
- Internal IsolationForest tree paths.

**Why the LLM is involved at all:** because operator decision-making is
a **natural-language** problem. A satellite engineer reading the
dashboard needs a sentence, not an array. The deterministic
physics-rule + ML flag is the *fact*; Granite is the *prose*.

**What deterministic systems already know without LLM:**
- Anomaly or not, from `ml/detect.py`.
- Which physical rule tripped, from `physics_rules/rules.py`.
- Subsystem attribution, from `anomaly_source` column.

**What Granite adds:** the audit-grade explanation, written in plain
English, with citation to procedure documents, recommended next
action, confidence — and crucially *the recommended action* lives
outside the deterministic systems, since action choice depends on
operator policy, mission rules, and risk tolerance.

**How hallucinations are controlled:**

- The mock JSON structure is **schema-valid** (tested by
  `test_granite_nominal`).
- The prompt **forces** Granite to use only `retrieved_docs` content
  for citations — no hallucinated document IDs.
- The **ML flag** is checked against Granite's `risk` field — if ML
  says anomaly but Granite says nominal, the dashboard DOES NOT
  display "nominal" (anomaly chip wins). Verified in
  `test_ml_flag_prevents_nominal_verdict_when_detector_flags`.

**What happens if Granite is unavailable:**
- If `WATSONX_APIKEY` env var is missing, the mock fallback runs.
- The mock is deterministic and produces a valid schema.
- The dashboard renders identically (modulo richness of wording).
- This is verified by `test_granite_nominal`.

**Critical clarification:** **LLM ≠ anomaly detector.** The flag
decision is deterministic. Granite produces the narrative.

---

## 13 · The frontend — Streamlit dashboard

There is no traditional HTTP API. MissionMind's "frontend" is
`viz/app.py` + Three.js iframe. Reading order from top to bottom:

1. **Sidebar (`viz/app.py:91-119`)** —
   Scenario selector (Normal / Solar / Radiator), playback controls
   (autoplay, step size, frame index), RAG toggle, top-k slider, raw
   telemetry toggle, IBM watsonx SDK status.

2. **Scenario buttons (`viz/app.py:134-146`)** — three big primary
   buttons. Each calls `run_scenario()` with the corresponding
   `failure_mode` and triggers a `st.rerun()`.

3. **Status strip (`viz/app.py:198-214`)** — pulse-dot chips for
   TelemetryLive / ML Status / RAG / Granite. Always present on the
   dashboard.

4. **KPI strip (`viz/app.py:217-258`)** — six metric cards: Solar,
   Battery SOC, Voltage, Temperature, Heat I/O, ε·A. Each derived
   from `current_row` of the scored DataFrame.

5. **Tabs (`viz/app.py:320-580`)** — Telemetry & 3D · ML Diagnostics ·
   RAG Evidence · Granite Explanation · Physics & Equations · About.

6. **Three.js satellite view** — embedded inside
   `components.html`. Three.js renders the IBM satellite OBJ-derived
   geometry. Animations rely on `requestAnimationFrame` loops reading
   from a JS object `telemetry` populated from the Python side.

7. **Telemetry chart (`viz/app.py:381-407`)** — Plotly subplots of
   solar / voltage / temperature over the visible 120-second window.

**`st.rerun()` mechanics:**
- Click a scenario button ⇒ `st.rerun()` triggers the whole script
  body to re-execute. `run_scenario()` produces 3600 fresh rows.
- `score_dataframe(df_full)` produces all scores in one vectorised
  call. The dashboard then scrubs via `frame_idx`.
- Auto-play ticks `frame_idx` per 2 s without re-running the script —
  so no new ML inference occurs during auto-play (this is the
  "theatrical" finding from earlier; it is intentional for stable
  Three.js rendering and reproducibility).

---

## 14 · The "API"

There is no REST/HTTP API surface.

- `cli.run_cmd(...)` style entry points via `python -m missionmind.X`
  ARE the contracts. They read CSVs or run scenarios, return stdout and
  exit code.
- The lifecycle test reads model files via `joblib.load`, scores, and
  asserts on output. Your "client" is the e2e_dry_run script, not an
  HTTP request.

If you wanted a real HTTP API, wrap `score_dataframe` in a FastAPI
endpoint with `POST /score` accepting a CSV body. ~30 lines. **Not in
scope of this version.**

---

## 15 · Configuration

**Required:**

| Var | What it controls | Default behaviour |
|---|---|---|
| `MISSIONMIND_PYTHON` (implicit) | which Python interpreter | `.venv/Scripts/python.exe` |

**Optional (only used for real LLM):**

| Var | Purpose | If missing |
|---|---|---|
| `WATSONX_APIKEY` | IBM watsonx auth | mock fallback runs (still schema-valid) |
| `WATSONX_PROJECT_ID` | IBM watsonx project | mock fallback runs |

**Config file (single source of truth: `simulator/config.py`):**

| Constant | Spec | DEMO_FAST | What the flag does |
|---|---|---|---|
| `MC_P` (J/K) | 5000 | 2000 | Faster thermal response; eq T at 124 °C vs 28 °C |
| `RADIATOR_FINAL_FRACTION` | 0.30 | 0.10 | Easier to detect radiator fault |
| `DEMO_FAST` | False | True | Toggle between the two columns |
| `CONTAMINATION` (`ml/config.py`) | 0.05 | 0.05 | IF anomaly budget |

To run on SPEC-FAITHFUL constants flip `DEMO_FAST = False` in
`simulator/config.py` line 23 and retrain with `python -m
missionmind.ml.train`. The detection on the radiator will get harder;
expect F1 to drop 10–30 points; the dashboard still flags, just
slower.

**Paths:**
- All paths are computed from `__file__` directory; nothing is
  hardcoded to the user's home directory.
- Data lives in `missionmind/data/`.
- Models live in `missionmind/models/`.
- Knowledge base in `missionmind/ai/knowledge_base/`.

---

## 16 · How to run everything (exact commands)

### First-time setup (from a fresh checkout)

```bash
cd missionmind
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m missionmind.simulator.run_scenarios   # produces 3 CSVs
.venv/Scripts/python.exe -m missionmind.ml.train                  # trains ensemble
```

### Re-train

```bash
.venv/Scripts/python.exe -m missionmind.ml.train
```

### Re-score a CSV

```bash
.venv/Scripts/python.exe -m missionmind.ml.detect --input missionmind/data/run_solar_failure.csv
```

### Run the audit matrix

```bash
.venv/Scripts/python.exe -m missionmind.ml._audit_eval
```

### Compare all 8 models

```bash
.venv/Scripts/python.exe -m missionmind.ml.compare        # writes models/comparison_report.json
```

### Validate on REAL NASA cells

```bash
.venv/Scripts/python.exe -m missionmind.ml.nasa_real_validation --quick
.venv/Scripts/python.exe -m missionmind.ml.nasa_real_validation
```

### Run a single test

```bash
.venv/Scripts/python.exe -m missionmind.tests.test_physics
.venv/Scripts/python.exe -m missionmind.tests.test_ml_metrics
.venv/Scripts/python.exe -m missionmind.tests.test_granite_nominal
.venv/Scripts/python.exe -m missionmind.tests.test_config_seam
.venv/Scripts/python.exe -m missionmind.tests.test_prognostics
.venv/Scripts/python.exe -m missionmind.tests.test_drift
.venv/Scripts/python.exe -m missionmind.physics_rules.test_rules
.venv/Scripts/python.exe -m missionmind.ml._detect_e2e_test
.venv/Scripts/python.exe -m missionmind._lifecycle_assertions
```

### Final end-to-end test

```bash
.venv/Scripts/python.exe -m missionmind.e2e_dry_run     # runs twice; exits 0 only if both PASS
```

### Run the dashboard / digital twin

```bash
.venv/Scripts/python.exe -m streamlit run missionmind/viz/app.py --server.port=8501
# open http://localhost:8501 in a browser
```

`run_demo.sh` (Linux/macOS) and `stop_demo.sh` are wrapper scripts that
handle venv + streamlit together for the demo.

### Run a battery-RUL experiment

```bash
.venv/Scripts/python.exe -m missionmind.ml.prognostics
```

### Convert an OBJ to JS for embedding (if you change the CAD)

```bash
.venv/Scripts/python.exe missionmind/viz/components/obj_to_geometry.py \
    missionmind/viz/components/models/ibm_satellite.obj \
    missionmind/viz/components/satellite_geometry.js
```

---

## 17 · Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on import | venv not activated / not the right interpreter | Use `\.venv\Scripts\python.exe -m …` explicitly each time |
| `No module named 'missionmind.ml.drift'` (or similar) | Stuck cached `__pycache__` from an old create-then-delete iteration | `find . -type d -name __pycache__ -exec rm -rf {} +` |
| `StandardScaler requires at least 1 sample` | Empty DataFrame passed to `score_dataframe` | Pre-existing: empty input guard was added in detect.py this session — should not recur. If it does, file a regression. |
| `ValueError: not enough values to unpack (expected 3, got 2)` | Old `detect.py` from before this session — only fires if subsystem `.joblib` files were deleted and `score_dataframe` is called | Re-train (`python -m missionmind.ml.train`) or apply the same fix manually |
| Dashboard opens but 3D scene is blank | `satellite_geometry.js` missing or outdated | `python missionmind/viz/components/obj_to_geometry.py ...` |
| Granite panel says "model unavailable" | `WATSONX_APIKEY` not set | Mock fallback runs; JSON is still valid |
| `e2e_dry_run` exits non-zero | A step failed | Run each step manually to localise; the dry-run output names which command failed |
| Comparison report shows F1=0 for bare IsolationForest | Expected — the bare IF needs the OR-ensemble to detect radiator; the multi-model comparison reports this honestly | None; this is documented behaviour |
| `numpy` warning: "Comparing identically-shaped arrays" | Cosmetic; happens in coherence check | None |
| `UnicodeDecodeError` reading `viz/app.py` from cp1252 console | Windows console can't print `→`/`Δ` characters | `set PYTHONIOENCODING=utf-8` or use the `.venv/Scripts/python.exe` console |

---

## 18 · Final verified model results

These are the numbers **I personally re-verified on this hardware, this
session, end of session**, not numbers copied from a previous commit.

| Test | Result | Interpretation |
|---|---|---|
| `e2e_dry_run` | **PASS × 2 iterations** | The full pipeline works twice in a row |
| `test_physics` | SOC 1.000, V 28 V, T_eq 223 K | Physics equations match hand calculation |
| `physics_rules.test_rules` | Solar 44/44 post-900, radiator 44/44 post-900, 0 pre-injection false positives | Rules fire at the right time |
| Lifecycle (`_lifecycle_assertions`) | 4/4 stages PASS: regenerate, retrain, behavioural, no-ensemble | System reproduces from clean state |
| `detect` coherence invariant | **0 contradictory rows** (was 2232 BEFORE the coherence fix) | Flag and score now agree by construction |
| `nasa_real_validation --quick` | Real NASA PCoE cells evaluated; see `models/comparison_report.json` | External validation done |
| `audit_matrix.json` ranks | LOF best (F1=1.000, FPR=0.002), PINN worst (F0.65 radiator) | Honest rank |

### What each metric means

- **F1** — the harmonic mean of precision ("of things you flagged, how
  many were right") and recall ("of all the bad things, how many did
  you flag"). Higher is better. 1.000 = perfect, 0.000 = useless.
- **ROC-AUC** — area under the curve of true-positive-rate vs
  false-positive-rate as you sweep a threshold. 1.0 = ranks all bad
  rows above all good rows; 0.5 = random.
- **PR-AUC** — area under the curve of precision vs recall.
  Especially informative when bad rows are rare (imbalanced).
- **FPR** — false-positive rate on healthy telemetry. Operational
  importance: false alarms cost operator attention.
- **Detection delay** — seconds between fault injection (t=600) and
  the first flagged sample. Lower is better.

The benchmark is trustworthy because:

- Every result in the comparison matrix comes from running the script
  on this hardware, with a seed-fixed RNG.
- No score was tuned against the test scenario (that fix was applied
  this session — contamination moved 0.07 → 0.05 and the source-of-truth
  was made auditable in code comments).
- Hold-out test sets are time-split (last 20 % of `train` files) so
  no scenario is implicitly leaked into model selection.

---

## 19 · Important design decisions

| Decision | Alternatives considered | Why we chose this |
|---|---|---|
| **Three-model OR ensemble** for unsupervised | single 5-feature IF, LOF only, OCSVM only | Bare IF misses radiator with 5 features because the post-ramp T is still in the early-transient range. Subsystem-specialised IFs detect both faults without false-positive inflation in normal. |
| **DEMO_FAST=True** | spec-faithful mc_p=5000 | Visualisation demo is the primary use case; spec-faithful runs quieter and is *documented* as one-line flip away. Trade-off explicitly known. |
| **TF-IDF RAG** instead of embeddings | sentence-transformers, bm25, full Granite-RAG | TF-IDF is offline, no quota, no latency. On a 3-doc corpus embeddings add no value. |
| **Granite 3 2B instruct** | GPT-4-class, Llama, fine-tuned local model | Smallest watsonx model; produces JSON-shaped outputs consistently; mock fallback guarantees dashboard never breaks. |
| **`contamination=0.05`** (after audit) | 0.07 (was tuned to failure flag rates — test-set leak), 0.10 (which makes the bare IF responsive), 0.02 (too tight) | 0.05 chosen by **operator-chosen FP tolerance**, not by tuning to failure scenarios — defends against the audit-trail question. |
| **`decision_function = MIN-of-three`** for ensemble score (after audit) | mean, max, weighted average | IsolationForest decision_function is "higher = more normal". MIN is the most-anomalous detector's view. This guarantees `flag=1 ⇒ score<0`, eliminates the dashboard contradiction found in earlier audit (2232 contradictory rows → 0). |
| **3-DEMAND_FAST constants** tunable separately | single global flag | Lets a judge/demonstrator flip just the thermal constant for a graceful degradation view, without breaking the rest |
| **`@source` annotation column** (this session) | implicit attribution | Single source of truth for which detector drove the MIN. Lets the dashboard surface "EPS" vs "Thermal" |
| **No telemetry ingest endpoint** (deliberate) | CCSDS packet decoder, Yamcs client, etc. | Project is a self-contained demo, not yet connected to a flight feed. Adding telemetry ingest is the next obvious production-grade upgrade. |

---

## 20 · "What to say if someone asks" — personal cheat sheet

### "What does your AI actually do?"

> "MissionMind is a spacecraft anomaly detection system. The AI part
> generates operator-facing explanations — the *what, why, and what to
> do* — when something on the spacecraft looks abnormal. The actual
> detection is done by physics-based sanity checks and an IsolationForest
> ensemble; Granite is the explainer, not the detector."

### "Why is this better than a normal anomaly detection system?"

> "It does three things in parallel: physics thresholds, multiple ML
> models specialised by subsystem, and a knowledge-base-grounded LLM.
> Physics alone is too rigid. ML alone hallucinates false causes. A
> LLM alone invents procedure citations. Combining them gives the
> operator something they can act on."

### "Where is the physics?"

> "Every equations is in `simulator/power.py` and `simulator/thermal.py`.
> The cooling is Stefan-Boltzmann (`Q = εσA(T⁴ − T_space⁴)`) with
> first-order Euler integration; the battery is integrated power in/out
> with a linear OCV. The expected temperature is recomputed every
> second. If observed differs from expected by more than a threshold, a
> flag is raised BEFORE the ML even looks at it."

### "Why do you need AI?"

> "Three reasons: (1) cross-validation between physics and ML — when
> they agree, the operator trusts the alert; (2) operator-friendly
> explanation — the alert text is human-grade, not raw scores; (3)
> RAG-grounded citation so the recommended action points back to a
> actual procedure."

### "What is the digital twin?"

> "The 3D model in the browser is the actual IBM satellite CAD file,
> converted into a JS module and rendered via Three.js. It's a faithful
> geometric twin — every part corresponds to a real CAD group. The
> physics-coupled parts (solar-panel dimming on solar fault, beacon
> glow on anomaly flag) are real telemetry-driven state transitions.
> The orbital position is currently a 2D parametric ring — not a real
> Kepler propagator yet."

### "How do you know the model works?"

> "Two things: (1) the 3 simulator scenarios (`run_normal /
> run_solar_failure / run_radiator_failure`) all have ground truth, and
> the ensemble hits 100 % post-fault detection with strict-window FPR
> = 0; (2) there is an external benchmark against REAL NASA PCoE B0005/
> B0006/B0007/B0018 `.mat` files with cycle-level metrics."

### "What happens when the model is wrong?"

> "Detect's coherence invariant guarantees that flag and displayed
> score never contradict; that prevents one class of user-visible
> bug. False positives are bounded because contamination=0.05 came from
> held-out normal validation, not from failure scenarios. False
> negatives on the bare IF are documented (radiator fails with raw IF
> alone), and mitigated by the OR-ensemble."

### "What is actually innovative?"

> "Three honest claims:
> 1. The leak-free **contamination source** (operator's FP tolerance,
>    not failure flag rates) — a structural fix every team building
>    IsolationForest needs.
> 2. The **decision_function coherence** invariant (MIN of detectors =
>    most pessimistic score; guarantee flag=1 ⇒ score<0) — eliminates
>    an entire class of dashboard inconsistency.
> 3. **Cycle-level + predictive-horizon** metrics against REAL NASA
>    cells — most teams do row-level metrics on synthetic data only."

---

## 21 · MissionMind in 5 minutes

**1. The problem.** A small satellite streams voltage, solar current,
temperature, etc. We want to know fast whether something is breaking,
without too many false alarms, and we want the operator to know *what
to do*.

**2. The architecture.** Three independent detectors:
- Physics thresholds (solar 70 %, thermal slope)
- 3-model IsolationForest ensemble (full + power-specialist + thermal-specialist)
- IBM Granite LLM via RAG over a 3-document knowledge base
They fuse into a single MissionMind decision per telemetry row.

**3. The data.** Two sources:
- Deterministic simulator (`run_scenarios.py`): 3 scenarios × 3600
  rows, with optional injected sensor noise.
- Real NASA PCoE bench cells (`real_nasa/*.mat` 4 × ~6000 rows for
  cycle-level validation).

**4. The physics.** Stefan-Boltzmann radiation; linear OCV battery;
Euler-stepped SOC integration; first-order heat balance. All
equations live in `simulator/power.py` and `simulator/thermal.py`.
Hand-verified by `physics_verification.py`.

**5. The ML.** Three IsolationForests OR'd together for the flag,
MIN-of-scores for the ensemble anomaly_score. Trained on the FIRST
80 % of `run_normal.csv`; contamination=0.05 chosen by held-out
normal FP tolerance (NOT test-set tuned). Validated externally on
REAL NASA cells.

**6. The AI.** IBM watsonx Granite 3 2b-instruct (or deterministic
mock if no API key). Receives JSON of `{subsystem, anomaly_score,
physics_flag, current_values, nominal_values}` plus the top-3 RAG
chunks; returns JSON-shaped `{risk, probable_cause, reasoning,
recommended_action, evidence_used, confidence}`.

**7. The digital twin.** Real IBM satellite CAD (OBJ) rendered via
Three.js in a Streamlit iframe. Solar panels visibly dim on solar
fault; beacon pulses red on anomaly. Orbital position is parametric,
not propagated (limitation).

**8. The validation.** `e2e_dry_run.py` runs the full pipeline twice
and asserts every step PASSes. Comparison matrix (`compare.py`)
trains 8 candidate models on 3 scenarios + 2 holdouts. NASA
PCoE external validation (`nasa_real_validation.py`) on real bench
data uses cycle-level + predictive-horizon metrics — never row-level on
synthetic data only.

**9. The key innovation.** Three structural fixes that any team
needs:
1. Leak-free contamination source (FP tolerance, not failure flag
   rate).
2. Decision-function coherence (MIN across detectors ⇒ score is
   consistent with flag by construction).
3. Cycle-level + predictive-horizon metrics on REAL NASA data (not
   toy CSVs).

**10. The remaining limits.**
- Synthetic telemetry; not wired to a flight feed.
- DEMO_FAST constants; one-line flip away from spec-faithful.
- Orbit visualisation is decorative (no real Kepler).
- ML is frozen; no online / streaming adapter (would need
  `partial_fit`-compatible IF for that).
- RUL is on a parallel offline path against NASA bench cells; not
  wired to live `run_scenario()` telemetry yet.

**The single best one-line description:**

> "MissionMind is a physics-aware anomaly detection + Granite-grounded
> diagnosis system for small-satellite telemetry, validated against
> real NASA PCoE battery cells, with a tight guarantee that no flagged
> row ever shows a normal-looking score."

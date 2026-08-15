<div align="center">

# 🛰️ MissionMind

### AI for spacecraft reliability: detect faults, explain causes, estimate remaining life.

[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)]()
[![Three.js](https://img.shields.io/badge/3D-Three.js-black.svg)]()
[![watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-1F70C1.svg)]()
[![NASA PCoE](https://img.shields.io/badge/validation-NASA%20PCoE%20B0005%2F6%2F7%2F18-orange.svg)]()
[![tests](https://img.shields.io/badge/tests-19%20suites%20PASS-brightgreen.svg)]()

</div>

MissionMind is a satellite mission-operations stack that starts from physics and ends at an operator-facing explanation. A simulator generates one hour of power and thermal telemetry, an ensemble of unsupervised anomaly detectors scores every sample, physics rules act as an independent second opinion, and a Granite-backed RAG layer writes the reasoning in plain language. The same pipeline feeds a Streamlit dashboard with a 3D satellite view and a React console. Validation runs against the real NASA PCoE battery dataset (B0005 / B0006 / B0007 / B0018).

---

## Demo

Thirty seconds of the dashboard: scrubbing the mission clock through a solar-array failure, watching telemetry degrade, the RUL chip count down, and the 3D spacecraft react on the same simulated clock.

<p align="center">
  <img src="demo.gif" alt="MissionMind dashboard scrubbing a 1-hour solar-failure mission" width="860">
</p>

Captured live from the running app. `demo.gif` regenerates from `streamlit run missionmind/viz/app.py` using the Mission Time Transport bar.

---

## What's in it

Most spacecraft-AI demos showcase a single technique. This repo wires several together, and each layer is independently checkable:

| Layer | What it does | Notes |
|---|---|---|
| **Physics simulator** | Coupled power + thermal ODE solver; configurable fault injection | `mc_p = 5000 J/K`, `r_final = 0.30` toggleable via `MISSIONMIND_PHYSICS_SPEC=1` |
| **ML ensemble** | Isolation Forest, LOF, OC-SVM, MLP-AE, Hybrid DIF, FCNN, XGBOD | Held-out contamination of 0.05 |
| **RAG** | TF-IDF over a hand-curated markdown knowledge base | Top-k docs pulled live per anomaly, each chunk citeable |
| **Granite (`ibm/granite-3-2b-instruct`)** | watsonx.ai JSON output with cited evidence | Falls back to a deterministic mock when `WATSONX_APIKEY` is unset; the sidebar shows which mode is active |
| **Streamlit + Three.js** | Mission-control dashboard with a real IBM satellite CAD | Time scrubber, RUL chip, 4-line causal alert, live physics + ML + RAG |

The full pipeline is reproducible from one command:

```bash
python missionmind/e2e_dry_run.py
```

---

## Quickstart

```bash
# 1. clone + virtualenv
git clone https://github.com/ojasvigoel598/IBM-spacecraft.git
cd IBM-spacecraft
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. install (includes pyod+xgboost for XGBOD, torch for the PINN twin, gmsh for CAD export)
pip install -r requirements.txt

# 3. generate telemetry and train the ensemble (writes models/*.joblib)
python -m missionmind.simulator.run_scenarios     # 3 deterministic CSVs
python -m missionmind.ml.train                    # 2880-row hold-out split, seed 42

# 4. run the dashboard
streamlit run missionmind/viz/app.py                  # Streamlit + Three.js 3D viewer (port 8501)

# 5. optional: real Granite on watsonx.ai (everything runs on the mock without this)
cp .env.example .env                                   # then fill in WATSONX_APIKEY + WATSONX_PROJECT_ID
python -m missionmind.ai.granite_client --check        # "CHECK PASS" = key works; restart the dashboard after

# web mission-control console (React + shadcn/ui + Tailwind)
python -m uvicorn missionmind.viz.api_server:app --port 8100   # JSON API on the same pipeline
cd web && npm install && npm run dev -- --port 5173            # console at http://localhost:5173
```

Open `http://localhost:8501`. The 3D spacecraft CAD loads inside the dashboard; pick a scenario in the sidebar (Normal / Solar-Array Degradation / Radiator Degradation) and scrub the Mission Time Transport bar to jump anywhere in the 1-hour mission. The web console at `http://localhost:5173` is a lighter React front-end over the same scored telemetry: scenario switch, live scrubber, anomaly evidence, and a Live Ingest tab streaming a virtual ESP32 edge node through the production ensemble.

### Environmental knobs

| Variable | Effect |
|---|---|
| `MISSIONMIND_PHYSICS_SPEC=1` | Swaps `mc_p=2000 J/K -> 5000 J/K` and `r_final=0.10 -> 0.30` per spec §3-§5. Default off for demo speed. |
| `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` | Real `ibm/granite-3-2b-instruct` call instead of the deterministic mock fallback. |
| `WATSONX_MODEL_ID` | Defaults to `ibm/granite-3-2b-instruct`; any model ID on watsonx.ai works. |

---

## Architecture

One telemetry sample through the whole stack:

```text
       Mission Time T+0:00 --------------------------------------------------> T+01:00
                                  |                |                  |
                -------------------+----------------+-----------------+
                v run_scenario()   v solver (RK4-equivalent Euler)    v physics rules
       power load = 400 W     dSOC/dt = (net) / 3600 / 100        solar_drop : P < 364 ?
       solar = 520 W (env)     dT/dt   = (Q_in - Q_out) / mc_p     soc_slope : dSOC/dt < -0.0002
       bus V = 28 V @ SOC=1    eps_A    = e * A                   temp_slope: dT/dt > 0.003
                                  |                |                  |
                -------------------+----------------+-----------------+
                v add_derivative_features(df)   score_dataframe()    v physics_rules.*
       d_temp_dt, d_volt_dt    full detector (IsolationForest)      check_power_subsystem()
                                power detector (LOF on EPS feats)   check_thermal_subsystem()
                                thermal detector (AE on thermal)
                                ensemble = OR, score = MIN
                                  |                |                  |
                -------------------+----------------+-----------------+
                v rag.query_from_anomaly()     granite_explain()    v viz.app.py 4-line alert
       top-k docs + scores      json{diagnosis, evidence, action}   WARN - SUBSYSTEM - EVIDENCE - ACTION
       thermal_subsystem.md               citations
       power_subsystem.md
       mission_rules.md
                                  |
                -------------------+----------------+-----------------
                v viz Streamlit       v Three.js (satellite_geometry.js)  v prognosis chip
       KPIs, time-scrubber      real IBM CAD, part-level animation    "BAT 240 m / THM inf"
       t-transport, alert chip  solar arrays dim/pulse on PV fault
                                main bus glows on radiator fault
```

---

## Architecture decision records

Three decisions that shaped the system, each with the on-disk evidence that justified it.

### ADR-001: PGNN over strict PINN for production RUL/degradation scoring

**Status:** Accepted (evidence-based) · **Date:** 2026-08

**Context.** The PINN candidate was expected to win because it is physics-informed; the audit question was whether the marginal effort actually beats a feature-only model on real NASA B0005 data. Both were evaluated on the same Arm-D protocol (one cycle per row, `min(AUC,|Spearman|)` selection metric).

**Decision.** The feature-only PGNN (healthy-envelope physics gates, no physics loss term) is the production model. The strict Raissi-2019 PINN stays in the zoo as a documented research artifact, and the "(Best)" label was dropped from it. A torch-autograd twin (`pinn_torch.py`) was built to rule out finite-difference artifacts; it changed the trade-off (better |Spearman|, worse AUC) but did not overturn the verdict.

**Evidence.** Head-to-head runner: `missionmind/ml/pinn_vs_pgnn.py` (reference config at `pinn_vs_pgnn.py:42-45`, table and verdict at `pinn_vs_pgnn.py:175-200`). Stored result: `missionmind/models/pinn_vs_pgnn_b0005.json`: **PGNN `min(AUC,|Sp|) = 0.789` (AUC 0.789, Spearman -0.94) vs best strict PINN `0.285`** across the λ sweep. Label rationale: `missionmind/ml/advanced_models.py:422-427`. Multi-seed sweep that selected the PGNN config: `missionmind/ml/pinn_layer_scan.py:58`. Torch twin: `missionmind/ml/pinn_torch.py:3-14`.

### ADR-002: `MISSIONMIND_PHYSICS_SPEC=1` toggle instead of edit-and-ship

**Status:** Accepted · **Date:** 2026-08

**Context.** The spec calls for `mc_p = 5000 J/K` and a 30 % radiator end-state; those values make the simulated faults unfold too slowly to demo in an hour. Earlier versions hardcoded the demo values into the source, which was both a demo-rigging hazard and a maintenance trap (silent spec drift).

**Decision.** All physics constants live in one central module with explicit `*_SPEC` / `*_DEMO` pairs; `DEMO_FAST` defaults on and is switched by the `MISSIONMIND_PHYSICS_SPEC=1` environment variable, so no source edit is required to run the true spec. Every downstream module binds the constants at import so the toggle propagates through the whole pipeline, and a test suite asserts those bindings so a silent divergence fails loudly.

**Evidence.** Central constants + env read: `missionmind/simulator/config.py:33-39` (thermal mass) and `config.py:56-58` (radiator fraction). Binding + loud-failure tests: `missionmind/tests/test_config_seam.py:31-70` (bindings), `test_config_seam.py:73` (`test_config_failure_is_loud_not_silent`). Env knob documented in this README under Environmental knobs. One intentional deviation: tuned physics-rule thresholds, because the spec thresholds are inconsistent with the spec's own failure magnitudes, documented at `missionmind/physics_rules/rules.py` (`check_power_subsystem` / `check_thermal_subsystem`).

### ADR-003: Mock-first Granite fallback instead of hard-fail

**Status:** Accepted · **Date:** 2026-08

**Context.** The explanation layer calls `ibm/granite-3-2b-instruct` on watsonx.ai. Credentials are not guaranteed in a demo or CI environment, and a hard-fail would break the entire dashboard whenever the API is unreachable. The system must stay operational without the LLM, and must never present mock output as real.

**Decision.** `generate_explanation()` tries the real watsonx call only when `WATSONX_AVAILABLE` and both credentials are present; on any failure it falls back to a deterministic rule-based mock that always returns schema-valid JSON. The UI surfaces the fallback honestly (sidebar shows `API Key: missing (mock fallback)`).

**Evidence.** Fallback policy: `missionmind/ai/granite_client.py:6-7` (docstring), `granite_client.py:26-28` (`WATSONX_AVAILABLE`), `granite_client.py:31-33` (deterministic mock), `granite_client.py:178` (real-call gate), `granite_client.py:202-204` (failure to mock, always valid). Behavior tests: `missionmind/tests/test_granite_nominal.py:53-87` (nominal/solar branches and the `ml_flag` guard). UI honesty contract: see Granite grounding below.

---

## Visual tour

Captured from the running Streamlit dashboard at 1600x1200. Each image shows a real MissionMind control state.

### Mission-control overview

![MissionMind mission-control overview](screenshots/overview.png)

### 3D spacecraft (Three.js, real IBM satellite CAD)

![MissionMind Three.js spacecraft scene](screenshots/threejs-scene.png)

### Solar-array degradation

![MissionMind solar-failure alert](screenshots/solar-failure.png)

### Radiator degradation

![MissionMind radiator-failure alert](screenshots/radiator-failure.png)

### RAG alert with engineering citations

![MissionMind RAG alert with citations](screenshots/rag-alert-citations.png)

---

## Validation evidence

Every number below is reproduced from the code on disk.

### 1. Real NASA PCoE battery benchmark (Arm-D protocol on B0005)

```
PGNN (64,32,16) reground a=0.30   (multi-seed robust)        AUC = 0.786 +/- 0.009 (6 seeds)
  Spearman rho  = 0.950 +/- 0.028  (sign-correct)
  Win-count     = 2 / 6 seeds among top-5 configs
Source: missionmind/ml/pinn_seed_robustness.py
```

### 2. Simulator fault-injection (run_normal / run_solar_failure / run_radiator_failure)

```
solar failure          FPR_strict 100-600s   = 0.000
                       flag_rate 900-3600s    = 1.000
                       post900_F1              ~ 1.00
radiator failure       FPR_strict 100-600s   = 0.000
                       flag_rate 900-3600s    = 1.000
                       post900_F1              ~ 1.00
normal                 FPR_strict 100-600s   = 0.000
                       flag_rate 900-3600s    = 0.093   (8 % burn-in drift, expected)
Source:  missionmind/ml/detect.py + missionmind/_lifecycle_assertions.py
```

### 3. Physics rule checks under load

```
test_rules     PASS all rule tests
test_physics   PASS SOC plateau 1.0, V 28 V, T-eq 223 K
test_ml_metrics  PASS cycle-level + predictive-horizon + fresh-capacity guard
Source: missionmind/tests/test_physics.py
        missionmind/physics_rules/test_rules.py
```

### 4. Raissi 2019 strict PINN: honest non-result

The repo includes a strict physics-informed NN (data loss + λ-weighted physics residual `(dC/dn_NN - (-a*C))^2`) benchmarked against the feature-only PGNN on the same B0005 scan:

```
PGNN        AUC 0.789   |Sp| 0.939   min(AUC,|Sp|)= 0.789
PINN(l=0.5) AUC 0.349   |Sp| 0.227   min(AUC,|Sp|)= 0.227
```

The strict PINN does not beat the feature-only PGNN on this benchmark; the composite loss collapses discrimination once the network fits the ODE. The production model is the feature-only PGNN, and the PINN is labeled as a research artifact, not as "best". Result on disk: `missionmind/models/pinn_vs_pgnn_b0005.json`.

A PyTorch-autograd twin of the PINN (`missionmind/ml/pinn_torch.py`) uses the same composite loss with `dC/dn_NN` from `torch.autograd.grad` (real backpropagation, Adam + cosine schedule) instead of finite differences. On the same B0005 protocol it changes the AUC/|Spearman| trade-off (higher |Sp|, lower AUC) but does not overturn the verdict, so it stays an experimental drop-in rather than a replacement.

---

## CAD assets

The spacecraft in the 3D viewer is a real Fusion-exported IBM satellite (not a procedural placeholder). All three exchange formats ship in the repo so visitors can view or download it without the code:

| Format | File | Use |
|---|---|---|
| **OBJ** | [`missionmind/viz/components/models/ibm_satellite.obj`](missionmind/viz/components/models/ibm_satellite.obj) | Source mesh (42,878 verts / 85,740 tris), what Three.js renders |
| **STL** | [`missionmind/viz/components/models/ibm_satellite.stl`](missionmind/viz/components/models/ibm_satellite.stl) | Binary STL; GitHub renders this inline in the browser (click to orbit/zoom) |
| **STEP** | [`missionmind/viz/components/models/ibm_satellite.step`](missionmind/viz/components/models/ibm_satellite.step) | AP203 faceted BRep; opens in Fusion / SolidWorks / FreeCAD |

> The STL and STEP are generated from the OBJ by `missionmind/viz/components/obj_to_step_stl.py` (gmsh mesh kernel + hand-rolled ISO-10303-21 writer; watertightness and winding are verified before export).

---

## Granite grounding

Granite is not used as an anomaly detector. The ML ensemble + physics rules are deterministic and frozen at every cursor position. Granite is invoked only for the human-readable explanation:

1. The retriever sees the current anomaly row plus the physics-rule hits.
2. It returns the top-k markdown chunks from `missionmind/ai/knowledge_base/*.md`, each scored by TF-IDF.
3. The system prompt declares the contract: output JSON with `diagnosis, evidence, recommended_action, risk` and `citations` arrays linking back to the chunks.
4. `ibm/granite-3-2b-instruct` fills that contract; the dashboard parses and displays the result.

If `WATSONX_APIKEY` is not set, the sidebar shows `API Key: missing (mock fallback)` and the function uses a deterministic mock that returns the same JSON shape with the RAG chunks surfaced as citations, so it is always obvious whether the output came from a real LLM call or the mock.

---

## What this is (and isn't)

Implemented:

- **Energy-conserving ODE**: `dSOC/dt = (P_solar - P_load) / (E_cap_Wh * 3600)`; `dT/dt = (Q_in - Q_out) / mc_p`, with `mc_p` and `r_final` toggled per spec on demand.
- **Fault injection**: solar degradation ramps `P_solar_max -> 0.48x` over `t = 600 -> 900 s`; radiator degradation ramps `e*A -> r_final * nominal`.
- **Real IBM satellite CAD** (`satellite_geometry.js`): part-level fault animation, solar arrays dim + pulse on PV failure, main bus glows on radiator failure.
- **Detector / physics-rule co-design**: every flagged row is visible in the alert card with the exact feature column that drove the flag.

Not implemented:

- **No real Kepler propagator** in `applyOrbit()` yet; the orbit ring is decorative. Tracked, not blocking.
- **Live ingest is real but virtual**: a simulated ESP32-class edge node publishes the same physics over a real JSON-lines TCP socket (MQTT when paho-mqtt is installed); the ensemble scores the stream as it arrives. A physical ESP32/RPi can replace the virtual node with the same wire format (`missionmind/telemetry/`).
- **The strict PINN does not beat the feature-only PGNN**; this is documented as a result, not hidden.

---

## Project map

```
missionmind/
|-- ai/                     IBM watsonx.ai client + RAG retriever + prompt templates
|   |-- granite_client.py   real + mock wrappers
|   |-- rag.py              TF-IDF over a 3-file knowledge base
|   +-- prompts.py          SYSTEM / RAG / EVIDENCE prompt contracts
|
|-- simulator/              coupled power + thermal ODE solver, fault injection
|   |-- config.py           central constants + DEMO/SPEC flag toggleable via env
|   |-- thermal.py          dT/dt = (Q_in - Q_out) / mc_p  (RK4-equivalent Euler)
|   |-- failures.py         solar + radiator fault ramps
|   |-- run_scenarios.py    @ 1 Hz over 3600 s -> DataFrame
|   +-- physics_verification.py    manual hand-calc vs sim
|
|-- telemetry/              the "electronics side" of the twin: live ingest
|   |-- edge_node.py        virtual ESP32-class device -> physics-derived frames
|   |-- ingest.py           JSON-lines TCP server/client + MQTT (paho) + LiveScorer
|   |-- frame.py            wire schema (identical to run_scenarios output)
|   +-- run_edge_demo.py    CLI demo: stream 1000 frames through the ensemble
|
|-- physics_rules/          Spec section 6 rule checks (solar_drop + soc_slope, temp_rising)
|   +-- rules.py            check_power_subsystem, check_thermal_subsystem
|
|-- ml/                     detector zoo + scoring + audit infrastructure
|   |-- train.py            ensemble (full + power + thermal) -> 3 .joblibs
|   |-- detect.py           score_dataframe(): UNSUP + ENSEMBLE, MIN-of-3 score + source attribution
|   |-- advanced_models.py  IF / LOF / OC-SVM / MLP-AE / Hybrid DIF / FCNN / XGBOD / PINN
|   |-- metrics.py          9 metrics + confusion-matrix helpers
|   |-- prognostics.py      NASA RUL on the real .mat files (offline)
|   |-- pinn_*.py           Raissi strict PINN + torch-autograd twin + multi-seed sweep + head-to-head
|   +-- compare.py          threshold-independent + dependent metrics table
|
|-- tests/                  19 TDD suites, all pass
|   |-- test_drift.py       KS drift test
|   |-- test_ml_metrics.py  predictive-horizon, fresh-capacity, cycle-level
|   |-- test_physics.py     SOC plateau, V, T-eq
|   +-- *.py
|
+-- viz/                    Streamlit dashboard + Three.js mission-control viewer
    |-- app.py              single-file dashboard
    |-- api_server.py       FastAPI JSON API: same scored scenarios + live edge-node stream + model zoo
    +-- components/         satellite_geometry.js + models/{ibm_satellite.obj,.stl,.step} + obj_to_step_stl.py

web/                        React 19 + Vite + Tailwind v4 + shadcn/ui mission-control console
    |-- src/App.tsx         KPI grid, SVG telemetry charts, time scrubber, alert evidence, live ingest
    +-- src/components/ui/  shadcn/ui components (Base UI flavour)
```

---

## Testing

```bash
# 19 regression suites (TDD) - run them all with pytest:
pytest

# or one at a time, e.g.:
python -m missionmind.tests.test_physics
python -m missionmind.tests.test_ml_metrics
python -m missionmind.tests.test_granite_nominal
python -m missionmind.tests.test_config_seam
python -m missionmind.tests.test_prognostics
python -m missionmind.tests.test_drift
python -m missionmind.tests.test_mlpae_tighten
python -m missionmind.tests.test_pinn_raissi
python -m missionmind.tests.test_telemetry_ingest
python -m missionmind.tests.test_api_server
python -m missionmind.physics_rules.test_rules
```

End-to-end run with the project interpreter:

```bash
python missionmind/e2e_dry_run.py
# regenerates scenarios, retrains the ensemble, runs all ML detectors,
# walks the RAG, exercises the Granite client (real + mock), validates the dashboards.
```

---

## IBM watsonx.ai integration

| What | Where |
|---|---|
| Model ID | `ibm/granite-3-2b-instruct` |
| Auth | `WATSONX_APIKEY` (env) + `WATSONX_PROJECT_ID` (env) |
| Code | `missionmind/ai/granite_client.py`: `_call_watsonx_granite()` is the only real-network path |
| Mock | `generate_explanation(...)` falls back to a deterministic mock that still returns schema-correct JSON with RAG citations; the sidebar shows which mode is active |
| Verify a key | `python -m missionmind.ai.granite_client --check` reports config and makes a real call when a key is set |
| RAG corpus | `missionmind/ai/knowledge_base/{power_subsystem, thermal_subsystem, mission_rules}.md` |
| Citations | Granite is asked to fill a `citations[]` array linking each claim to a TF-IDF chunk, so each dashboard claim is traceable to a file |

---

## Status (2026-08)

- Real NASA PCoE battery benchmark wired (Arm-D protocol, multi-seed robust)
- Ensemble coherence under inspector (flag=1 implies score<0 by construction)
- 4-line causal narrative (WARN -> SUBSYSTEM -> EVIDENCE -> ACTION)
- Granite client + RAG integrated with honest mock fallback
- Real Kepler propagator: not yet (decorative orbit only)
- Live telemetry ingest: virtual IoT edge node -> TCP/MQTT -> live ensemble scoring (`missionmind/telemetry/`)
- Session persistence: dashboard resumes at last-viewed scenario/time after restart, plus login auto-start (`scripts/install_autostart.ps1`)
- PINN vs PGNN result disclosed on disk and in this README

---

## Contributing

PRs welcome. The two highest-leverage follow-ups:

1. Real LEO Kepler propagator + visible eclipse in `viz/app.js` `applyOrbit()`.
2. Physical edge hardware: an ESP32/RPi running the same JSON-lines wire format the virtual node publishes, replacing `VirtualEdgeNode` in `run_edge_demo.py`.

Both are tracked in `CHANGES.md` with risk / impact / effort annotations.

## License

MIT. See [LICENSE.md](LICENSE.md).

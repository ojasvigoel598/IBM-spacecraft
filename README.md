<div align="center">

# 🛰️ MissionMind

**AI-powered spacecraft fault detection — 7 seconds from onset to alert.**

[![IBM Bob Certificate](https://img.shields.io/badge/IBM%20Bob-Certificate-1F70C1.svg)](https://skills.yourlearning.ibm.com/certificate/share/99e8a93d06ewogICJvYmplY3RJZCIgOiAiQUxNLUNPVVJTRV80MDc2MzExIiwKICAibGVhcm5lckNOVU0iIDogIjg0NTM0MzFSRUciLAogICJvYmplY3RUeXBlIiA6ICJBQ1RJVklUWSIKfQ1ee785e3df-10)
[![IBM Certificate](https://img.shields.io/badge/IBM-SkillsBuild%20Certificate-1F70C1.svg)](https://skills.yourlearning.ibm.com/certificate/share/3dfa573d92ewogICJvYmplY3RUeXBlIiA6ICJBQ1RJVklUWSIsCiAgImxlYXJuZXJDTlVNIiA6ICI4NDUzNDMxUkVHIiwKICAib2JqZWN0SWQiIDogIkFMTS1DT1VSU0VfNDA3NjMxMSIKfQ2778ae28b9-10)
[![IBM](https://img.shields.io/badge/IBM-watsonx.ai%20Granite-1F70C1.svg)]()
[![NASA](https://img.shields.io/badge/NASA%20PCoE-B0005%20Validated-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-30%20suites%20PASS-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

</div>

Spacecraft faults happen at 3 AM. The operator has minutes to decide. MissionMind gives them the answer in seven seconds.

---

## The Problem

- Spacecraft anomalies (solar array degradation, radiator failure) are caught by threshold alarms **3 minutes after onset** — too late to prevent cascading failures.
- Eclipse shadows cause **false alarms every orbital pass**, training operators to ignore alerts.
- Existing fault-detection systems flag anomalies but don't explain **why** or **what to do**.
- A $100M satellite has ~39 minutes between first fault and bus shutdown. Every second counts.

## The Solution

MissionMind is a spacecraft digital twin that couples a physics simulator, an ML anomaly detector, and an IBM Granite-powered explanation layer.

**Key features:**
- **7-second detection** — ML ensemble detects faults within seconds of onset, 25× faster than threshold alarms.
- **Zero false alarms during eclipse** — Kepler propagator predicts orbital shadows; normal dips are suppressed.
- **4-line causal alert** — every anomaly produces: `WARN → SUBSYSTEM → EVIDENCE → ACTION`.
- **IBM Granite explanations** — cited, structured JSON diagnoses from watsonx.ai, not unstructured chatbot text.

On real NASA battery data, the system achieves **AUC 0.786** validated across 6 seeds.

---

## How It Works

1. **Simulate.** A physics engine generates one hour of power and thermal telemetry from a virtual satellite.
2. **Propagate.** A Kepler propagator calculates where the satellite is in orbit and whether it is in sunlight or eclipse.
3. **Detect.** An ensemble of unsupervised ML models (Isolation Forest, LOF, OC-SVM, and more) scores every sample for anomalies.
4. **Validate.** Physics rules check whether the anomaly is real or just an eclipse shadow — false alarms drop to zero.
5. **Retrieve.** A RAG retriever pulls relevant engineering documents from a curated knowledge base.
6. **Explain.** IBM Granite on watsonx.ai formats the diagnosis, evidence, and recommended action as structured JSON.
7. **Display.** The operator sees a 4-line alert card, a countdown of remaining useful life, and a 3D view of the satellite.

---

## Demo

Watch the 2-minute narrated walkthrough: https://youtu.be/wg2fR0hICrs

<p align="center">
  <video controls width="860" style="border-radius:8px;" src="demo/final_demo.mp4"></video>
</p>

---

## Try It Locally

No API keys required. The dashboard works immediately.

```bash
git clone https://github.com/ojasvigoel598/IBM-spacecraft.git && cd IBM-spacecraft
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run missionmind/viz/app.py                 # opens at localhost:8501
```

To enable real IBM Granite explanations, add your watsonx.ai credentials:

```bash
cp .env.example .env
# fill in WATSONX_APIKEY and WATSONX_PROJECT_ID
python -m missionmind.ai.granite_client --check      # verifies the connection
```

---

## Architecture

```
  Telemetry Source                 Pipeline                      Output
 ┌──────────────┐    ┌────────────────────────────────────┐    ┌──────────────┐
 │  Simulator   │───▶│  Kepler Propagator (eclipse-aware) │───▶│  ML Ensemble  │
 │  1 Hz power  │    │  Coupled EPS + Thermal ODE solver  │    │  IF / LOF /   │
 │  + thermal   │    │  Reaction-wheel micro-vibration    │    │  OC-SVM / AE  │
 │  3600 frames │    └────────────────────────────────────┘    │  + XGBOD      │
 └──────────────┘              │                               └──────┬───────┘
                               ▼                                      │
                  ┌────────────────────────┐                          │
                  │  Physics Rules Layer   │◀─── cross-check ────────┘
                  │  Eclipse suppression   │
                  │  Threshold validation  │
                  └────────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     RAG Retriever     │
                    │  TF-IDF over 4-file   │
                    │  engineering KB        │
                    └──────────┬───────────┘
                               │
                  ┌────────────▼────────────┐
                  │  IBM Granite-4 (watsonx) │
                  │  JSON diagnosis +        │
                  │  cited evidence          │
                  └────────────┬────────────┘
                               │
              ┌────────────────▼────────────────┐
              │         Operator Dashboard       │
              │  Digital twin CAD (Three.js)      │
              │  4-line causal alert card        │
              │  RUL countdown chip              │
              │  Streamlit + React console       │
              └─────────────────────────────────┘
```

---

## Results

| Metric | Value |
|---|---|
| Detection latency | **7 seconds** after fault onset |
| Warning before shutdown | **39 minutes** (vs 36 min with threshold alarms only) |
| False positives during eclipse | **0** — Kepler geometry suppresses shadow passes |
| NASA PCoE AUC | **0.786 ± 0.009** (6-seed robust, B0005 battery) |
| Test suites | **30 / 30 passing** |

**Before MissionMind:** solar-array degradation triggers threshold alarms ~3 minutes after onset, giving ~36 minutes of warning.

**After MissionMind:** the ML ensemble detects within 7 seconds, providing 39 minutes of advance warning — 3 extra minutes that matter when protecting a $100M satellite.

---

## Technical Details

### Physics Simulator

- Coupled power + thermal ODE solver (Euler integration, RK4 extension point)
- Kepler propagator with conical shadow eclipse model (umbra / penumbra / sun-exposure factor)
- Reaction-wheel micro-vibration model (McMullan power-law disturbance, Arrhenius-Coffin-Manson battery fade)
- Configurable fault injection: solar degradation, radiator degradation
- Physics constants toggleable between spec-faithful and demo-tuned via `MISSIONMIND_PHYSICS_SPEC`

### ML Ensemble

Seven unsupervised detectors trained on normal-only data (contamination 0.05):

| Detector | Type |
|---|---|
| Isolation Forest | Tree-based anomaly scoring |
| Local Outlier Factor | Density-based |
| One-Class SVM | Boundary-based |
| MLP Autoencoder | Reconstruction error |
| Hybrid Deep Isolation Forest | Deep feature + IF |
| FCNN | Supervised feed-forward |
| XGBOD | Extreme boosting outlier |

The ensemble uses OR-logic: any detector flags → anomaly. Score = MIN of three sub-detectors (full, power, thermal).

### RAG + Granite

- TF-IDF retrieval over 4 curated engineering documents (power subsystem, thermal subsystem, mission rules, telemetry reference)
- Metadata-scoped: queries are scoped to the relevant subsystems
- IBM Granite-4 (`ibm/granite-4-h-small`) on watsonx.ai generates cited JSON diagnoses
- Deterministic mock fallback when API key is absent — the UI always shows which mode is active

### Physics Rules

Independent rule engine that cross-checks the ML output:

- Eclipse-aware solar residual (measured vs expected solar during orbital shadow)
- Heat-rejection residual (thermal model vs observed temperature)
- SOC/UVLO policy (safe mode at 20%, bus trip at 0%)

### Digital Twin

- Real Fusion 360 satellite CAD (OBJ / STL / STEP) — 42,878 vertices, 85,740 triangles
- Three.js renders the CAD with part-level fault animation (solar arrays dim, main bus glows)
- Live telemetry ingest via virtual ESP32-class edge node (JSON-lines TCP / MQTT)
- Bidirectional: `send_command()` supports reset, rate change, fault injection

### PINN vs PGNN Finding

The repo includes a strict physics-informed neural network (PINN) benchmarked against the feature-only PGNN on real NASA B0005 data:

```
PGNN (feature-only physics gate)    AUC 0.789   |Spearman| 0.939
strict PINN (Raissi 2019)           AUC 0.349   |Spearman| 0.227
```

The PINN's composite loss collapses discrimination once the network fits the ODE. The production model keeps physics as a gate, not a loss term.

---

## Challenges & Accomplishments

**PINN doesn't work.** The most surprising finding: a strict physics-informed neural network (PINN) — the kind every paper says should win — loses badly to a feature-only model that merely gates on physics. We proved this on real NASA B0005 data (AUC 0.349 vs 0.789) and documented the finding honestly rather than hiding it.

**Eclipse false positives.** Before the Kepler propagator, every orbital shadow pass triggered a false alarm (solar drops to 0W → ML flags "fault"). Solving this required coupling eclipse geometry into the power model, the thermal model, and the physics rules — three independent systems had to agree.

**10+ Vercel build failures.** The React console + FastAPI backend deploy to Vercel as one project. Debugging Node version incompatibilities, output directory resolution, and package.json conflicts across workspaces took significant iteration.

**30 test suites, all passing.** Every component has regression tests: physics numerics, ML metrics, RAG retrieval, auth security, API server, config seams, and more. The test suite runs on every push via GitHub Actions.

---

## Repository Structure

```
IBM-spacecraft/
├── README.md
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Test configuration
├── vercel.json                       # Vercel deploy config
├── package.json                      # Root (Vercel build)
├── MissionMind_Full_ML_Analysis.ipynb  # Jupyter notebook
├── demo/
│   ├── final_demo.mp4                # 2-minute narrated walkthrough
│   └── frames/                       # Demo frame screenshots
│
├── missionmind/
│   ├── simulator/                    # Physics engine
│   │   ├── config.py                 # Central constants + DEMO/SPEC toggle
│   │   ├── power.py                  # EPS: dSOC/dt, battery policy
│   │   ├── thermal.py                # dT/dt = (Q_in - Q_out) / mc_p
│   │   ├── failures.py               # Fault injection ramps
│   │   ├── orbital.py                # Eclipse geometry
│   │   ├── propagation.py            # Kepler + RK4 + DOPRI5 + J2
│   │   ├── vibration.py              # Reaction-wheel micro-vibration
│   │   └── run_scenarios.py          # Generates 3 CSV scenarios
│   │
│   ├── ml/                           # Detector zoo + training
│   │   ├── train.py                  # Ensemble training → .joblib files
│   │   ├── detect.py                 # score_dataframe(): UNSUP + ENSEMBLE
│   │   ├── advanced_models.py        # IF / LOF / OC-SVM / MLP-AE / DIF / FCNN / XGBOD
│   │   ├── metrics.py                # 9 evaluation metrics
│   │   ├── prognostics.py            # NASA RUL on real .mat files
│   │   ├── pinn_vs_pgnn.py           # Head-to-head PINN vs PGNN
│   │   ├── pinn_torch.py             # PyTorch-autograd PINN twin
│   │   ├── compare.py                # Threshold-independent metrics
│   │   └── causal_narrative.py       # 4-line alert generation
│   │
│   ├── ai/                           # IBM Granite + RAG
│   │   ├── granite_client.py         # watsonx.ai SDK + mock fallback
│   │   ├── rag.py                    # TF-IDF retriever over 4-file KB
│   │   ├── prompts.py                # System / RAG / Evidence prompts
│   │   ├── rag_eval.py               # Golden dataset evaluation
│   │   ├── rag_validation.py         # 10-consecutive-clean-runs gate
│   │   └── knowledge_base/           # Engineering documents
│   │       ├── power_subsystem.md
│   │       ├── thermal_subsystem.md
│   │       ├── mission_rules.md
│   │       └── telemetry_reference.md
│   │
│   ├── physics_rules/                # Spec rule checks
│   │   ├── rules.py                  # check_power / check_thermal
│   │   └── test_rules.py
│   │
│   ├── telemetry/                    # Live ingest layer
│   │   ├── edge_node.py              # Virtual ESP32 device
│   │   ├── ingest.py                 # TCP/MQTT server + LiveScorer
│   │   ├── frame.py                  # Wire schema
│   │   └── run_edge_demo.py          # CLI demo
│   │
│   ├── auth/                         # Multi-user authentication
│   │   ├── service.py                # Signup / verify / login / reset
│   │   ├── security.py               # PBKDF2-HMAC-SHA256
│   │   ├── api.py                    # Auth endpoints + rate limiting
│   │   ├── db.py                     # SQLite user/session/token store
│   │   └── ratelimit.py              # Per-IP / per-email limits
│   │
│   ├── viz/                          # Dashboards
│   │   ├── app.py                    # Streamlit + Three.js dashboard
│   │   ├── api_server.py             # FastAPI JSON API
│   │   └── components/
│   │       ├── satellite_geometry.py # trimesh CAD loader
│   │       ├── obj_to_geometry.py    # OBJ → Three.js geometry
│   │       ├── obj_to_step_stl.py    # OBJ → STEP + STL export
│   │       └── models/               # Real IBM satellite CAD
│   │           ├── ibm_satellite.obj
│   │           ├── ibm_satellite.stl
│   │           └── ibm_satellite.step
│   │
│   ├── tests/                        # 30 test suites
│   │   ├── test_physics.py           # SOC plateau, voltage, temperature
│   │   ├── test_ml_metrics.py        # Predictive horizon, fresh capacity
│   │   ├── test_auth.py              # 31 auth tests (brute-force, injection)
│   │   ├── test_granite_nominal.py   # Mock vs real Granite modes
│   │   ├── test_rag_retrieval.py     # Recall@k, MRR, nDCG
│   │   ├── test_pinn_raissi.py       # PINN vs PGNN on B0005
│   │   └── ...                       # 24 more suites
│   │
│   └── models/                       # Trained inference artifacts
│       ├── iforest.joblib            # Isolation Forest
│       ├── iforest_power.joblib      # Power sub-detector
│       ├── iforest_thermal.joblib    # Thermal sub-detector
│       ├── scaler.joblib             # Feature scaler
│       ├── pinn_vs_pgnn_b0005.json   # Head-to-head result
│       └── ...                       # Audit matrices, rankings
│
├── web/                              # React mission-control console
│   ├── src/
│   │   ├── App.tsx                   # KPI grid, charts, scrubber
│   │   ├── auth.tsx                  # Auth context
│   │   └── components/
│   │       ├── auth/AuthScreen.tsx   # Login / signup / verify
│   │       └── ui/                   # shadcn/ui components
│   ├── vite.config.ts
│   └── package.json
│
├── api/                              # Vercel serverless function
│   ├── index.py                      # FastAPI on Vercel
│   └── requirements.txt
│
├── screenshots/
│   ├── overview.png
│   ├── cad-normal.png                # Fusion 360 assembled view
│   ├── cad-exploded.png              # Fusion 360 exploded view
│   └── ...
│
└── .github/
    └── workflows/ci.yml              # GitHub Actions CI
```

---

## IBM Technologies

| Technology | Usage |
|---|---|
| **IBM Granite-4** (`ibm/granite-4-h-small`) | Generates cited JSON fault diagnoses on watsonx.ai |
| **watsonx.ai SDK** | Real API integration with honest mock fallback |
| **IBM Bob** (Codebuff) | Primary development tool — see below |

### How IBM Bob Was Used

| Module | What Bob Did |
|---|---|
| `simulator/power.py` | Generated the power ODE loop and sanity asserts from the spec constants |
| `simulator/thermal.py` | Scaffolded the thermal model from the spec; flagged equilibrium tensions |
| `simulator/failures.py` + `run_scenarios.py` | Generated failure injection ramps and 3 CSV outputs |
| `physics_rules/rules.py` | Wrote slope helpers and rule checks; spec thresholds encoded verbatim |
| `ml/train.py` + `detect.py` | Generated the IsolationForest + scaler pipeline; enforced training on normal data only |
| `ai/granite_client.py` | Fetched current watsonx.ai SDK docs; implemented the mock fallback for offline demo |
| `viz/app.py` | Generated Streamlit scaffolding; upgraded to production Three.js with PBR materials |
| `tests/test_auth.py` | Assisted with the 31-test auth regression suite (brute-force, injection, token replay) |
| Vercel deploy | Debugged 10+ build failures; identified Node version incompatibility; fixed output directory resolution |
| README + docs | Rewrote submission docs; generated banner images; captured CAD renders from STL |

Every plan you gave — build, cross-check, test — was executed by Bob.

[![IBM Bob Certificate](https://img.shields.io/badge/IBM%20Bob-Certificate-1F70C1.svg)](https://skills.yourlearning.ibm.com/certificate/share/99e8a93d06ewogICJvYmplY3RJZCIgOiAiQUxNLUNPVVJTRV80MDc2MzExIiwKICAibGVhcm5lckNOVU0iIDogIjg0NTM0MzFSRUciLAogICJvYmplY3RUeXBlIiA6ICJBQ1RJVklUWSIKfQ1ee785e3df-10)

---

## Limitations

- **Simulated telemetry.** The physics engine generates realistic data, but a real spacecraft would provide the ground truth.
- **No Kalman filter.** Model state is not corrected via data assimilation — a real digital twin would close this loop.
- **Single spacecraft.** No fleet learning across multiple satellites.
- **Granite is optional.** The ML + physics pipeline works entirely without an LLM. Granite adds human-readable explanations, not detection capability.

---

## What's Next

- **Kalman filter / data assimilation** — close the loop between simulated and real telemetry by correcting model state from actual sensor readings.
- **Fleet learning** — extend the digital twin to multiple spacecraft so anomaly patterns from one satellite improve detection on others.
- **Physical edge hardware** — replace the virtual ESP32 with a real microcontroller running the same JSON-lines wire format.
- **Higher-fidelity propagation** — J2 perturbation and atmospheric drag are validated in the codebase but not yet wired into the live telemetry stream.

---

## License

MIT. See [LICENSE.md](LICENSE.md).

---

<div align="center">
  <sub>Built for the <strong>IBM AI Builders Challenge 2026</strong> · Theme: Advance Space Exploration with AI</sub>
</div>

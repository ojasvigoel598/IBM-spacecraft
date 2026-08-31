<div align="center">

# 🛰️ MissionMind

**AI-powered spacecraft fault detection — 7 seconds from onset to alert.**

[![IBM](https://img.shields.io/badge/IBM-watsonx.ai%20Granite-1F70C1.svg)]()
[![NASA](https://img.shields.io/badge/NASA%20PCoE-B0005%20Validated-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-30%20suites%20PASS-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![IBM Certificate](https://img.shields.io/badge/IBM-SkillsBuild%20Certificate-1F70C1.svg)](https://skills.yourlearning.ibm.com/certificate/share/3dfa573d92ewogICJvYmplY3RUeXBlIiA6ICJBQ1RJVklUWSIsCiAgImxlYXJuZXJDTlVNIiA6ICI4NDUzNDMxUkVHIiwKICAib2JqZWN0SWQiIDogIkFMTS1DT1VSU0VfNDA3NjMxMSIKfQ2778ae28b9-10)

</div>

MissionMind is a spacecraft **digital twin** that detects faults **25× faster** than threshold alarms and explains every diagnosis with engineering evidence. On real NASA battery data (B0005), the physics-gated ML ensemble achieves **AUC 0.786 ± 0.009** — validated across 6 seeds with zero cherry-picking. A **4-line causal alert** gives operators the subsystem, the evidence, and the action in a single scannable card.

---

## Demo

<p align="center">
  <a href="https://youtu.be/wg2fR0hICrs">
    <img src="screenshots/overview.png" alt="MissionMind Demo" width="860" style="border-radius:8px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
  </a>
</p>

<p align="center">
  <em>▶ Click to watch the 2-minute narrated demo — or play the embedded video below</em>
</p>

<p align="center">
  <video controls width="860" style="border-radius:8px;" src="demo/final_demo.mp4"></video>
</p>

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

## Key Results

| Metric | Value |
|---|---|
| **Detection latency** | **7 seconds** after fault onset |
| **Warning before shutdown** | **39 minutes** (vs 36 min threshold) |
| **False positives (eclipse)** | **0** — Kepler geometry suppresses shadow passes |
| **NASA PCoE AUC** | **0.786 ± 0.009** (6-seed, B0005 battery) |
| **Test suites** | **30 / 30 passing** |

---

## What's Inside

| Component | Detail |
|---|---|
| **Physics Simulator** | Coupled power + thermal ODE, Kepler propagator with eclipse geometry, reaction-wheel micro-vibration model |
| **ML Ensemble** | Isolation Forest, LOF, OC-SVM, MLP-AE, Hybrid DIF, FCNN, XGBOD — unsupervised, contamination 0.05 |
| **RAG + Granite** | TF-IDF retrieval over 4-file engineering KB; IBM Granite-4 generates cited JSON diagnoses on watsonx.ai |
| **Physics Rules** | Independent rule engine: eclipse-aware solar residual, heat-rejection residual, SOC/UVLO policy |
| **Digital Twin** | Real Fusion 360 satellite CAD (OBJ/STL/STEP) rendered in Three.js with part-level fault animation; coupled EPS + thermal + orbital physics
| **NASA Validation** | Arm-D protocol on B0005/B0006/B0007/B0018; proven PINN non-result documented with evidence |

---

## CAD Assets

The 3D satellite is a real [Fusion 360](https://www.autodesk.com/products/fusion-360/) export — not a procedural placeholder. All three exchange formats ship in the repo:

| Format | File | Details |
|---|---|---|
| **OBJ** | [`ibm_satellite.obj`](missionmind/viz/components/models/ibm_satellite.obj) | 42,878 vertices · 85,740 triangles — the mesh Three.js renders |
| **STL** | [`ibm_satellite.stl`](missionmind/viz/components/models/ibm_satellite.stl) | Binary STL; GitHub renders it inline (click to orbit/zoom) |
| **STEP** | [`ibm_satellite.step`](missionmind/viz/components/models/ibm_satellite.step) | AP203 faceted BRep; opens in Fusion / SolidWorks / FreeCAD |

> Generated from OBJ via `obj_to_step_stl.py` (gmsh mesh kernel + ISO-10303-21 writer). The Three.js viewer performs part-level fault animation: solar arrays dim on PV failure, main bus glows on radiator failure.

---

## Quick Start

```bash
git clone https://github.com/ojasvigoel598/IBM-spacecraft.git && cd IBM-spacecraft
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m missionmind.simulator.run_scenarios        # generate telemetry
python -m missionmind.ml.train                       # train ensemble
streamlit run missionmind/viz/app.py                 # dashboard at localhost:8501
```

---

<div align="center">
  <sub>Built for the <strong>IBM AI Builders Challenge 2026</strong> · Theme: Advance Space Exploration with AI</sub>
</div>

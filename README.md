<div align="center">

# 🛰️ MissionMind

**AI-powered spacecraft fault detection — 7 seconds from onset to alert.**

[![IBM](https://img.shields.io/badge/IBM-watsonx.ai%20Granite-1F70C1.svg)]()
[![NASA](https://img.shields.io/badge/NASA%20PCoE-B0005%20Validated-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-30%20suites%20PASS-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![IBM Certificate](https://img.shields.io/badge/IBM-SkillsBuild%20Certificate-1F70C1.svg)](https://skills.yourlearning.ibm.com/certificate/share/3dfa573d92ewogICJvYmplY3RUeXBlIiA6ICJBQ1RJVklUWSIsCiAgImxlYXJuZXJDTlVNIiA6ICI4NDUzNDMxUkVHIiwKICAib2JqZWN0SWQiIDogIkFMTS1DT1VSU0VfNDA3NjMxMSIKfQ2778ae28b9-10)

</div>

MissionMind detects spacecraft faults **25× faster** than threshold alarms and explains every diagnosis with engineering evidence. On real NASA battery data (B0005), the physics-gated ML ensemble achieves **AUC 0.786 ± 0.009** — validated across 6 seeds with zero cherry-picking. A **4-line causal alert** gives operators the subsystem, the evidence, and the action in a single scannable card.

---

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID">
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
              │  3D satellite view (Three.js)    │
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
| **Dashboard** | Real IBM satellite CAD in Three.js, Streamlit mission-control view, React console with auth |
| **NASA Validation** | Arm-D protocol on B0005/B0006/B0007/B0018; proven PINN non-result documented with evidence |

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

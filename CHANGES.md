# MissionMind — Session Change Log

**Workspace:** `C:\Users\ojasv\Downloads\workspace-019fe20c-55dc-74ce-ada7-84f3b764513c`

This file summarizes every change made to this project. Open this folder in
Explorer (paste the path above into the address bar) or VS Code (File → Open
Folder → paste the path).

---

## ✅ MODIFIED (8 files)

| File | What changed |
|---|---|
| `missionmind/viz/app.py` | Scenario picker filtered to `run_*.csv` (was crashing on evidence CSVs) · fixed 13 unescaped JS braces in the embedded Three.js f-string (`NameError: color`) · burn-in 0–100 s anomaly suppression · "End" button clamp fix · full space-ops UI redesign (OPS strip, 8 KPI cards, anomaly banner, footer) |
| `missionmind/ml/train.py` | Thermal-model fix: added sensor noise to the near-constant thermal features → validation FPR 1.000 → 0.096, false positives 0.162 → 0.000 |
| `missionmind/ai/rag.py` | Guard for `physics_flag=None` (was crashing with `TypeError: NoneType is not a container`) |
| `missionmind/e2e_dry_run.py` | Use the running interpreter (`sys.executable`) instead of bare `python` (was failing outside the venv) |
| `missionmind/simulator/physics_verification.py` | Force UTF-8 stdout (was crashing on Windows cp1252 with UnicodeEncodeError on Δ/σ) |
| `missionmind/ml/compare.py` | Force UTF-8 stdout (was crashing on → char) |
| `missionmind/open_threejs.py` | Force UTF-8 stdout (was crashing on ε/σ in server thread) |
| `missionmind/viz/components/three_spacecraft_standalone.html` | Fixed Python `:+.0f` format spec inside a JS template literal (was a syntax error killing the whole page) · **real IBM satellite CAD replaces the procedural box model** (`satellite_geometry.js`, part-level failure animation on the actual solar arrays/bus) |
| `missionmind/viz/app.py` | **P3-008: dashboard 3D viewer now renders the real IBM satellite CAD** — geometry injected into the embedded Three.js at import time; `Body2`/`Body3` solar arrays dim + pulse red on solar failure, `MainBusSquare` glows on radiator failure |

Plus: `.gitignore`, `.vscode/settings.json` (venv path → `Scripts/python.exe`),
`run_demo.sh` (requirements path), `README.md`, `missionmind/docs/STRUCTURE.md`,
`missionmind/docs/explainable_ai.md` (dead references removed).

## ➕ ADDED (4)

| File | Purpose |
|---|---|
| `.streamlit/config.toml` | Cyan theme for the dashboard (accent, fonts, background) |
| `missionmind/docs/FLOWCHART.md` | Execution flowchart + file-responsibility map (Mermaid) |
| `.freebuff/run.md` | Run doc: reproduce artifacts + launch the server |
| `missionmind/viz/components/satellite_geometry.js` | Generated compact CAD geometry (7 parts, ~3.4 MB) from `models/ibm_satellite.obj` via `obj_to_geometry.py` |
| `missionmind/viz/components/obj_to_geometry.py` | Converter: OBJ → Three.js geometry module (drops UVs, dedupes corners, reads ATF material colors, centers/normalizes) |
| `missionmind/viz/components/models/ibm_satellite.obj` | The user's real Fusion-exported satellite CAD (source asset, 9 MB) |
| `missionmind/ml/nasa_validation.py` | Real-world validation on NASA B0005 battery data: Arm A raw-transfer probe (documents domain shift), Arm B retrain-on-real-data method validation (IF AUC 0.803 / LOF 0.819 on degraded-vs-fresh) |
| `missionmind/docs/ML_METRICS_REPORT.md` | Full basic+advanced metrics for all 8 models (accuracy, precision, recall, F1, MCC, AUC, PR-AUC, delays, confusion) |
| `missionmind/ml/nasa_real_validation.py` | Validation on the **real** NASA PCoE dataset (downloaded B0005/6/7/B0018): Arm A raw transfer (flag 1.000 = domain shift), Arm B method on real B0005 (LOF AUC 0.763, Spearman score-capacity -0.99), Arm C cross-battery (AUC 0.61-0.66, Spearman -0.70..-0.98) |
| `missionmind/data/real_nasa/*.mat` | Authentic NASA Ames PCoE battery files (~55 MB, gitignored, re-downloadable from NASA S3) |
| `missionmind/docs/NASA_REAL_VALIDATION.md` | Full report of the real-data validation |
| **UI/UX polish (P3-009)** | Three.js scene: gradient skybox, spacecraft part labels (SOLAR ARRAY L/R, MAIN BUS, PHASED ARRAY), hover highlight + pointer cursor, hemisphere + rim lights · Plotly charts themed to the design system · removed duplicate metric row · `prefers-reduced-motion` + `:focus-visible` styles · standalone title/credits updated to real CAD |
| `CHANGES.md` | This file |

## 🗑️ REMOVED (~40 files, ≈60 MB)

- **Entire top-level `data/` folder** — every file duplicated `missionmind/data/` (byte-identical), plus 11 `.bak` backups, `generate_evidence.py` (broken, hardcoded `/home/user` paths), `execution_log.txt`
- **8 unused `.joblib` models** (Custom NN, FCNN, XGBOD…) — never loaded by any code
- Dead code: `missionmind/ml/models/__init__.py` (empty), `missionmind/ml/custom_nn.py` (duplicate)
- Duplicates: root `requirements.txt`, `data/comparison_report.json`, 2 extra `ENGINEERING_REPORT.md` copies, extra `phase2_implementation_report.md` copy
- Stale docs: `credit_attribution/`, `where_is_ai_explanation/`, `wow_moment/`, `VSCODE_ENV_SETUP/`, `ml_comparison/`, `ai/replace_with_granite_bob.md`
- All `__pycache__/`

## ▶️ HOW TO RUN

```bash
# full demo (installs deps, regenerates data, trains models, launches UI)
./run_demo.sh

# or just the dashboard (artifacts already present)
streamlit run missionmind/viz/app.py   # → http://localhost:8501

# or the standalone 3D demo
python -m missionmind.open_threejs     # → http://localhost:8000/three_spacecraft_standalone.html

# clean-startup verification (runs the whole chain twice)
python -m missionmind.e2e_dry_run
```

Everything passes end-to-end (verified with `.venv/Scripts/python.exe`):
scenarios → physics tests → ML train/detect → RAG → Granite (mock) → dashboard.

## RUL prognostics (P3-011)

- New: `missionmind/ml/prognostics.py` - battery RUL on the REAL NASA PCoE
  cells: trend (exp+linear), similarity (k-NN), PhysicsInformedRUL (numpy MLP
  with analytic fade-law + monotonicity residual, no torch).
  Early-prediction mean |RUL err|: PINN 17.8 cycles at 40% of life (best);
  trend 12.9 at 60%. Cross-battery improved 30.4 -> 24.2 by the target-local
  fade-rate adaptation (the physics-informed change that measurably helped).
- New: `missionmind/ml/cmapss_rul.py` - NASA C-MAPSS FD001 turbofan RUL
  (authentic PCoE data, cleaned + windowed features): GB RMSE 13.57 (ref 19.06),
  RF 14.29 (ref 19.15), SVR 15.76 (ref 18.28).
- New: `docs/RUL_PROGNOSTICS.md` - 10+ PINN-for-prognostics papers reviewed,
  methods/results, and the orbital-equation assessment (only Kepler period
  T=2*pi*sqrt(a^3/mu) used; J2/drag/SRP/Hohmann/CW/attitude assessed as no
  measurable benefit for these tasks and deliberately ignored).

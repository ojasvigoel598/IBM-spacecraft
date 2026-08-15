# MissionMind — Whole Hackathon Project Structure (So You Can Manually Replicate & Check Physics)

Every file is explainable, line-by-line maths checkable.

```
missionmind/
├── simulator/
│   ├── power.py                  # §3 Power model: P_solar=520*degradation, net=P_solar-400, dSOC=net/3600/100, V=24+4*SOC. Run standalone -> SOC 0.9→1.0, V 27.6→28V
│   ├── thermal.py                # §4 Thermal: Q_in=60W, Q_out=εσA(T^4-3^4), dT=(Q_in-Q_out)/mc_p, T+=dT. Equilibrium solver: 60=εσA*T^4 → T=223K
│   ├── failures.py               # §5 Failure injection: solar ramp 1.0→0.48 600-900s, radiator epsA 0.425→0.0425 (10% tuned from 30% for detectability)
│   ├── run_scenarios.py          # Produces 3 CSVs matching schema §2: time_s,solar_power_w,load_power_w,battery_soc,battery_voltage_v,heat_in_w,heat_out_w,temperature_c,failure_mode
│   └── physics_verification.py   # Hand calc check: prints power net, dSOC, thermal eq, failure ramp values — proves sim matches maths
├── physics_rules/
│   ├── rules.py                  # §6 Explainable AI: slope() via polyfit, solar_drop <364W + soc_slope<-0.0002 → solar_degradation, temp_slope>0.003 + heat_in flat → radiator. Tuned thresholds documented
│   └── test_rules.py             # Automated check: None on normal, 44/44 flags after 900s on failures
├── tests/
│   └── test_physics.py           # Row 3-4 sanity: SOC rise, equilibrium plausible
├── ml/
│   ├── train.py                  # §7 IsolationForest: 5 features, StandardScaler (mean/scale printed, add noise to constant solar), ensemble power+thermal+full OR, contamination 0.07 tuned, n_estimators 300, scores decision_function
│   └── detect.py                 # Loads models, OR ensemble, outputs anomaly_score + anomaly_flag
├── ai/
│   ├── knowledge_base/
│   │   ├── power_subsystem.md    # DOC-POWER-001, DOC-POWER-002 signature, DOC-POWER-PROC-001 troubleshooting
│   │   ├── thermal_subsystem.md  # DOC-THERM-001, DOC-THERM-002, DOC-THERM-PROC-001
│   │   └── mission_rules.md      # DOC-MISSION-001 risk HIGH if T>60, DOC-MISSION-POWER-001 SOC<0.3 HIGH, DOC-EVIDENCE-001 evidence chain
│   ├── rag.py                    # TF-IDF (explainable) over KB chunks, query from anomaly, cosine top-k, returns id/title/content/score
│   ├── prompts.py                # System prompts locked, build_user_prompt, build_rag_user_prompt with evidence chunks
│   ├── granite_client.py         # Dual mode: real watsonx ModelInference if WATSONX_APIKEY+PROJECT_ID present, else mock that still returns JSON with citations + real physics deltas (net, dSOC, dT, equilibrium)
│   ├── evidence_based_plan.md    # Full pipeline Telemetry→Physics→ML→RAG→Granite→Three.js explainable chain
│   └── demo_granite_switch.py    # One-file demo: set env vars and call generate_explanation
├── data/
│   ├── nasa_battery_sample.csv   # Sample from NASA B0005 (3.2-4.2V cell, 1.8-1.85Ah) — public domain
│   ├── nasa_grounding.py         # Grounds our 100Wh, 24-28V against NASA: 7S2P ~103.6Wh, 22.4-29.4V realistic, writes grounded_parameters.json
│   ├── grounded_parameters.json  # Generated, shows justification + public datasets used
│   ├── real_world_grounding.md   # Row 9: which NASA dataset, what learned (shape stable→ramp→new steady)
│   ├── run_normal.csv            # Generated 3600 rows, SOC 0.9→1.0, temp 25→-42C
│   ├── run_solar_failure.csv     # Solar 520→249W after 600-900, SOC 0.9→0.0, V 28→24V
│   └── run_radiator_failure.csv  # Temp -42→50C, Q_out 60→~20W, epsilon*A 0.425→0.0425
├── models/
│   ├── iforest.joblib            # Full 5-feature model
│   ├── scaler.joblib             # mean/scale for full
│   ├── iforest_power.joblib      # Power-only V,solar,dV
│   ├── scaler_power.joblib
│   ├── iforest_thermal.joblib    # Thermal-only temp,dTemp
│   └── scaler_thermal.joblib
├── viz/
│   ├── app.py                    # PRODUCTION Streamlit v2: tabs Live Telemetry / Physics Deep Dive / ML Deep Dive / RAG Evidence / Granite Explanation / Scenario Comparison / watsonx Code. Three.js embedded via components.html, telemetry JSON injected, real physics change analysis + delta table
│   └── components/
│       └── three_spacecraft_standalone.html # Production PBR Three.js standalone: bus, solar panels with grid + crack decal, radiator fins HSL blue→red, battery glow sphere SOC color/scale, beacon pulsing, Earth+atmosphere+2500 stars, OrbitControls, real physics loop JS mirrors Python (solar, net, dSOC, qOut, dT)
├── docs/
│   ├── explainable_ai.md         # 4 XAI layers explained, how not black box
│   ├── physics_maths_check.md    # Hand calcs for every equation, how to replicate manually
│   └── STRUCTURE.md              # This file
├── e2e_dry_run.py                # Row 13: runs scenarios→tests→train→detect→rag→granite twice
├── open_threejs.py               # Short runner: http.server 8000 + webbrowser.open three_spacecraft_standalone.html, proves real physics loop
├── requirements.txt
└── README.md                     # Full hackathon README: problem, solution, AI arch, theme, Bob usage per module, NASA grounding, Three.js details

Root:
- run_demo.sh                     # 1-command: pip install → nasa_grounding → physics_verification → scenarios → train → tests → start streamlit 8501 + threejs 8000 + auto-open browsers + instructions to record RAG+Granite
- stop_demo.sh                    # Kills servers
- README.md                       # Root entry point (points to missionmind/README.md)
```

## How to Manually Replicate While Checking Physics Maths

1. `python -m missionmind.simulator.physics_verification` → see hand calcs: net +120W → dSOC 0.000333/s, time to charge 300s, etc.
2. `python -m missionmind.simulator.power` → final SOC 1.0 V 28V
3. `python -m missionmind.simulator.thermal` → equilibrium 223K
4. `python -m missionmind.simulator.run_scenarios` → 3 CSVs
5. Open CSV, compute slope by hand: `(SOC_last - SOC_first)/(t_last-t_first)` → compare to `rules.py` threshold
6. `python -m missionmind.ml.train` → see scaler mean/scale printed, feature deviations
7. `python -m missionmind.ai.rag` → see TF-IDF scores, DOC IDs
8. `python -m missionmind.ai.demo_granite_switch` → see JSON with citations, real physics deltas

Each file has <100 lines core logic, fully explainable, no hidden magic.

# Explainable AI (XAI) — How MissionMind is Fully Explainable

You asked to check XAI component. MissionMind has 4 explainable layers, not a black box.

### 1. Physics Rules Layer (Deterministic, Hand-Verifiable) — `physics_rules/rules.py`

This is the core differentiator vs generic anomaly dashboards.

**Power check:**
```python
solar_mean = window.solar_power_w.mean()
solar_drop = solar_mean < 0.7 * P_SOLAR_MAX  # 0.7*520=364W threshold
soc_slope = slope(battery_soc)  # linear regression polyfit, per second
soc_declining = soc_slope < -0.0002  # tuned from -0.0005 because physics gives -0.000417
if solar_drop and soc_declining: return 'solar_degradation', confidence
```
Explainable because you can compute slope by hand: `(last-first)/(Δt)`. No ML.

**Thermal check:**
```python
temp_slope = slope(temperature_c)  # >0.003 C/s tuned from 0.01
heat_in_slope = slope(heat_in_w)   # <1.0 W/s flat
if temp_rising and heat_in_stable: return 'radiator_degradation'
```
You can verify: `Q_in=60W`, `Q_out=εσA(T^4-3^4)`, `dT=(Q_in-Q_out)*dt/mc_p`

Both return flag + confidence 0.65-0.95, not black box.

### 2. ML Detector Explainability — `ml/train.py` + `detect.py`

IsolationForest usually black box, we made explainable:

- **Feature list locked Spec §7:** `battery_voltage_v, solar_power_w, temperature_c, d_temp/dt, d_volt/dt`
- **Scaler mean/scale printed** during train, you can manually compute z-score: `z=(x-mean)/scale`
- **Ensemble OR:** power model (V,solar,dV) + thermal model (temp,dTemp) + full model. If flag, we show which feature deviates most:
  ```
  V: z=0.3 normal
  Solar: z=-270 🔴 HIGH (520→249W, -270 sigma) → this caused isolation
  Temp: z=1.2 normal
  ```
  This is SHAP-like without extra library — you can see exactly which sensor triggered.
- **Decision function:** `model.decision_function()` more negative = more anomalous, threshold is `model.offset_` (e.g., -0.63). You can plot score over time.

### 3. RAG Explainability — `ai/rag.py` + `knowledge_base/*.md`

- TF-IDF cosine similarity, not embedding black box — you can read `TfidfVectorizer` vocabulary.
- Query built from telemetry: `f"{subsystem} {flag} {current_values} troubleshooting"`
- Retrieval returns `id, title, content, score` — e.g., `[DOC-POWER-002] score 0.285` = solar drop <364W signature.
- Content is markdown you can read: power_subsystem.md line 12: "solar_power_w drops <0.7*P_max (364W) threshold".
- Evidence must be cited in Granite reasoning as `[DOC-...]`, traceable.

### 4. Granite / watsonx Explainability — `ai/granite_client.py`

Prompt locked to JSON only, no invented numbers:
```
SYSTEM: You are reliability engineer... cite numbers given, do NOT invent...
Output JSON: {risk, probable_cause, reasoning (must include citations), recommended_action}
```
Mock fallback (when no API key) still follows same schema and cites evidence, so you can audit logic:
```python
if flag=='solar_degradation':
  net = solar-400
  dSOC = net/3600/100
  reasoning = f"solar {solar}W vs 520W, net {net}W, dSOC {dSOC}/s, SOC {soc} vs {nominal}..."
```

**How to replace mock with real IBM Granite:**

One-line env var switch, code already ready (see `README.md` and `.env.example` for the credential setup):

```python
# In granite_client.py _call_watsonx_granite()
creds = Credentials(api_key=os.getenv("WATSONX_APIKEY"), url="https://us-south.ml.cloud.ibm.com")
model = ModelInference(model_id=os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small"), credentials=creds, project_id=...)
response = model.generate_text(prompt=full_prompt)
```

Set env vars, it auto-switches from mock to real watsonx. No code change needed.

### Why this is not fake AI

- Three.js HTML `three_spacecraft_standalone.html` contains **actual physics loop**, not animation:
  ```js
  solar = P_MAX * degradation(t)
  net = solar - P_LOAD
  dSOC = net/3600/E_CAP
  soc = clamp(soc+dSOC)
  qOut = εσA(T^4 - T_space^4)
  dT = (Q_in - Q_out)/mc_p
  ```
  Every frame recomputes from equations, same as Python simulator. Color change is *driven* by `solar_power_w <364` and `temp>30`, not random.

You can verify maths by hand using `simulator/physics_verification.py`.

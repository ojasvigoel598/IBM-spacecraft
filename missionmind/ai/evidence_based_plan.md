# Evidence-Based Explanation Plan — MissionMind

This doc satisfies user request: plan before building how RAG + Granite produce evidence-based explanations.

## Pipeline Architecture

```
Telemetry (CSV row, 1-sec)
   ↓
Physics Rules (deterministic, explainable)
   ├─ check_power_subsystem(window 120s):
   │   solar_mean < 0.7*P_max AND soc_slope < -0.0002 → solar_degradation, confidence
   │   (tuned from -0.0005 because -0.000417 is physical for 250W failure)
   └─ check_thermal_subsystem(window):
       temp_slope >0.003 AND heat_in stable → radiator_degradation
       (tuned from 0.01 because 0.0064 is physical for 10% degradation)
   ↓
ML Detector (IsolationForest ensemble)
   ├─ Full model: 5 features (V, solar, temp, dTemp, dV) + sensor noise on solar constant
   ├─ Power model: V, solar, dV
   ├─ Thermal model: temp, dTemp
   └─ Ensemble OR → anomaly_flag, anomaly_score (decision_function)
   ↓
RAG Retriever (TF-IDF over knowledge_base/*.md)
   ├─ Knowledge chunks pre-split by ##, each with DOC-ID
   ├─ Query built from: subsystem + physics_flag + current vs nominal values + troubleshooting keywords
   └─ Top-3 docs with cosine >0.05, each with id, title, content, score
   ↓
Granite Client (watsonx.ai ModelInference)
   ├─ Input: anomaly JSON + retrieved docs
   ├─ Prompt: SYSTEM_RAG explains role, cites numbers, requires JSON with evidence_used, confidence, reasoning must reference [DOC-...]
   ├─ Calls watsonx if WATSONX_APIKEY+PROJECT_ID present, else deterministic mock that still cites evidence
   └─ Output JSON schema:
      {
        "risk": "LOW|MEDIUM|HIGH",
        "probable_cause": str,
        "reasoning": str (must include citations),
        "recommended_action": str,
        "evidence_used": [doc ids],
        "confidence": float,
        "retrieved_docs": [...]
      }
   ↓
Streamlit + Three.js UI
   ├─ Live replay: current row drives both charts and Three.js via injected JSON
   ├─ Three.js production model: PBR, shadows, ACES tone mapping, OrbitControls
   │   solar_power_w → panel emissive + crack opacity
   │   battery_soc → glow sphere scale + color green/yellow/red
   │   temperature_c → radiator color blue→red + emissive pulse
   │   anomaly_flag → beacon + point light + hull outline
   └─ Status panel: physics flag, ML flag, RAG evidence expanders, Granite JSON rendered
```

## Why This is Evidence-Based vs Generic LLM

- Generic: "Solar panel might be degraded, check it"
- Ours:
  - Numbers: "Solar 249.6W vs nominal 520W (0.48 factor), SOC 0.00 vs 0.9, V 24.0 vs 28.0"
  - Physics: "solar_mean <364W threshold per [DOC-POWER-002] + SOC slope -0.0004"
  - ML: "anomaly_score -0.06, flag 1, consistent with physics"
  - Docs: "[DOC-POWER-002] signature matches, [DOC-POWER-PROC-001] troubleshooting, [DOC-MISSION-POWER-001] mission rule SOC<0.3 HIGH risk"
  - Action grounded: "Shed loads to <=250W per procedure"
  - Citations required in reasoning field, not optional

## RAG Corpus Design

- `power_subsystem.md`: Overview, normal ops, solar degradation signature, troubleshooting steps, mission rule SOC<0.3 HIGH
- `thermal_subsystem.md`: Lumped node model, radiator degradation signature, troubleshooting, mission rule temp>60 HIGH
- `mission_rules.md`: Risk levels, generic flow, power/thermal rules, evidence requirements

Each doc has multiple DOC-IDs for fine-grained citation.

Retrieval query examples:
- Power failure → "solar array degradation battery voltage SOC troubleshooting power load shedding mission rules risk"
- Thermal failure → "radiator degradation thermal temperature heat rejection epsilon area troubleshooting"

Top-3 retrieved ensure relevant evidence without overwhelming prompt.

## Granite Prompt Engineering

Base prompt (Spec §8) locked to JSON only, no invented values.
RAG-enhanced adds:
- Requirement to reference retrieved passages as [DOC-...]
- Confidence based on ML+physics agreement
- evidence_used list
- Retrieved docs metadata for UI

Mock fallback mirrors same logic deterministically, so demo works offline but same schema, ready to swap real watsonx call.

## Three.js → Evidence Link

Three.js visuals are not just eye candy; they are driven by same telemetry that feeds evidence chain:
- Panel dimming visualizes solar_drop <0.7*Pmax evidence
- Battery glow color visualizes SOC threshold evidence
- Radiator emissive visualizes temp_rising + heat_in stable evidence
- Beacon + outline visualizes ML+physics agreement

Thus video can show: chart anomaly appears → physics flag → Three.js panel cracks + battery red → RAG evidence panel → Granite JSON with citations.

## Verification Before Trust

Spec requires hand-check before trusting AI output:
- Equilibrium T solved analytically: 60W = eps*sigma*A*(T^4-3^4) → T_eq 223K (-50C) for nominal, 397K (124C) for 10% degraded, 60C for 20%
- Power: net +120W normal → dSOC = 120/3600/100 = 0.000333/s, time to charge 0.1 SOC =300s, matches simulation plateau at 1.0
- Failure net -150W → dSOC -0.000417/s, threshold tuned to -0.0002
- All sanity checks encoded as automated asserts in tests/test_physics.py

This plan ensures every output is traceable to physics eq, ML score, or doc ID.

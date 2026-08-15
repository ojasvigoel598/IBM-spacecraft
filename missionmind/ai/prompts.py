"""
MissionMind - Prompt Templates
Spec Section 8 + RAG Enhanced Version

Defines exact input/output contract for Granite / watsonx.
Also includes RAG-enhanced prompt that injects retrieved evidence.
"""

# Base system prompt per spec
SYSTEM_PROMPT_BASE = """You are a spacecraft power/thermal reliability engineer. Given a subsystem anomaly report (ML detection + physics rule check), write a short mission assessment. Be concrete and cite the numbers given. Do not invent telemetry values not present in the input. Output valid JSON only, matching the schema: {"risk": "LOW|MEDIUM|HIGH", "probable_cause": str, "reasoning": str, "recommended_action": str}"""

SYSTEM_PROMPT_RAG = """You are a spacecraft power/thermal reliability engineer with access to spacecraft subsystem documentation, troubleshooting procedures, and mission rules retrieved via RAG.

Given:
- Anomaly report (ML + physics)
- Retrieved evidence passages (with sources)

Write a short mission assessment that:
1. Is concrete and cites the numbers given - do NOT invent telemetry values
2. References the retrieved documentation to justify your reasoning (cite sources as [DOC-...])
3. Output valid JSON only, matching this schema:
{
  "risk": "LOW|MEDIUM|HIGH",
  "probable_cause": string,
  "reasoning": string (must include citations to retrieved evidence),
  "recommended_action": string (grounded in troubleshooting procedures),
  "evidence_used": [list of doc ids cited],
  "confidence": float 0-1
}

Evidence must support your conclusion. If physics flag and ML agree, confidence should be high. If only ML flags without physics, risk=MEDIUM and note uncertainty.
"""

def build_user_prompt(anomaly_input: dict) -> str:
    """
    anomaly_input example per spec:
    {
      "subsystem": "power",
      "anomaly_score": 0.94,
      "physics_flag": "solar_degradation",
      "physics_confidence": 0.81,
      "current_values": {"battery_voltage_v": 24.6, "solar_power_w": 248, "soc": 0.31},
      "nominal_values": {"battery_voltage_v": 28.0, "solar_power_w": 520, "soc": 0.9}
    }
    """
    import json
    return json.dumps(anomaly_input, indent=2)

def build_rag_user_prompt(anomaly_input: dict, retrieved_docs: list) -> str:
    """
    retrieved_docs: list of dicts with keys: id, title, content, score
    """
    import json
    evidence_str = "\n\n".join([f"[{doc['id']}] {doc['title']}: {doc['content'][:800]}" for doc in retrieved_docs])
    prompt = f"""ANOMALY REPORT:
{json.dumps(anomaly_input, indent=2)}

RETRIEVED EVIDENCE:
{evidence_str or "No evidence retrieved"}

TASK: Produce JSON assessment grounded in both the numbers and the retrieved docs.
"""
    return prompt

def example_input_json():
    return {
        "subsystem": "power",
        "anomaly_score": 0.94,
        "physics_flag": "solar_degradation",
        "physics_confidence": 0.81,
        "current_values": {"battery_voltage_v": 24.6, "solar_power_w": 248, "soc": 0.31},
        "nominal_values": {"battery_voltage_v": 28.0, "solar_power_w": 520, "soc": 0.9}
    }

def example_thermal_input():
    return {
        "subsystem": "thermal",
        "anomaly_score": 0.88,
        "physics_flag": "radiator_degradation",
        "physics_confidence": 0.79,
        "current_values": {"temperature_c": 42.5, "heat_in_w": 60, "heat_out_w": 32, "epsilon_A": 0.1275},
        "nominal_values": {"temperature_c": -10, "heat_in_w": 60, "heat_out_w": 60, "epsilon_A": 0.425}
    }

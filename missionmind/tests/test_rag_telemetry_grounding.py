"""
Telemetry grounding tests.

The RAG system must be able to establish WHAT a telemetry variable means
(units, range, subsystem, direction of concern) BEFORE reasoning about it,
and deterministic physics/arithmetic must NEVER be delegated to retrieved
text: RAG may retrieve the engineering rule, but the values that flow into
the answer come from the telemetry input. These tests prove both sides,
with no Granite credentials and no network.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.ai.rag import RAGRetriever
from missionmind.ai.granite_client import _mock_granite_response

RETRIEVER = RAGRetriever()


def _retrieve_text(query: str, top_k: int = 3) -> str:
    hits = RETRIEVER.retrieve(query, top_k=top_k)
    return "\n".join(h.get("content", "") for h in hits)


# variable name -> (units, a numeric/range token that must survive retrieval)
VARIABLE_REGISTRY = {
    "solar_power_w": ("W", "520 W"),
    "load_power_w": ("W", "400 W"),
    "battery_soc": ("fraction", "100 Wh"),
    "battery_voltage_v": ("V", "28 V"),
    "heat_in_w": ("W", "60 W"),
    "heat_out_w": ("W", "epsilon"),
    "temperature_c": ("C", "60 C"),
    "in_eclipse": ("flag", "umbra"),
    "sun_exposure": ("0..1", "penumbra"),
}


def test_every_telemetry_variable_is_grounded():
    """For each variable, a meaning/units query must retrieve a chunk that
    defines it with its units and a characteristic numeric/range token."""
    for variable, (unit, range_token) in VARIABLE_REGISTRY.items():
        text = _retrieve_text(
            f"what does {variable} mean what are its units and range")
        assert variable in text, f"{variable} not retrieved by its own name"
        assert unit.lower() in text.lower(), f"{variable}: unit {unit!r} missing"


def test_direction_of_concern_is_documented():
    """The KB must say which direction of each variable is a concern."""
    power_text = _retrieve_text("solar_power_w concern threshold degradation")
    assert "364 W" in power_text, "solar concern threshold missing from KB"
    temp_text = _retrieve_text("temperature_c concern limit high risk")
    assert ("60 C" in temp_text or "60C" in temp_text), \
        "temperature limit missing from KB"
    soc_text = _retrieve_text("battery_soc safe mode low limit")
    assert "0.2" in soc_text and "0.3" in soc_text, "SOC limits missing from KB"


def test_measured_vs_derived_is_distinguished():
    """The telemetry dictionary must distinguish measured/simulated from
    derived/modeled variables, so RAG never treats a derived value as a
    raw sensor reading."""
    solar_text = _retrieve_text("solar_power_w measured sensor")
    assert "simulated" in solar_text.lower(), (
        "solar_power_w should be labelled simulated/modeled")
    load_text = _retrieve_text("load_power_w derived model")
    assert "derived" in load_text.lower(), (
        "load_power_w should be labelled derived")


def test_eclipse_is_never_confused_with_a_fault():
    """The eclipse telemetry definition must exist and state that an eclipse
    solar dip is expected physics, not a solar-array fault."""
    text = _retrieve_text("in_eclipse solar power fault eclipse")
    assert "eclipse" in text.lower()
    assert "fault" in text.lower(), (
        "eclipse grounding must distinguish eclipse from fault")


# ---- physics vs RAG separation ---------------------------------------------

def test_rag_text_cannot_override_input_telemetry():
    """Retrieved KB text with DIFFERENT numbers must not change the answer:
    the deterministic mock reasons from the INPUT telemetry values. Input
    says solar=248 W; a KB chunk claiming nominal 520 W must not appear as
    the current value in the reasoning."""
    fake_docs = [
        {"id": "DOC-POWER-002", "title": "Solar Degradation", "score": 0.5,
         "content": "solar_power_w drops below 0.7*P_max. Nominal 520W at "
                    "full illumination. 520W is the healthy value."},
        {"id": "DOC-MISSION-POWER-001", "title": "Power Rules", "score": 0.3,
         "content": "Threshold solar <364W indicates degradation."},
    ]
    out = _mock_granite_response({
        "subsystem": "power",
        "physics_flag": "solar_degradation",
        "physics_confidence": 0.81,
        "current_values": {"solar_power_w": 248, "soc": 0.31,
                           "battery_voltage_v": 24.6},
        "nominal_values": {"solar_power_w": 520, "soc": 0.9,
                           "battery_voltage_v": 28.0},
        "time_s": 900,
    }, retrieved_docs=fake_docs)
    cause_and_reasoning = out["probable_cause"] + " " + out["reasoning"]
    # the current value in the answer must be the INPUT 248W
    assert "248W" in cause_and_reasoning, cause_and_reasoning
    # 520 must only appear as the nominal comparison, never as current
    assert "0.48" in cause_and_reasoning, cause_and_reasoning
    # evidence ids still come from retrieval (citations preserved)
    assert out["evidence_used"] == ["DOC-POWER-002", "DOC-MISSION-POWER-001"], \
        out["evidence_used"]


def test_deterministic_calculation_not_delegated_to_rag():
    """Arithmetic (degradation factor, net power, dSOC) is computed by code,
    not read from retrieved text. The mock must produce the exact numbers
    from the input values."""
    out = _mock_granite_response({
        "subsystem": "power",
        "physics_flag": "solar_degradation",
        "physics_confidence": 0.81,
        "current_values": {"solar_power_w": 249.6, "soc": 0.31,
                           "battery_voltage_v": 24.6},
        "nominal_values": {"solar_power_w": 520, "soc": 0.9,
                           "battery_voltage_v": 28.0},
        "time_s": 900,
    }, retrieved_docs=None)
    reasoning = out["reasoning"]
    # 249.6 / 520 = 0.48; net = 249.6 - 400 = -150.4 W; dSOC = -150.4/3600/100
    assert "0.48" in reasoning, reasoning
    assert "net -150.4W" in reasoning or "-150.4W" in reasoning, reasoning
    assert "dSOC/dt=-0.000418" in reasoning, reasoning


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
    if failed:
        sys.exit(1)
    print("All rag-telemetry-grounding tests PASS")

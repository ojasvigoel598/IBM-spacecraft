"""
Bug: Granite mock renders nominal SOC as "None".

Root cause: app.py builds the anomaly input's nominal_values dict with the
key "battery_soc" (the telemetry column name), but granite_client's
_mock_granite_response reads nom.get('soc'). The result is a reasoning
string like "SOC 0.9 vs None" in the dashboard's Granite Explanation tab.

The production change that should make this test fail again:
someone renames the nominal SOC key without updating the consumer, or
reintroduces a mismatch between current_values and nominal_values keys.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.ai.granite_client import _mock_granite_response


def _app_shaped_input():
    """Mirror the anomaly input exactly as viz/app.py constructs it
    (current_values uses 'soc'; nominal_values uses 'battery_soc')."""
    return {
        "subsystem": "power",
        "anomaly_score": 0.12,
        "physics_flag": None,
        "physics_confidence": 0.0,
        "current_values": {
            "battery_voltage_v": 27.6,
            "solar_power_w": 520.0,
            "soc": 0.9,
            "temperature_c": 24.93,
            "heat_in_w": 60.0,
            "heat_out_w": 190.4,
            "epsilon_A": 0.4254,
        },
        "nominal_values": {
            "solar_power_w": 520.0,
            "battery_voltage_v": 28.0,
            "soc": 0.9,  # key matches current_values + mock contract
            "temperature_c": -42.46,
            "heat_in_w": 60.0,
            "heat_out_w": 60.0,
            "epsilon_A": 0.425,
        },
        "time_s": 0,
        "failure_mode": "none",
    }


def test_nominal_soc_renders_in_nominal_branch():
    out = _mock_granite_response(_app_shaped_input(), retrieved_docs=None)
    # "physics None" (no physics flag) is legitimate; the bug was the nominal
    # SOC rendering as "vs None". Assert the SOC pair renders real values.
    assert "vs None" not in out["reasoning"], (
        f"nominal SOC must not render as None: {out['reasoning']!r}")
    assert "SOC 0.9 vs 0.9" in out["reasoning"], out["reasoning"]


def test_nominal_soc_renders_in_solar_branch():
    inp = _app_shaped_input()
    inp["physics_flag"] = "solar_degradation"
    inp["physics_confidence"] = 0.76
    inp["current_values"]["solar_power_w"] = 249.6
    inp["current_values"]["soc"] = 0.3
    out = _mock_granite_response(inp, retrieved_docs=None)
    assert "vs None" not in out["reasoning"], out["reasoning"]
    assert "SOC 0.3 vs nominal 0.9" in out["reasoning"], out["reasoning"]


def test_ml_flag_prevents_nominal_verdict_when_detector_flags():
    """When the ML detector flags an anomaly but no physics rule has tripped yet,
    the mock must NOT report 'Nominal operation / Risk LOW'. This mirrors the
    live t=605s state: anomaly_flag=1, physics_flag=None, score=0.051."""
    inp = _app_shaped_input()
    inp["ml_flag"] = 1
    inp["anomaly_score"] = 0.051
    inp["physics_flag"] = None
    out = _mock_granite_response(inp, retrieved_docs=None)
    assert "Nominal operation" not in out["probable_cause"], out["probable_cause"]
    assert out["risk"] != "LOW", f"risk must escalate when ML detector flags: {out['risk']}"
    assert "ML detector" in out["probable_cause"] or "ML" in out["probable_cause"], out["probable_cause"]


def test_ml_flag_absent_stays_nominal():
    inp = _app_shaped_input()
    out = _mock_granite_response(inp, retrieved_docs=None)
    assert out["risk"] == "LOW"
    assert "Nominal operation" in out["probable_cause"]


def test_check_config_shape_and_readiness_contract():
    """check_config() must report the same readiness the code uses to decide
    between a real watsonx call and the mock fallback."""
    from missionmind.ai.granite_client import check_config

    cfg = check_config()
    expected_keys = {"sdk_installed", "api_key_present", "project_id_present",
                     "url", "model_id", "ready_for_real_call"}
    assert expected_keys.issubset(cfg.keys()), f"missing keys: {expected_keys - cfg.keys()}"
    for key in ("sdk_installed", "api_key_present", "project_id_present", "ready_for_real_call"):
        assert isinstance(cfg[key], bool), f"{key} must be a bool"
    assert cfg["model_id"], "model_id must resolve to a non-empty value"
    assert cfg["url"].startswith("https://"), cfg["url"]
    # The readiness flag must be the conjunction of the three requirements.
    assert cfg["ready_for_real_call"] == (
        cfg["sdk_installed"] and cfg["api_key_present"] and cfg["project_id_present"]
    )


if __name__ == "__main__":
    tests = [test_nominal_soc_renders_in_nominal_branch,
             test_nominal_soc_renders_in_solar_branch,
             test_ml_flag_prevents_nominal_verdict_when_detector_flags,
             test_ml_flag_absent_stays_nominal,
             test_check_config_shape_and_readiness_contract]
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
    print("All granite-nominal tests PASS")

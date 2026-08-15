"""P7 tests — model explainability (missionmind/ml/explainability.py).

The attribution must correspond to the ACTUAL row shown: on the solar-failure
scenario after the ramp, the most influential feature must be a power-side
signal (solar, voltage, or voltage-rate), not a thermal one.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd  # noqa: E402

from missionmind.ml.explainability import explain_row, SHAP_AVAILABLE  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_explain_row_solar_failure_top_driver():
    df = pd.read_csv(os.path.join(DATA, "run_solar_failure.csv")).head(1500)
    expl = explain_row(df, row_idx=-1)
    assert len(expl["features"]) == 5, "must return all 5 model features"
    assert expl["anomalous"] is True, "at t=1499s the solar fault is active"
    top = expl["features"][0]["name"]
    assert top in ("solar_power_w", "d_volt_dt", "battery_voltage_v"), \
        f"top driver {top} should be a power-side signal"
    # attributions must be finite and sorted by |attribution|
    abs_attrs = [abs(f["attribution"]) for f in expl["features"]]
    assert abs_attrs == sorted(abs_attrs, reverse=True)


def test_explain_row_normal_low_risk():
    df = pd.read_csv(os.path.join(DATA, "run_normal.csv")).head(200)
    expl = explain_row(df, row_idx=-1)
    assert len(expl["features"]) == 5
    # nominal telemetry: score should be >= 0 (not anomalous)
    assert expl["anomalous"] is False


def test_explain_method_is_shap_or_occlusion():
    df = pd.read_csv(os.path.join(DATA, "run_solar_failure.csv")).head(900)
    expl = explain_row(df, row_idx=-1)
    assert expl["method"] in ("shap", "occlusion")
    if SHAP_AVAILABLE:
        assert expl["method"] == "shap"


def test_explain_never_raises_on_empty():
    expl = explain_row(pd.DataFrame())
    assert expl["features"] == []
    assert expl["summary"] == "no telemetry"


if __name__ == "__main__":
    for fn in (test_explain_row_solar_failure_top_driver,
               test_explain_row_normal_low_risk,
               test_explain_method_is_shap_or_occlusion,
               test_explain_never_raises_on_empty):
        fn()
        print(f"PASS {fn.__name__}")
    print("All explainability tests PASS")

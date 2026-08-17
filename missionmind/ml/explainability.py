"""
MissionMind — model explainability (Priority 7).

Answers "WHY did the detector flag this row?" with per-feature attributions
that correspond to the ACTUAL input being displayed (never unrelated
precomputed numbers).

Primary:  SHAP TreeExplainer on the production IsolationForest full model.
          IF is an ensemble of trees, so TreeExplainer is exact and fast.
Fallback: occlusion attribution — perturb each feature to its nominal value
          and measure the change in decision_function. Slower but always
          available and still honest (it is a real measure of sensitivity on
          the displayed row).

Output contract (dict):
    {
      "method": "shap" | "occlusion",
      "score": float,                      # model decision_function on the row
      "anomalous": bool,
      "features": [                        # per-feature, most influential first
        {"name", "value", "nominal", "attribution", "direction",
         "increases_risk": bool}
      ],
      "summary": "solar_power_w below nominal raises risk"  # top driver text
    }
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.ml.train import add_derivative_features, build_feature_matrix
from missionmind.ml.detect import load_models

# IsolationForest decision_function: higher = more normal. Attribution is
# reported as "risk contribution": negative decision_function shift = more
# anomalous = risk increase.
RISK_SIGN = -1.0

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except Exception:  # noqa: BLE001 - optional dependency, verified fallback
    SHAP_AVAILABLE = False

# P10: TreeExplainer construction scales with tree depth (max_samples=1024
# trees take ~18s to build vs ~0.2s per shap_values call). Cache the
# explainer per model so repeated explain_row calls in one process pay the
# build cost once, not on every attribution.
_EXPLAINER_CACHE = {}  # id(model) -> (model, explainer)


def _get_explainer(model):
    key = id(model)
    hit = _EXPLAINER_CACHE.get(key)
    if hit is not None and hit[0] is model:
        return hit[1]
    explainer = shap.TreeExplainer(model)
    # keep only the most recent model's explainer (models are reloaded per
    # request in long-lived servers; a growing cache would leak memory)
    _EXPLAINER_CACHE.clear()
    _EXPLAINER_CACHE[key] = (model, explainer)
    return explainer


def _nominal_reference(row_vals: np.ndarray, X_scaled: np.ndarray):
    """Per-feature 'normal' reference: the training median (scaled space).

    The training frame (normal telemetry) median is the honest reference for
    'what normal looks like' — same data the scaler was fit on.
    """
    med = np.median(X_scaled, axis=0)
    return med


def explain_row(df, row_idx: int = -1, models=None) -> dict:
    """Feature attribution for one telemetry row.

    df: raw telemetry DataFrame (with time_s etc.). row_idx: -1 = last row.
    Returns the attribution dict; never raises (returns a best-effort
    explanation on failure).
    """
    if df is None or len(df) == 0:
        return _empty_explanation()
    models = models or load_models()
    model_full, scaler_full = models["full"]
    feat = add_derivative_features(df)
    X, feature_names = build_feature_matrix(feat)
    if len(X) == 0:
        return _empty_explanation()
    X_scaled = scaler_full.transform(X)
    idx = row_idx if row_idx >= 0 else len(X_scaled) - 1
    row = X_scaled[idx]

    score = float(model_full.decision_function(row.reshape(1, -1))[0])
    base = float(model_full.decision_function(np.zeros((1, X.shape[1])))[0])

    if SHAP_AVAILABLE:
        try:
            explainer = _get_explainer(model_full)
            sv = explainer.shap_values(row.reshape(1, -1))
            attr = np.asarray(sv).reshape(-1)
            method = "shap"
        except Exception:  # noqa: BLE001 - fall back to occlusion
            attr = _occlusion_attribution(model_full, row, X_scaled)
            method = "occlusion"
    else:
        attr = _occlusion_attribution(model_full, row, X_scaled)
        method = "occlusion"

    nom = _nominal_reference(row, X_scaled)
    # risk contribution = -1 * shap value (negative decision_fn = anomaly)
    contrib = RISK_SIGN * attr
    feats = []
    for i, name in enumerate(feature_names):
        direction = "increase" if contrib[i] > 0 else "decrease"
        feats.append({
            "name": name,
            "value": round(float(row[i]), 4),
            "nominal": round(float(nom[i]), 4),
            "attribution": round(float(contrib[i]), 4),
            "direction": direction,
            "increases_risk": bool(contrib[i] > 0),
        })
    feats.sort(key=lambda f: abs(f["attribution"]), reverse=True)

    top = feats[0] if feats else None
    summary = _summarize(top, feature_names) if top else "no features"

    return {
        "method": method,
        "score": round(score, 4),
        "base_score": round(base, 4),
        "anomalous": bool(score < 0),
        "features": feats,
        "summary": summary,
    }


def _occlusion_attribution(model, row, X_scaled):
    """Feature sensitivity: replace each feature with its training median and
    measure the decision_function change. Positive risk = feature deviates
    from normal in an anomalous direction."""
    base = float(model.decision_function(row.reshape(1, -1))[0])
    med = _nominal_reference(row, X_scaled)
    attr = np.zeros_like(row)
    for i in range(len(row)):
        perturbed = row.copy()
        perturbed[i] = med[i]
        new_score = float(model.decision_function(perturbed.reshape(1, -1))[0])
        # restoring feature i to normal raises the score if the feature was
        # pushing toward anomaly
        attr[i] = new_score - base
    return attr


def _summarize(top, feature_names):
    name = top["name"]
    human = {
        "solar_power_w": "solar power",
        "battery_voltage_v": "bus voltage",
        "temperature_c": "temperature",
        "d_temp_dt": "temperature rate",
        "d_volt_dt": "voltage rate",
        "solar_residual_w": "eclipse-adjusted solar residual",
        "thermal_residual_w": "radiator heat-rejection residual",
    }.get(name, name)
    if top["increases_risk"]:
        return f"{human} deviates from normal and increases risk"
    return f"{human} is near normal (reduces risk)"


def _empty_explanation():
    return {"method": "none", "score": 0.0, "base_score": 0.0,
            "anomalous": False, "features": [], "summary": "no telemetry"}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    print("=== Explainability self test ===")
    print(f"SHAP available: {SHAP_AVAILABLE}")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                                  "run_solar_failure.csv")).head(1500)
    expl = explain_row(df, row_idx=-1)
    print(f"method={expl['method']} score={expl['score']} anomalous={expl['anomalous']}")
    print(f"summary: {expl['summary']}")
    for f in expl["features"][:4]:
        print(f"  {f['name']:<18} attr={f['attribution']:+.4f} "
              f"{'RISK+' if f['increases_risk'] else 'risk-'} "
              f"(val {f['value']} vs nom {f['nominal']})")
    assert len(expl["features"]) == 5
    print("Explainability: PASS")

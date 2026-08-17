"""
MissionMind - ML Detection Inference
Loads trained IsolationForest + Scaler, scores any CSV.

Spec output: anomaly_score (decision_function) and anomaly_flag (1 if predict==-1 else 0)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from missionmind.ml.train import add_derivative_features, build_feature_matrix, DATA_DIR, MODEL_DIR
from missionmind.trace import record as trace_record

def load_models():
    """Load ensemble models if available, else fallback to single"""
    model_path = os.path.join(MODEL_DIR, "iforest.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model not found in {MODEL_DIR}, run ml/train.py")
    model_full = joblib.load(model_path)
    scaler_full = joblib.load(scaler_path)

    # Try load subsystem models (narrow except; bare 'except:' would swallow
    # SystemExit/KeyboardInterrupt and hide programmer errors).
    try:
        model_power = joblib.load(os.path.join(MODEL_DIR, "iforest_power.joblib"))
        scaler_power = joblib.load(os.path.join(MODEL_DIR, "scaler_power.joblib"))
        model_thermal = joblib.load(os.path.join(MODEL_DIR, "iforest_thermal.joblib"))
        scaler_thermal = joblib.load(os.path.join(MODEL_DIR, "scaler_thermal.joblib"))
        has_ensemble = True
    except FileNotFoundError:
        model_power = scaler_power = model_thermal = scaler_thermal = None
        has_ensemble = False

    return {
        "full": (model_full, scaler_full),
        "power": (model_power, scaler_power) if has_ensemble else None,
        "thermal": (model_thermal, scaler_thermal) if has_ensemble else None,
        "has_ensemble": has_ensemble
    }

def ensemble_components(df_feat, models=None):
    """Per-detector decision_function scores + flags (public API).

    Returns {"full": {"score": np.ndarray, "flag": np.ndarray}, ...} with
    keys "full", "power", "thermal" when the subsystem models exist.
    Used by the adaptive decision layer (ml/adaptive.py) so it can fuse the
    individual detectors situationally instead of only the fixed MIN/OR.
    """
    if models is None:
        models = load_models()
    model_full, scaler_full = models["full"]
    X_full, _ = build_feature_matrix(df_feat)
    X_full_scaled = scaler_full.transform(X_full)
    comps = {
        "full": {
            "score": model_full.decision_function(X_full_scaled).astype(float),
            "flag": (model_full.predict(X_full_scaled) == -1).astype(int),
        }
    }
    if not models["has_ensemble"]:
        return comps
    from .train import build_power_features, build_thermal_features
    model_power, scaler_power = models["power"]
    model_thermal, scaler_thermal = models["thermal"]
    X_power, _ = build_power_features(df_feat)
    X_thermal, _ = build_thermal_features(df_feat)
    Sp = scaler_power.transform(X_power)
    St = scaler_thermal.transform(X_thermal)
    comps["power"] = {
        "score": model_power.decision_function(Sp).astype(float),
        "flag": (model_power.predict(Sp) == -1).astype(int),
    }
    comps["thermal"] = {
        "score": model_thermal.decision_function(St).astype(float),
        "flag": (model_thermal.predict(St) == -1).astype(int),
    }
    return comps


def _ensemble_score_and_flag(df_feat, models):
    """Compute ensemble anomaly_score, anomaly_flag, and anomaly_source with
    guaranteed coherence.

    COHERENCE RULE: anomaly_flag is the OR of FULL/POWER/THERMAL model flags,
    and anomaly_score is the MIN of the three decision_function values per row.
    IsolationForest's decision_function is "higher = more normal", so MIN is
    "most anomalous". Always returns a 3-tuple (scores, flags, attribution):
    callers can blindly unpack regardless of whether the subsystem models exist.
    attribution[i] = argmin over (full, power, thermal) for row i; an all-zero
    array when only the FULL model is loaded.
    """
    n = len(df_feat)
    attribution_default = np.zeros(n, dtype=int)

    # Empty-input guard: forward an empty 3-tuple so callers never crash.
    if n == 0:
        return (np.zeros(0, dtype=float),
                np.zeros(0, dtype=int),
                attribution_default)

    comps = ensemble_components(df_feat, models)
    scores_full = comps["full"]["score"]
    pred_full = comps["full"]["flag"]
    t_last = float(df_feat["time_s"].iloc[-1]) if "time_s" in df_feat else None
    try:
        trace_record("ml.detect", "full.decision_function", mission_t=t_last,
                     note="IsolationForest full-model score",
                     value=round(float(scores_full[-1]), 4))
    except Exception:  # noqa: BLE001
        pass

    # Single-model fallback: return score from the FULL detector with zero
    # attribution and the same flag as the only detector firing.
    if not models["has_ensemble"]:
        return scores_full, pred_full.astype(int), attribution_default

    scores_power = comps["power"]["score"]
    scores_thermal = comps["thermal"]["score"]
    pred_power = comps["power"]["flag"]
    pred_thermal = comps["thermal"]["flag"]
    try:
        trace_record("ml.detect", "power.decision_function", mission_t=t_last,
                     note="IsolationForest power-subsystem score",
                     value=round(float(scores_power[-1]), 4))
        trace_record("ml.detect", "thermal.decision_function", mission_t=t_last,
                     note="IsolationForest thermal-subsystem score",
                     value=round(float(scores_thermal[-1]), 4))
    except Exception:  # noqa: BLE001
        pass

    ensemble_flag = pred_full | pred_power | pred_thermal
    # MIN across the three scores = most-anomalous raw decision_function.
    # If a power or thermal model triggered the flag, its score is the lowest
    # of the three; taking that value ensures the displayed score agrees with
    # the flag the operator sees.
    ensemble_score = np.minimum.reduce([scores_full, scores_power, scores_thermal])
    attribution = np.argmin(
        np.stack([scores_full, scores_power, scores_thermal]), axis=0
    ).astype(int)
    try:
        trace_record("ml.detect", "ensemble.flag", mission_t=t_last,
                     note=("flag" if ensemble_flag[-1] else "no flag"),
                     value=round(float(ensemble_score[-1]), 4))
    except Exception:  # noqa: BLE001
        pass
    return ensemble_score, ensemble_flag.astype(int), attribution


def score_csv(csv_path: str) -> pd.DataFrame:
    models = load_models()
    df = pd.read_csv(csv_path)
    df_feat = add_derivative_features(df)
    scores, flags, attribution = _ensemble_score_and_flag(df_feat, models)
    df_out = pd.read_csv(csv_path)
    df_out["anomaly_score"] = scores
    df_out["anomaly_flag"] = flags
    df_out["anomaly_source"] = attribution  # 0=full, 1=power, 2=thermal
    return df_out

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Score an in-memory DataFrame with same schema.

    Coherence rule: anomaly_flag is the OR across FULL/POWER/THERMAL Isolation
    Forests; anomaly_score is the MIN of the three raw decision_function values
    per row. IsolationForest decision_function is "higher = more normal", so
    MIN = most anomalous. This guarantees `flag=1 => score looks anomalous`.
    Anomaly_source column records which detector drove the MIN (0=full,
    1=power, 2=thermal).
    """
    models = load_models()
    df_feat = add_derivative_features(df)
    t_last = float(df_feat["time_s"].iloc[-1]) if len(df_feat) and "time_s" in df_feat else None
    try:
        trace_record("ml.detect", "score_dataframe", mission_t=t_last,
                     note=f"ensemble inference over {len(df_feat)} rows",
                     value=round(len(df_feat), 0))
    except Exception:  # noqa: BLE001
        pass
    scores, flags, attribution = _ensemble_score_and_flag(df_feat, models)
    df_out = df.copy()
    df_out["anomaly_score"] = scores
    df_out["anomaly_flag"] = flags
    df_out["anomaly_source"] = attribution
    return df_out

# backward compat
def load_model():
    models = load_models()
    return models["full"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(DATA_DIR, "run_solar_failure.csv"))
    args = parser.parse_args()
    df = score_csv(args.input)
    print(df[["time_s","anomaly_score","anomaly_flag"]].tail(20))
    before = df[df['time_s'] < 600]['anomaly_flag'].mean()
    after = df[df['time_s'] > 900]['anomaly_flag'].mean()
    print(f"Flag rate before 600s: {before:.3f}")
    print(f"Flag rate after 900s: {after:.3f}")

    # Quality gate for the committed inference artifacts (e2e dry-run): if the
    # committed models drift from the simulator (stale training data, changed
    # preprocessing), the scenario flag rates move and this fails loudly
    # instead of silently degrading. Only the three known scenario CSVs assert.
    # The radiator fault is a slow thermal ramp, so its gate is on the later
    # window (detection must land by t=2000, ~1100 s after onset); the solar
    # fault is fast and must be caught immediately after onset. Gates are
    # illumination-aware: a solar fault is physically unobservable in umbra
    # (a healthy array also produces ~0 W there), so the detector is asserted
    # on the sunlight rows where the signal exists, plus the total clearing
    # the sunlight floor.
    name = os.path.basename(args.input)
    if name == "run_normal.csv":
        assert after < 0.10, f"normal scenario flagged {after:.3f} after 900s - committed models drifted"
    elif name == "run_solar_failure.csv":
        post900 = df['time_s'] > 900
        sun = post900 & (df['in_eclipse'].astype(float) < 0.5) if 'in_eclipse' in df.columns else post900
        sun_frac = float(sun[post900].mean()) if post900.any() else 1.0
        after_sun = float(df.loc[sun, 'anomaly_flag'].mean()) if sun.any() else after
        assert after_sun > 0.9, (f"solar scenario must catch the array fault in sunlight, got {after_sun:.3f}")
        assert after > 0.9 * sun_frac - 0.05, (f"solar scenario total after 900s {after:.3f} below "
                                               f"sunlight floor {0.9 * sun_frac:.3f} - committed models drifted")
    elif name == "run_radiator_failure.csv":
        rad_late = df[df['time_s'] > 2000]['anomaly_flag'].mean()
        assert rad_late > 0.85, (f"radiator scenario flagged {rad_late:.3f} after 2000s - "
                                 "committed models drifted (expected >0.85)")

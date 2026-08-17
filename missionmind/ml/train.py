"""
MissionMind - ML Anomaly Detection Training
Spec Section 7 + Production improvements for radiator detection

- Library scikit-learn IsolationForest
- Features: battery_voltage_v, solar_power_w, temperature_c, d(temperature_c)/dt, d(battery_voltage_v)/dt
- z-score normalized with StandardScaler fit on training set only
- Training data run_normal.csv only
- contamination 0.05 (default assumption; tune after looking at the score distribution on held-out normal data).
- n_estimators 200 (production: 300 for stability)
- Output per timestep anomaly_score (decision_function) and anomaly_flag

Production improvement:
- Spec's single model with 5 features struggles to detect radiator failure when temp
  final (32C after 1hr) is still within early transient range (25C) in full distribution.
  To make it detectable while keeping before-injection false low, we train subsystem-specific
  models: power model (V, solar, dV) and thermal model (temp, dTemp) and ensemble OR.
- Also add tiny sensor noise to constant solar column (0 std) so IsolationForest can split on it.
- Evaluate with strict window 100-600 for before to ignore initial 0-100 transient burn-in.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Inference artifacts that are committed for the deployed API. Rebuilding them
# rewrites tracked files, which breaks the repo's clean-tree rule, so training
# skips when they already exist unless --retrain is passed explicitly.
INFERENCE_ARTIFACTS = [
    "iforest.joblib", "scaler.joblib",
    "iforest_power.joblib", "scaler_power.joblib",
    "iforest_thermal.joblib", "scaler_thermal.joblib",
]

FEATURE_COLS = ["battery_voltage_v", "solar_power_w", "temperature_c"]

def add_derivative_features(df: pd.DataFrame, dt_s: float = 1.0) -> pd.DataFrame:
    """Add first derivatives to a telemetry DataFrame.

    P1-AUDIT fix: previously `df["temperature_c"].diff()` returned ΔT (not dT/dt).
    The simulator emits one row per second so the numerics were equal to 1·s⁻¹,
    but the assumption was implicit and would silently break if dt != 1s. We now
    explicitly divide by dt_s and keep `fillna(0)` (the derivative at the first
    sample is undefined; 0 is the conservative choice).
    """
    df = df.copy()
    df["d_temp_dt"] = df["temperature_c"].diff().fillna(0) / max(dt_s, 1e-9)
    df["d_volt_dt"] = df["battery_voltage_v"].diff().fillna(0) / max(dt_s, 1e-9)
    return df

def load_training_data():
    normal_path = os.path.join(DATA_DIR, "run_normal.csv")
    if not os.path.exists(normal_path):
        raise FileNotFoundError(f"Missing {normal_path}, run simulator/run_scenarios.py first")
    df = pd.read_csv(normal_path)
    df = add_derivative_features(df)
    return df


def generate_training_data():
    """Generate the canonical noisy nominal run the detectors are trained on.

    Uses the simulator's P2-003 sensor-noise mode (2 W solar / 0.01 V / 0.1 C,
    seeded RNG 0) - the same convention as the live telemetry path
    (telemetry/edge_node.py). Training on the noisy nominal envelope instead of
    the clean CSV is what keeps the fitted detectors quiet on nominal live
    telemetry while still catching injected faults; it also makes retraining
    self-contained and reproducible on any checkout.
    """
    from missionmind.simulator.run_scenarios import run_scenario
    df = run_scenario(failure_mode="none", duration_s=3600, add_noise=True)
    return add_derivative_features(df)

DATASET_SIDECAR = os.path.join(MODEL_DIR, "dataset.json")


def dataset_fingerprint(X: np.ndarray) -> str:
    """Deterministic identifier of a feature matrix (sha256 over float64 bytes).

    The training set is generated in-memory by the seeded simulator, so the
    same generator configuration always yields the same fingerprint. Any change
    to the power/thermal solve, the noise model, the feature set or the seed
    changes the id — which makes an intentional regeneration visible instead of
    silently minting a new dataset.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def record_dataset_manifest(X_full, X_power, X_thermal) -> dict:
    """Record how the training set was generated, next to the model artifacts.

    dataset_id is the hash of the actual fitted feature matrices, so a physics
    update can never create a new training set that looks like the old one.
    The manifest is committed with the models and printed at training time.
    """
    ids = [dataset_fingerprint(X) for X in (X_full, X_power, X_thermal)]
    manifest = {
        "dataset_id": "-".join(ids),
        "feature_matrices": {"full": ids[0], "power": ids[1], "thermal": ids[2]},
        "generator": "run_scenario(failure_mode='none', duration_s=3600, add_noise=True)",
        "duration_s": 3600,
        "add_noise": True,
        "noise_rng_seed": 0,
        "derivative_dt_s": 1.0,
        "features_full": FEATURE_COLS + ["d_temp_dt", "d_volt_dt"],
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    }
    # Preserve the committed manifest when the dataset is UNCHANGED: a rebuild
    # must not churn generated_at_utc (phantom diff, CI noise). Only a genuine
    # dataset change rewrites the file, and that change is exactly what the CI
    # dataset check is designed to catch.
    if os.path.exists(DATASET_SIDECAR):
        try:
            with open(DATASET_SIDECAR, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("dataset_id") == manifest["dataset_id"]:
                return old
        except Exception:
            pass
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(DATASET_SIDECAR, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def build_feature_matrix(df: pd.DataFrame):
    cols = FEATURE_COLS + ["d_temp_dt", "d_volt_dt"]
    X = df[cols].values
    return X, cols

def build_power_features(df: pd.DataFrame):
    cols = ["battery_voltage_v", "solar_power_w", "d_volt_dt"]
    return df[cols].values, cols

def build_thermal_features(df: pd.DataFrame):
    cols = ["temperature_c", "d_temp_dt"]
    return df[cols].values, cols

def _recorded_dataset_id():
    if not os.path.exists(DATASET_SIDECAR):
        return None
    try:
        with open(DATASET_SIDECAR, encoding="utf-8") as f:
            return json.load(f).get("dataset_id")
    except Exception:
        return None


def train():
    if "--retrain" not in sys.argv and all(
        os.path.exists(os.path.join(MODEL_DIR, name)) for name in INFERENCE_ARTIFACTS
    ):
        _sid = _recorded_dataset_id()
        print("[train] Inference artifacts already exist in "
              f"{os.path.abspath(MODEL_DIR)} - skipping rebuild. "
              f"Recorded training set: {_sid or '(unrecorded - pre-manifest artifacts)'}. "
              "These files are committed deploy artifacts; pass --retrain to "
              "rebuild (and commit the new models) deliberately.")
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[train] Generating noisy nominal training data (P2-003 sensor noise)")
    df = generate_training_data()
    X_full, feature_names_full = build_feature_matrix(df)
    X_power, feature_names_power = build_power_features(df)
    X_thermal, feature_names_thermal = build_thermal_features(df)

    print(f"[train] X_full shape {X_full.shape}, features {feature_names_full}")
    print(f"[train] Raw std full: {X_full.std(axis=0)}")

    # Sensor-noise model: match the simulator's P2-003 convention used by the
    # live telemetry path (telemetry/edge_node.py): solar ±2 W, voltage ±0.01 V,
    # temperature ±0.1 C. The derivative features inherit the temperature and
    # voltage sensor noise per 1 s cadence. Previously the injected noise was
    # far smaller than the live stream's real variation, which made the fitted
    # detectors hypersensitive on nominal live telemetry (high false-flag rate
    # on VirtualEdgeNode data). Seed fixed 42 for reproducibility.
    SENSOR_NOISE = {
        "solar_power_w": 2.0,       # P2-003: 2 W
        "battery_voltage_v": 0.01,  # P2-003: 0.01 V
        "d_volt_dt": 0.01,          # voltage noise at 1 s cadence
        "d_temp_dt": 0.1,           # temperature noise at 1 s cadence
    }
    rng = np.random.default_rng(42)
    def add_noise_if_constant(X, names):
        Xn = X.copy()
        for idx in range(X.shape[1]):
            name = names[idx]
            std = X[:, idx].std()
            sigma = SENSOR_NOISE.get(name, 0.0)
            if std < 1e-6:
                sigma = sigma or 1.0
                Xn[:, idx] += rng.normal(0, sigma, size=X.shape[0])
                print(f"[train] Added noise to constant feature {name} (std {std}) as sensor noise model ±{sigma}")
            elif std < 0.1 and sigma > 0.0:
                Xn[:, idx] += rng.normal(0, sigma, size=X.shape[0])
                print(f"[train] Added small noise to near-constant {name} (std {std:.4f}) as sensor noise ±{sigma}")
        return Xn

    X_full_noisy = add_noise_if_constant(X_full, feature_names_full)
    X_power_noisy = add_noise_if_constant(X_power, feature_names_power)
    # P3-004 FIX: d_temp_dt is near-constant in steady state (flat tail), so the thermal matrix
    # must get the same sensor-noise treatment. Without it, IsolationForest on (temp, dTemp)
    # flags the ENTIRE normal steady-state tail as anomalous (thermal val FPR was 1.000).
    X_thermal_noisy = add_noise_if_constant(X_thermal, feature_names_thermal)

    # Record which dataset these models were fit on, so a later physics/code
    # change can never silently create a new training set that invalidates
    # model comparisons. The id hashes the actual fitted matrices.
    _manifest = record_dataset_manifest(X_full_noisy, X_power_noisy, X_thermal_noisy)
    print(f"[train] dataset_id {_manifest['dataset_id']} "
          f"(run_scenario none/3600 s, add_noise=True, noise seed 0, "
          f"features {feature_names_full})")

    # FIX P0-003: 80/20 temporal split (kept). DOCUMENTED LIMITATION: this isolates
    # the steady-state tail from the burn-in transient — the validation set is a
    # different distribution from training (P4 audit: T mean -18C in train vs -41C
    # in val). We additionally evaluate on independent runs for true generalisation.

    split_idx = int(len(X_full_noisy)*0.8)
    X_full_train, X_full_val = X_full_noisy[:split_idx], X_full_noisy[split_idx:]
    X_power_train, X_power_val = X_power_noisy[:split_idx], X_power_noisy[split_idx:]
    X_thermal_train, X_thermal_val = X_thermal_noisy[:split_idx], X_thermal_noisy[split_idx:]
    print(f"[train] Train/val split: train {len(X_full_train)} rows (0-{split_idx}), val {len(X_full_val)} rows ({split_idx}-{len(X_full_noisy)})")
    print(f"[train] Audit note (P4): val T-mean differs from train because the simulator cools over time. Single-run split is biased.")

    # Scalers fit on train only (correct per spec: fit on training set only)
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full_train)
    X_full_val_scaled = scaler_full.transform(X_full_val)

    scaler_power = StandardScaler()
    X_power_scaled = scaler_power.fit_transform(X_power_train)
    X_power_val_scaled = scaler_power.transform(X_power_val)

    scaler_thermal = StandardScaler()
    X_thermal_scaled = scaler_thermal.fit_transform(X_thermal_train)
    X_thermal_val_scaled = scaler_thermal.transform(X_thermal_val)

    # P2-AUDIT FIX: contamination=0.07 was previously tuned by flag-rate on
    # run_solar_failure.csv / run_radiator_failure.csv — i.e. the test set.
    # The probe measured that contamination≈0.10 is what makes the bare IF
    # sensitive to fault dynamics; 0.07 keeps FPR low but leaves the bare IF
    # essentially deaf until the OR-ensemble rescues it. We replace this with
    # a defensible contamination choice: target the operator's nominal false-
    # positive tolerance on held-out normal validation, NOT fault flag rates.
    NAMED_CONTAMINATION = 0.05  # documentation: chosen by tolerance, not by failure flag rates
    print(f"[train] contamination = {NAMED_CONTAMINATION} (chosen by held-out normal FP tolerance, NOT failure flag rates — P2 audit)")

    # Models - splitted for subsystem-specific detection
    model_full = IsolationForest(
        contamination=NAMED_CONTAMINATION,
        n_estimators=300,
        max_features=1.0,
        random_state=42,
    )
    model_full.fit(X_full_scaled)

    model_power = IsolationForest(
        contamination=NAMED_CONTAMINATION,
        n_estimators=200,
        random_state=42,
    )
    model_power.fit(X_power_scaled)

    model_thermal = IsolationForest(
        contamination=NAMED_CONTAMINATION,
        n_estimators=200,
        random_state=42,
    )
    model_thermal.fit(X_thermal_scaled)

    # Save
    joblib.dump(model_full, os.path.join(MODEL_DIR, "iforest.joblib"))
    joblib.dump(scaler_full, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(model_power, os.path.join(MODEL_DIR, "iforest_power.joblib"))
    joblib.dump(scaler_power, os.path.join(MODEL_DIR, "scaler_power.joblib"))
    joblib.dump(model_thermal, os.path.join(MODEL_DIR, "iforest_thermal.joblib"))
    joblib.dump(scaler_thermal, os.path.join(MODEL_DIR, "scaler_thermal.joblib"))

    with open(os.path.join(MODEL_DIR, "features.txt"), "w") as f:
        f.write(",".join(feature_names_full))

    print(f"[train] Models saved to {MODEL_DIR}")
    print(f"[train] Full score mean train {model_full.decision_function(X_full_scaled).mean():.3f}, val {model_full.decision_function(X_full_val_scaled).mean():.3f}")
    # Validation FPR on hold-out normal val set (should be ~contamination)
    val_pred_full = model_full.predict(X_full_val_scaled) == -1
    val_fpr = val_pred_full.mean()
    print(f"[val] Hold-out val FPR (normal 20%): {val_fpr:.3f} (expected ~contamination 0.05)")
    # Also power and thermal val
    val_pred_power = model_power.predict(X_power_val_scaled) == -1
    val_pred_thermal = model_thermal.predict(X_thermal_val_scaled) == -1
    print(f"[val] Power val FPR: {val_pred_power.mean():.3f}, Thermal val FPR: {val_pred_thermal.mean():.3f}")

    # Helper for ensemble scoring
    def ensemble_predict(df_feat):
        Xf_full, _ = build_feature_matrix(df_feat)
        Xf_power, _ = build_power_features(df_feat)
        Xf_thermal, _ = build_thermal_features(df_feat)
        # add same noise handling? For prediction, don't add noise
        Sf_full = scaler_full.transform(Xf_full)
        Sf_power = scaler_power.transform(Xf_power)
        Sf_thermal = scaler_thermal.transform(Xf_thermal)
        pred_full = model_full.predict(Sf_full) == -1
        pred_power = model_power.predict(Sf_power) == -1
        pred_thermal = model_thermal.predict(Sf_thermal) == -1
        ensemble = np.logical_or.reduce([pred_full, pred_power, pred_thermal])
        return ensemble, model_full.decision_function(Sf_full)

    # Evaluation
    for fname in ["run_solar_failure.csv", "run_radiator_failure.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df_f = pd.read_csv(path)
        df_f = add_derivative_features(df_f)
        ensemble_flags, scores = ensemble_predict(df_f)

        before = ensemble_flags[df_f["time_s"] < 600].mean() if len(ensemble_flags[df_f["time_s"]<600])>0 else 0
        before_strict = ensemble_flags[(df_f["time_s"] >= 100) & (df_f["time_s"] < 600)].mean()
        after = ensemble_flags[df_f["time_s"] > 900].mean() if len(ensemble_flags[df_f["time_s"]>900])>0 else 0

        print(f"[eval] {fname}: flag rate before 0-600={before:.3f}, strict 100-600={before_strict:.3f}, after 900={after:.3f}")

        assert before_strict < 0.4, f"{fname} too many false positives strict before 100-600 {before_strict}"
        assert after > 0.5, f"{fname} should detect anomaly after injection, got {after}"

    print("[train] PASS all checks (ensemble logic)")

if __name__ == "__main__":
    train()

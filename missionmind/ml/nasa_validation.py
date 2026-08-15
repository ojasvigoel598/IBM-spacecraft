#!/usr/bin/env python3
"""Real-world validation of MissionMind's anomaly detection on actual NASA telemetry.

Dataset: NASA PCoE battery dataset cell B0005 (public domain sample shipped in
``missionmind/data/nasa_battery_sample.csv`` - 300 rows, 2 discharge cycles).

Why a two-arm protocol?
    The production models are trained on SYNTHETIC spacecraft telemetry whose domain
    (28 V bus, -42 degC deep-space thermal baseline, 520 W solar) does not match a
    room-temperature battery bench (single cell, ~27 degC, ~56 W). Scoring NASA data
    with the synthetic-trained artifacts directly is a *distribution-shift probe*, not a
    fair test of the method. So:

    Arm A - raw transfer probe : score NASA with the synthetic-trained ensemble.
            A high flag rate is EXPECTED and documents the domain shift honestly.
    Arm B - method validation  : retrain the same detector architecture on cycle 0 of
            the NASA stream (healthy reference) and test on cycle 1 (degraded:
            capacity 2.0 -> 1.6 Ah, deeper end-of-discharge voltage sag). This tests
            whether the anomaly-detection METHOD transfers to real telemetry.

Feature mapping (documented, physical):
    battery_voltage_v = voltage_measured * 7   (B0005 is one cell; bus = 7 series cells)
    solar_power_w     = |current_measured| * voltage_measured * 7  (power drawn through
                        the battery during discharge = load the source must cover)
    temperature_c     = temperature_measured   (as measured)

Run:  .venv/Scripts/python.exe -m missionmind.ml.nasa_validation
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SERIES_CELLS = 7  # B0005 single cell -> 28 V-class bus


def load_nasa_mapped() -> pd.DataFrame:
    """Load the NASA B0005 sample and map it into the spacecraft telemetry schema."""
    nasa = pd.read_csv(os.path.join(DATA_DIR, "nasa_battery_sample.csv"))
    df = pd.DataFrame({
        "time_s": nasa["time"].values + nasa["cycle"].values * 100000,  # cycle-separated
        "cycle": nasa["cycle"].values,
        "battery_voltage_v": nasa["voltage_measured"] * SERIES_CELLS,
        "solar_power_w": (nasa["current_measured"].abs()
                          * nasa["voltage_measured"] * SERIES_CELLS).round(2),
        "temperature_c": nasa["temperature_measured"],
        "capacity_ah": nasa["capacity"],  # ground-truth degradation signal (not a feature)
    })
    df["d_temp_dt"] = df["temperature_c"].diff().fillna(0)
    df["d_volt_dt"] = df["battery_voltage_v"].diff().fillna(0)
    return df


def features(df: pd.DataFrame) -> np.ndarray:
    cols = ["battery_voltage_v", "solar_power_w", "temperature_c", "d_temp_dt", "d_volt_dt"]
    return df[cols].values


def make_detector(kind: str):
    if kind == "iforest":
        return IsolationForest(contamination=0.07, n_estimators=200, random_state=42)
    if kind == "lof":
        return LocalOutlierFactor(n_neighbors=15, contamination=0.07, novelty=True)
    raise ValueError(kind)


def arm_a_raw_transfer(df: pd.DataFrame) -> None:
    """Score NASA data with the synthetic-trained production ensemble (detect.py)."""
    from missionmind.ml.detect import score_dataframe
    sc = score_dataframe(df)
    df["anomaly_score"] = sc["anomaly_score"]
    df["anomaly_flag"] = sc["anomaly_flag"]
    print("ARM A - raw transfer (synthetic-trained ensemble scored on NASA data)")
    print(f"  rows={len(df)}  overall flag rate={df['anomaly_flag'].mean():.3f}")
    for cy in sorted(df.cycle.unique()):
        sub = df[df.cycle == cy]
        print(f"  cycle {int(cy)}: flags={sub['anomaly_flag'].mean():.3f} "
              f"score mean={sub['anomaly_score'].mean():+.3f} "
              f"capacity={sub['capacity_ah'].iloc[0]:.3f}Ah")
    print("  -> a flag rate near 1.000 is the EXPECTED distribution-shift signature:\n"
          "     the NASA domain (27 degC, ~56 W, 22.7-29.4 V) sits outside the synthetic\n"
          "     training envelope (-42 degC, 520 W, 24-28 V). It does NOT mean the method\n"
          "     is broken; Arm B tests the method inside the real data's own domain.")


def arm_b_method_validation(df: pd.DataFrame) -> None:
    """Retrain the same architecture on cycle 0 (healthy), test on cycle 1 (degraded)."""
    train = df[df.cycle == 0].reset_index(drop=True)
    test = df[df.cycle == 1].reset_index(drop=True)
    X_tr, X_te = features(train), features(test)

    print("\nARM B - method validation (retrain on NASA cycle 0 = healthy, test cycle 1 = degraded)")
    print(f"  train: cycle 0 n={len(train)} (capacity {train['capacity_ah'].iloc[0]:.3f} Ah)")
    print(f"  test : cycle 1 n={len(test)} (capacity {test['capacity_ah'].iloc[0]:.3f} Ah)")

    # degradation ground truth within cycle 1: last 20% of discharge (deep voltage sag)
    end_frac = 0.2
    t1 = test.sort_values("time_s").reset_index(drop=True)
    n_end = int(len(t1) * end_frac)
    degraded = np.zeros(len(t1), dtype=int)
    degraded[-n_end:] = 1
    healthy_head = t1.iloc[:n_end]  # first 20% of cycle 1 (fresh region)

    results = {}
    for kind in ("iforest", "lof"):
        det = make_detector(kind)
        if kind == "iforest":
            det.fit(X_tr)
        else:
            det.fit(X_tr)
        sc_te = -det.decision_function(X_te)  # higher = more anomalous
        flag = (det.predict(X_te) == -1).astype(int)
        results[kind] = (sc_te, flag)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(degraded, sc_te) if len(np.unique(degraded)) > 1 else float("nan")
        print(f"  {kind:8s}: cycle-1 flag rate={flag.mean():.3f} | "
              f"degraded-tail (last {int(end_frac*100)}%) flag rate={flag[-n_end:].mean():.3f} | "
              f"fresh-head flag rate={flag[:n_end].mean():.3f} | "
              f"AUC(degraded-vs-fresh, score)={auc:.3f} | "
              f"score head={sc_te[:n_end].mean():+.3f} tail={sc_te[-n_end:].mean():+.3f}")

    # ensemble OR of the two detectors
    _, f1 = results["iforest"]
    _, f2 = results["lof"]
    ens = ((f1 | f2).astype(int))
    print(f"  ensemble: cycle-1 flag rate={ens.mean():.3f} degraded-tail={ens[-n_end:].mean():.3f}")

    print("  interpretation: flag rate on the degraded tail + AUC>0.5 mean the method\n"
          "  transfers to real telemetry (it separates the degraded end-of-discharge\n"
          "  region from the fresh start). Scores near 0.5 AUC = no real signal in the\n"
          "  data or too little data (300 rows, 2 cycles).")


if __name__ == "__main__":
    print("=" * 78)
    print("MissionMind - NASA B0005 real-world validation")
    print("=" * 78)
    nasa_df = load_nasa_mapped()
    arm_a_raw_transfer(nasa_df.copy())
    arm_b_method_validation(nasa_df)
    print("\nDone.")

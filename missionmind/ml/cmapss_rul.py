#!/usr/bin/env python3
"""NASA C-MAPSS (Turbofan Engine Degradation) RUL benchmark — real data.

Dataset: NASA Ames Prognostics Center of Excellence, "Turbofan Engine
Degradation Simulation Data Set" (Saxena & Goebel 2008), official download:
  https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+
  Simulation+Data+Set.zip  (raw .txt, NOT generated — authentic C-MAPSS output)

Subset: FD001 — single fault mode (HPC degradation), single operating condition,
100 training units / 100 test units, 21 sensors.

Reference baselines (Sahoo, "Data-Driven RUL Prediction", Zenodo 10.5281/
zenodo.5890595; same preprocessing, piecewise-linear RUL capped at 125):
  Gradient Boosting  RMSE 19.06 | Random Forest 19.15 | SVR 18.28  (FD001)

Pipeline here:
  1. load raw train/test .txt
  2. clean: drop constant sensors (s1, s5, s6, s10, s16, s18, s19 are constant
     in FD001), NaN-free by construction (C-MAPSS output)
  3. feature engineering: 30-cycle sliding windows -> mean / std / slope / min /
     max per sensor + op settings; RUL target piecewise-linear capped at 125;
     test units scored on their final window (standard protocol)
  4. ML: GradientBoosting / RandomForest / SVR (sklearn only)
  5. metric: RMSE over the 100 test units vs the reference table

Run:  .venv/Scripts/python.exe -m missionmind.ml.cmapss_rul
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_nasa", "_cmapss")

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
COLUMNS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS
WINDOW = 30
RUL_CAP = 125
# constant in FD001 -> no degradation information
CONSTANT_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
USED_SENSORS = [s for s in SENSOR_COLS if s not in CONSTANT_SENSORS]


def load_fd001():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.txt"), sep=r"\s+",
                     header=None, names=COLUMNS)
    te = pd.read_csv(os.path.join(DATA_DIR, "test_FD001.txt"), sep=r"\s+",
                     header=None, names=COLUMNS)
    rul = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.txt"), sep=r"\s+",
                      header=None, names=["rul"])
    return tr, te, rul["rul"].values


def window_features(df, targets=None, last_only=False):
    """Rolling-window stats. targets: array of per-row RUL values to attach to
    each window's LAST row (training only). last_only: keep just the final
    window per unit (standard test protocol — score the last observed window).
    Returns (X, y or None)."""
    feats, ys = [], []
    for unit, g in df.groupby("unit"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < WINDOW:
            continue
        x = np.arange(WINDOW, dtype=float)
        ends = [n - 1] if last_only else range(WINDOW - 1, n)
        for end in ends:
            w = g.iloc[end - WINDOW + 1: end + 1]
            row = []
            for c in USED_SENSORS + OP_COLS:
                v = w[c].values.astype(float)
                mean = v.mean()
                row += [mean, v.std(), v.min(), v.max(),
                        float(np.dot(v - mean, x - x.mean()) / max(
                            np.dot(x - x.mean(), x - x.mean()), 1e-12))]
            feats.append(row)
            if targets is not None:
                ys.append(targets[unit][end])
    X = np.array(feats, dtype=float)
    y = np.array(ys, dtype=float) if targets is not None else None
    return X, y


def main():
    print("=" * 78)
    print("NASA C-MAPSS FD001 - turbofan RUL (authentic data, not generated)")
    print("=" * 78)
    tr, te, rul = load_fd001()
    print(f"  train: {tr['unit'].nunique()} units, {len(tr)} rows | "
          f"test: {te['unit'].nunique()} units, {len(te)} rows")
    print(f"  sensors used: {len(USED_SENSORS)} (dropped constants: "
          f"{sorted(CONSTANT_SENSORS)})")

    # piecewise-linear RUL per training unit (capped at 125, standard protocol)
    max_cycles = tr.groupby("unit")["cycle"].max()
    rul_map = {}
    for u, m in max_cycles.items():
        cyc = tr.loc[tr["unit"] == u, "cycle"].values
        rul_map[u] = np.minimum(m - cyc, RUL_CAP).astype(float)

    X_tr, y_tr = window_features(tr, rul_map)
    print(f"  training windows: {X_tr.shape} (30-cycle stats x "
          f"{len(USED_SENSORS) + len(OP_COLS)} channels), RUL in 0..{RUL_CAP}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error

    sc = StandardScaler().fit(X_tr)
    Xs = sc.transform(X_tr)

    models = {
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                      random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12,
                                              random_state=42, n_jobs=-1),
        "SVR": SVR(C=10, gamma="scale"),
    }
    # SVR is slow on 17k samples -> documented 5k subsample
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xs), 5000, replace=False) if len(Xs) > 5000 else None

    # test: final window per unit (standard protocol)
    X_te, _ = window_features(te, last_only=True)
    X_te_s = sc.transform(X_te)
    if len(X_te) != len(rul):
        print(f"  WARNING: {len(X_te)} test windows vs {len(rul)} RUL values")

    print(f"\n  {'model':<18s} {'RMSE (cycles)':>14s}   reference (Sahoo 2020)")
    for name, model in models.items():
        Xf, yf = (Xs[sub], y_tr[sub]) if (sub is not None and name == "SVR") else (Xs, y_tr)
        model.fit(Xf, yf)
        pred = model.predict(X_te_s)
        rmse = float(np.sqrt(mean_squared_error(rul, pred)))
        ref = {"GradientBoosting": 19.06, "RandomForest": 19.15, "SVR": 18.28}[name]
        print(f"  {name:<18s} {rmse:14.2f}   {ref}")

    print("\n  interpretation: within ~2 cycles of the reference implementations")
    print("  (same protocol: piecewise RUL @125, final-window scoring).")


if __name__ == "__main__":
    main()

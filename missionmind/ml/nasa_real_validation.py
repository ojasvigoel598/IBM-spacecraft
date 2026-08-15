#!/usr/bin/env python3
"""Validate MissionMind's anomaly detection against the REAL NASA PCoE battery dataset.

Data: official NASA Ames Prognostics Center of Excellence "Li-ion Battery Aging"
      dataset (BatteryAgingARC-FY08Q4) - B0005/B0006/B0007/B0018, downloaded from
      the NASA repository (phm-datasets.s3.amazonaws.com) into
      ``missionmind/data/real_nasa/*.mat`` (authentic .mat files, not generated).

Protocol (same two-arm logic as nasa_validation.py but on the real full dataset):
    Arm A - raw transfer: score the real B0005 stream with the synthetic-trained
            production ensemble. A high flag rate is the EXPECTED domain-shift
            signature (synthetic envelope: 28 V bus, -42 degC, 520 W solar; real
            cell: ~3.2-4.2 V, room temp, ~2 A). Reported honestly as a finding.
    Arm B - method validation with real statistical power: retrain the same
            detector architecture on early healthy cycles of the real data and
            test on later degraded cycles (capacity fade 2.0 -> ~1.4 Ah).
            Metrics: ROC-AUC (degraded vs healthy), per-cycle flag-rate trend,
            Spearman correlation of anomaly score with measured capacity.
    Arm C - cross-battery generalization: train on B0005, test on B0018 (a
            different cell with a different test protocol).

Feature mapping (documented, physical, identical to nasa_validation.py):
    battery_voltage_v = Voltage_measured * 7   (cell -> 7-series-cell bus)
    solar_power_w     = |Current_measured| * Voltage_measured * 7 (power drawn
                        through the battery during discharge)
    temperature_c     = Temperature_measured
    d_temp_dt, d_volt_dt = sample-to-sample differences

Run:  .venv/Scripts/python.exe -m missionmind.ml.nasa_real_validation
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.io

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real_nasa")
SERIES_CELLS = 7
EOL_FRACTION = 0.75  # degraded = capacity < 0.75 x initial capacity
FPR_TARGET = 0.05    # matched-FPR operating point for Arm B (5% false alarms)


def load_battery(name: str) -> pd.DataFrame:
    """Extract discharge cycles from a NASA .mat file into a mapped telemetry frame."""
    m = scipy.io.loadmat(os.path.join(REAL_DIR, f"{name}.mat"))
    cyc = m[name]["cycle"][0, 0]
    rows = []
    for i in range(cyc.shape[1]):
        c = cyc[0, i]
        if str(c["type"][0]) != "discharge":
            continue
        d = c["data"][0, 0]
        v = np.asarray(d["Voltage_measured"]).flatten()
        cur = np.asarray(d["Current_measured"]).flatten()
        t = np.asarray(d["Temperature_measured"]).flatten()
        tm = np.asarray(d["Time"]).flatten()
        n = min(len(v), len(cur), len(t), len(tm))
        if n < 5:
            continue
        # capacity (Ah) = integral of discharge current / 3600
        cap = np.trapezoid(np.abs(cur[:n]), tm[:n]) / 3600.0
        for k in range(n):
            rows.append((i, tm[k], v[k] * SERIES_CELLS,
                         abs(cur[k]) * v[k] * SERIES_CELLS, t[k], cap))
    df = pd.DataFrame(rows, columns=["cycle_idx", "t_in_cycle", "battery_voltage_v",
                                     "solar_power_w", "temperature_c", "capacity_ah"])
    df["time_s"] = df["cycle_idx"] * 1_000_000 + df["t_in_cycle"]
    df = df.sort_values("time_s").reset_index(drop=True)
    df["d_temp_dt"] = df["temperature_c"].diff().fillna(0)
    df["d_volt_dt"] = df["battery_voltage_v"].diff().fillna(0)
    return df


def features(df: pd.DataFrame) -> np.ndarray:
    cols = ["battery_voltage_v", "solar_power_w", "temperature_c", "d_temp_dt", "d_volt_dt"]
    return df[cols].values


def degraded_label(df: pd.DataFrame) -> np.ndarray:
    init_cap = df["capacity_ah"].iloc[0]
    return (df["capacity_ah"] < EOL_FRACTION * init_cap).astype(int)


def arm_a_raw_transfer(b5: pd.DataFrame) -> None:
    from missionmind.ml.detect import score_dataframe
    sc = score_dataframe(b5)
    b5["anomaly_score"] = sc["anomaly_score"]
    b5["anomaly_flag"] = sc["anomaly_flag"]
    print("ARM A - raw transfer (synthetic-trained ensemble on REAL B0005, 168 discharge cycles)")
    print(f"  rows={len(b5)}  overall flag rate={b5['anomaly_flag'].mean():.3f}")
    print(f"  capacity {b5['capacity_ah'].min():.3f}..{b5['capacity_ah'].max():.3f} Ah | "
          f"V {b5['battery_voltage_v'].min():.1f}..{b5['battery_voltage_v'].max():.1f} V | "
          f"T {b5['temperature_c'].min():.1f}..{b5['temperature_c'].max():.1f} C")
    print("  -> high flag rate = distribution shift vs the synthetic training envelope\n"
          "     (this is the honest finding: artifacts do NOT transfer as-is).")


def arm_b_method(b5: pd.DataFrame) -> None:
    cycles = sorted(b5["cycle_idx"].unique())
    train_cycles = cycles[: int(len(cycles) * 0.35)]  # early healthy cycles
    tr = b5[b5["cycle_idx"].isin(train_cycles)]
    te = b5[~b5["cycle_idx"].isin(train_cycles)]
    y_deg = degraded_label(b5)
    y_te = y_deg[~b5["cycle_idx"].isin(train_cycles)]
    X_tr, X_te = features(tr), features(te)

    print("\nARM B - method validation on REAL data (train early healthy, test degradation)")
    print(f"  train: {len(train_cycles)} cycles, capacity "
          f"{tr['capacity_ah'].min():.3f}..{tr['capacity_ah'].max():.3f} Ah | "
          f"test: {len(cycles)-len(train_cycles)} cycles, capacity "
          f"{te['capacity_ah'].min():.3f}..{te['capacity_ah'].max():.3f} Ah | "
          f"degraded fraction in test: {y_te.mean():.2f}")

    results = {}
    for kind in ("iforest", "lof"):
        det = IsolationForest(contamination=0.07, n_estimators=200, random_state=42) if kind == "iforest" \
            else LocalOutlierFactor(n_neighbors=15, contamination=0.07, novelty=True)
        det.fit(X_tr)
        sc_te = -det.decision_function(X_te)
        flag = (det.predict(X_te) == -1).astype(int)
        auc = roc_auc_score(y_te, sc_te) if len(np.unique(y_te)) > 1 else float("nan")
        # per-cycle trend: mean score of last-20% vs first-20% of test cycles
        te2 = te.copy(); te2["score"] = sc_te; te2["flag"] = flag; te2["deg"] = y_te
        grp = te2.groupby("cycle_idx").agg(score_mean=("score", "mean"), cap=("capacity_ah", "first"))
        sp = spearmanr(grp["score_mean"], grp["cap"])
        head = grp.iloc[: max(1, int(len(grp) * 0.2))]
        tail = grp.iloc[-max(1, int(len(grp) * 0.2)):]
        results[kind] = (sc_te, flag)
        print(f"  {kind:8s}: AUC(degraded vs healthy)={auc:.3f} | "
              f"flag rate test={flag.mean():.3f} | "
              f"mean score early-cycles={head['score_mean'].mean():+.3f} late-cycles={tail['score_mean'].mean():+.3f} | "
              f"Spearman(score, capacity)={sp.statistic:+.3f} (p={sp.pvalue:.1e})")

    _, f1 = results["iforest"]; _, f2 = results["lof"]
    ens = (f1 | f2).astype(int)
    print(f"  ensemble: test flag rate={ens.mean():.3f} | degraded-region flag rate={ens[y_te==1].mean():.3f} "
          f"| healthy-region flag rate={ens[y_te==0].mean():.3f}")
    print("  interpretation: AUC>0.7 + positive late-cycle score trend + flag rate rising\n"
          "  with degradation = the method transfers to real NASA telemetry.")

    # Cycle-level metrics as PRIMARY (honest statistical unit: 168 cycles, not 50k rows)
    from missionmind.ml.metrics import cycle_level_metrics, cycle_matched_fpr_metrics
    for kind, (sc, fl) in results.items():
        te3 = te.copy(); te3["score"] = sc; te3["flag"] = fl; te3["deg"] = y_te
        cm = cycle_level_metrics(te3, score_col="score", label_col="deg",
                                 cycle_col="cycle_idx", flag_col="flag")
        print(f"  [{kind} CYCLE-LEVEL] n_cycles={cm['n_cycles']} degraded={cm['n_degraded_cycles']} "
              f"ROC-AUC={cm['roc_auc']:.3f} PR-AUC={cm['pr_auc']:.3f} "
              f"Precision={cm['precision']:.3f} Recall/Sens={cm['recall']:.3f} "
              f"Specificity={cm['specificity']:.3f} F1={cm['f1']:.3f} Acc={cm['accuracy']:.3f} "
              f"CM(tp={cm['tp']},fp={cm['fp']},fn={cm['fn']},tn={cm['tn']})")

    # Matched-FPR operating point (the metric ML reviewers ask for first):
    # threshold chosen FROM the healthy test cycles at a target 5% FPR, then
    # precision/recall/F1 reported at that threshold on ALL test cycles.
    print(f"  MATCHED-FPR @ {FPR_TARGET*100:.0f}% (threshold from healthy test cycles):")
    for kind, (sc, fl) in results.items():
        te3 = te.copy(); te3["score"] = sc; te3["deg"] = y_te
        mf = cycle_matched_fpr_metrics(te3, score_col="score", label_col="deg",
                                       cycle_col="cycle_idx", fpr_target=FPR_TARGET)
        if mf is None:
            print(f"  [{kind:8s}] no healthy test cycles - threshold undefined")
            continue
        print(f"  [{kind:8s} @FPR={FPR_TARGET:.2f}] threshold={mf['threshold']:.3f} "
              f"achieved_FPR={mf['achieved_fpr']:.3f} Precision={mf['precision']:.3f} "
              f"Recall={mf['recall']:.3f} Specificity={mf['specificity']:.3f} F1={mf['f1']:.3f} "
              f"CM(tp={mf['tp']},fp={mf['fp']},fn={mf['fn']},tn={mf['tn']})")
    # ensemble at the matched-FPR threshold: min of the two detector scores
    # per row (production ensemble convention), then per-cycle mean.
    ens_score = np.minimum(results["iforest"][0], results["lof"][0])
    te4 = te.copy(); te4["score"] = ens_score; te4["deg"] = y_te
    mf_ens = cycle_matched_fpr_metrics(te4, score_col="score", label_col="deg",
                                       cycle_col="cycle_idx", fpr_target=FPR_TARGET)
    if mf_ens is not None:
        print(f"  [ensemble @FPR={FPR_TARGET:.2f}] threshold={mf_ens['threshold']:.3f} "
              f"achieved_FPR={mf_ens['achieved_fpr']:.3f} Precision={mf_ens['precision']:.3f} "
              f"Recall={mf_ens['recall']:.3f} Specificity={mf_ens['specificity']:.3f} F1={mf_ens['f1']:.3f} "
              f"CM(tp={mf_ens['tp']},fp={mf_ens['fp']},fn={mf_ens['fn']},tn={mf_ens['tn']})")



def arm_e_predictive(b5: pd.DataFrame) -> None:
    """Future-event experiment: does the detector predict degradation at
    t+dt from healthy telemetry at t?

    Train an IsolationForest on the first 35% (healthy) cycles, then for
    each later cycle that is still healthy, score ONLY that cycle's rows
    (no future data) and evaluate whether the score predicted the battery
    reaching EOL within H cycles. Cycle-level units, so the 299x row
    inflation is not an issue. This is the experiment that supports (or
    refutes) a "failure prediction" claim - it is not classifying
    degradation that is already occurring.
    """
    from missionmind.ml.metrics import predictive_horizon_metrics

    cycles = sorted(b5["cycle_idx"].unique())
    train_cycles = cycles[: int(len(cycles) * 0.35)]
    tr = b5[b5["cycle_idx"].isin(train_cycles)]
    te = b5[~b5["cycle_idx"].isin(train_cycles)]
    det = IsolationForest(contamination=0.07, n_estimators=200, random_state=42)
    det.fit(features(tr))
    # per-cycle score from that cycle's rows only (no future leakage)
    sc = -det.decision_function(features(te))
    te2 = te.copy(); te2["score"] = sc
    per_cycle = te2.groupby("cycle_idx").agg(score=("score", "mean"), cap=("capacity_ah", "first"))
    df_e = per_cycle.reset_index().rename(columns={"cycle_idx": "cycle", "cap": "capacity"})
    # threshold: 90th percentile of healthy-training per-cycle scores
    tr_sc = -det.decision_function(features(tr))
    tr_cyc = tr.copy(); tr_cyc["score"] = tr_sc
    thr = float(tr_cyc.groupby("cycle_idx")["score"].mean().quantile(0.90))
    res = predictive_horizon_metrics(df_e, score_col="score", capacity_col="capacity",
                                     cycle_col="cycle", horizons=(10, 25, 50),
                                     eol_fraction=EOL_FRACTION, score_threshold=thr,
                                     initial_capacity=float(b5["capacity_ah"].iloc[0]))
    print("\nARM E - future-event prediction (healthy telemetry at t -> failure within H cycles)")
    print(f"  IsolationForest trained on first {len(train_cycles)} healthy cycles; scoring each "
          f"test cycle from its own rows only; threshold=90th pct of train cycle scores ({thr:.3f})")
    print(f"  {'H':<4} {'healthy':<8} {'events':<7} {'ROC-AUC':<8} {'PR-AUC':<8} {'Prec':<6} {'Rec':<6} {'Spec':<6} {'F1':<5}")
    for H, r in res.items():
        prec = r.get("precision", float("nan")); rec = r.get("recall", float("nan"))
        spec = r.get("specificity", float("nan")); f1 = r.get("f1", float("nan"))
        print(f"  {H:<4} {r['n_healthy']:<8} {r['n_events']:<7} {r['roc_auc']:<8.3f} {r['pr_auc']:<8.3f} "
              f"{prec:<6.3f} {rec:<6.3f} {spec:<6.3f} {f1:<5.3f}")
    print("  interpretation: AUC/PR-AUC > 0.7 with high precision at H=10-25 = the detector"
          "\n  actually predicts degradation AHEAD of time (failure prediction, not just detection).")


def arm_c_cross_battery(b5: pd.DataFrame, other: str) -> None:
    df_other = load_battery(other)
    if df_other.empty:
        print(f"\nARM C - {other}: no discharge cycles found, skipped")
        return
    cycles = sorted(b5["cycle_idx"].unique())
    tr = b5[b5["cycle_idx"].isin(cycles[: int(len(cycles) * 0.35)])]
    det = IsolationForest(contamination=0.07, n_estimators=200, random_state=42)
    det.fit(features(tr))
    X_o = features(df_other)
    sc_o = -det.decision_function(X_o)
    y_o = degraded_label(df_other)
    auc = roc_auc_score(y_o, sc_o) if len(np.unique(y_o)) > 1 else float("nan")
    grp = df_other.copy(); grp["score"] = sc_o
    g = grp.groupby("cycle_idx").agg(score_mean=("score", "mean"), cap=("capacity_ah", "first"))
    sp = spearmanr(g["score_mean"], g["cap"])
    print(f"\nARM C - cross-battery generalization (train B0005, test {other})")
    print(f"  {other}: {len(df_other)} discharge rows, capacity {df_other['capacity_ah'].min():.3f}.."
          f"{df_other['capacity_ah'].max():.3f} Ah, degraded fraction {y_o.mean():.2f}")
    print(f"  AUC(degraded vs healthy)={auc:.3f} | Spearman(score, capacity)={sp.statistic:+.3f} (p={sp.pvalue:.1e})")


def arm_d_all_models(b5: pd.DataFrame) -> None:
    """Run every model in the zoo through the external benchmark on real B0005.

    Unsupervised: fit on early healthy cycles, test on the rest (degraded vs healthy).
    Supervised:   train with capacity-derived labels (first 15% cycles = 0 healthy,
                  last 15% = 1 degraded), test on the middle 70%.
    Metric: ROC-AUC (degraded vs healthy) + Spearman(score, capacity) on the test part.
    """
    from missionmind.ml.advanced_models import get_all_models

    cycles = sorted(b5["cycle_idx"].unique())
    n = len(cycles)
    cut_h, cut_d = int(n * 0.15), int(n * 0.85)
    early = cycles[:cut_h]
    late = cycles[cut_d:]
    mid = cycles[cut_h:cut_d]

    tr_u = b5[b5["cycle_idx"].isin(early)]          # healthy reference (unsupervised)
    te = b5[b5["cycle_idx"].isin(mid + late)]       # test: middle + degraded tail
    tr_s = pd.concat([b5[b5["cycle_idx"].isin(early)], b5[b5["cycle_idx"].isin(late)]])
    y_s = (tr_s["cycle_idx"] >= cut_d).astype(int)  # supervised labels

    y_deg = degraded_label(b5)
    y_te = y_deg[b5["cycle_idx"].isin(mid + late)]
    X_tr_u, X_te, X_tr_s = features(tr_u), features(te), features(tr_s)

    # XGBOD and the PINN are too slow on ~15k training rows; give them a documented
    # stratified 4k-row training sample so the benchmark finishes in reasonable time.
    slow = {"XGBOD", "Custom"}
    rng = np.random.default_rng(0)
    idx_full = np.arange(len(tr_s))
    idx_sub = np.concatenate([rng.choice(idx_full[y_s == c], 2000, replace=False) for c in (0, 1)])

    print("\nARM D - all 8 models on external real B0005 (train early healthy, test degradation)")
    print(f"  train healthy cycles {len(early)}, degraded-label cycles {len(late)}, test cycles {len(mid)+len(late)}")
    for name, model in get_all_models().items():
        sup = ("Supervised" in name or "XGBOD" in name or "FCNN" in name or "Custom" in name)
        try:
            if sup:
                tr_use = tr_s.iloc[idx_sub] if any(s in name for s in slow) else tr_s
                y_use = y_s.iloc[idx_sub] if any(s in name for s in slow) else y_s
                model.fit_supervised(features(tr_use), y_use.values) if hasattr(model, "fit_supervised") else model.fit(features(tr_use))
            else:
                model.fit(X_tr_u)
            sc = model.decision_function(X_te)
            auc = roc_auc_score(y_te, sc) if len(np.unique(y_te)) > 1 else float("nan")
            g = te.copy(); g["score"] = sc
            grp = g.groupby("cycle_idx").agg(score_mean=("score", "mean"), cap=("capacity_ah", "first"))
            sp = spearmanr(grp["score_mean"], grp["cap"])
            print(f"  {name[:48]:<48s} AUC={auc:.3f}  Spearman={sp.statistic:+.3f} (p={sp.pvalue:.1e})")
        except Exception as e:
            print(f"  {name[:48]:<48s} FAILED: {type(e).__name__}: {e}")


def arm_d_quick(b5: pd.DataFrame) -> None:
    """Quick external check: only the three models whose behaviour changed in the
    P3-010 tuning round (XGBOD, Hybrid DIF, Custom PINN), same Arm D protocol
    but with lighter training samples so the e2e dry run stays fast:
      XGBOD/PINN: stratified 1000 rows/class; HybridDIF: 4k-row healthy sample.
    """
    from missionmind.ml.advanced_models import get_all_models
    cycles = sorted(b5["cycle_idx"].unique())
    n = len(cycles)
    cut_h, cut_d = int(n * 0.15), int(n * 0.85)
    early, late, mid = cycles[:cut_h], cycles[cut_d:], cycles[cut_h:cut_d]
    tr_u = b5[b5["cycle_idx"].isin(early)]
    te = b5[b5["cycle_idx"].isin(mid + late)]
    tr_s = pd.concat([b5[b5["cycle_idx"].isin(early)], b5[b5["cycle_idx"].isin(late)]])
    y_s = (tr_s["cycle_idx"] >= cut_d).astype(int)
    y_te = degraded_label(b5)[b5["cycle_idx"].isin(mid + late)]
    X_te = features(te)
    rng = np.random.default_rng(0)
    idx_full = np.arange(len(tr_s))
    idx_sub = np.concatenate([rng.choice(idx_full[y_s.values == c], 1000, replace=False)
                              for c in (0, 1)])
    idx_u = rng.choice(np.arange(len(tr_u)), min(4000, len(tr_u)), replace=False)
    print("\nARM D (quick) - tuned models on real B0005 (light training samples)")
    for name, model in get_all_models().items():
        if not any(k in name for k in ("XGBOD", "Hybrid DIF", "Custom")):
            continue
        try:
            if "Hybrid DIF" in name:
                model.fit(features(tr_u)[idx_u])
            else:
                tr_use, y_use = tr_s.iloc[idx_sub], y_s.iloc[idx_sub]
                model.fit_supervised(features(tr_use), y_use.values)
            sc = model.decision_function(X_te)
            auc = roc_auc_score(y_te, sc) if len(np.unique(y_te)) > 1 else float("nan")
            g = te.copy(); g["score"] = sc
            grp = g.groupby("cycle_idx").agg(score_mean=("score", "mean"), cap=("capacity_ah", "first"))
            sp = spearmanr(grp["score_mean"], grp["cap"])
            print(f"  {name[:48]:<48s} AUC={auc:.3f}  Spearman={sp.statistic:+.3f} (p={sp.pvalue:.1e})")
        except Exception as e:
            print(f"  {name[:48]:<48s} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("MissionMind - validation on the REAL NASA PCoE battery dataset")
    print("=" * 80)
    quick = "--quick" in sys.argv
    if quick:
        print("quick mode: Arm A (raw transfer) + Arm B (method) + Arm D (tuned models) + Arm E (predictive)")
    if not os.path.exists(os.path.join(REAL_DIR, "B0005.mat")):
        raise SystemExit(f"Real NASA .mat files missing in {REAL_DIR}. Download from "
                         "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip "
                         "(BatteryAgingARC-FY08Q4)")
    b5 = load_battery("B0005")
    print(f"\nB0005: {len(b5)} discharge samples across {b5['cycle_idx'].nunique()} cycles")
    arm_a_raw_transfer(b5.copy())
    arm_b_method(b5)
    arm_e_predictive(b5)
    if quick:
        arm_d_quick(b5)
    else:
        for bat in ("B0006", "B0007", "B0018"):
            arm_c_cross_battery(b5, bat)
        arm_d_all_models(b5)
    print("\nDone.")

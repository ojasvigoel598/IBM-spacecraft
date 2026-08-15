#!/usr/bin/env python3
"""Multi-seed robustness sweep on the top-5 PGNN configs from pinn_layer_scan.py.

Runs each (layers, gate_mode, blend) tuple across seeds {0,1,2,7,42,123} and
reports the AUC and |Spearman| distributions (mean +/- std, min, max) so we
can confirm or refute which config stays the winner across seeds.

Protocol matches Arm D of nasa_real_validation.py on real NASA PCoE B0005:
  - healthy label rows = first 15% of cycles, degraded label rows = last 15%
  - test (scoring only) = middle 70% + degraded tail
  - metrics: ROC-AUC(degraded vs healthy) and signed Spearman(score, capacity)

The only thing that varies between seeds is the PGNN's sklearn random_state
(MLPClassifier init + early_stopping seed + autoencoder); the data split is
identical for every (config, seed) pair so the comparison is honest.

Run:
  .venv/Scripts/python.exe -m missionmind.ml.pinn_seed_robustness
"""
import os
import sys
import warnings
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.ml.nasa_real_validation import (
    load_battery, features, degraded_label, REAL_DIR,
)
from missionmind.ml.pinn_layer_scan import PGNN_variant


# Top-5 configs the user identified from the scan; each is (layers, gate, blend).
TOP5 = [
    ("(64,32,16)  reground  alpha=0.30", (64, 32, 16), "reground",  0.30),
    ("(256,128,64) none    alpha=0.30",  (256, 128, 64), "none",    0.30),
    ("(32,16)     none    alpha=0.30",  (32, 16),        "none",    0.30),
    ("(16,)       synth   alpha=0.00",  (16,),           "synthetic", 0.0),
    ("(64,64,64)  reground alpha=0.30", (64, 64, 64),    "reground",  0.30),
]
SEEDS = [0, 1, 2, 7, 42, 123]


def _split_data(b5, train_rows=4000, seed=0):
    """Return the standard Arm-D split. `seed` only controls train-row sub-sampling."""
    cycles = sorted(b5["cycle_idx"].unique())
    n = len(cycles)
    cut_h, cut_d = int(n * 0.15), int(n * 0.85)
    early, late, mid = cycles[:cut_h], cycles[cut_d:], cycles[cut_h:cut_d]
    tr_s = pd.concat([b5[b5["cycle_idx"].isin(early)],
                      b5[b5["cycle_idx"].isin(late)]])
    y_s = (tr_s["cycle_idx"] >= cut_d).astype(int).values
    te = b5[b5["cycle_idx"].isin(mid + late)]
    y_te = degraded_label(b5)[b5["cycle_idx"].isin(mid + late)]
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y_s == c)[0],
                                     train_rows // 2, replace=False)
                          for c in (0, 1)])
    return (features(tr_s)[idx], y_s[idx], features(te), te, y_te)


def _score_one(layers_cfg, gate, blend, X_tr, y_tr, X_te, te, y_te, seed):
    """Fit + score a single (config, seed); return (auc, sp, |sp|)."""
    model = PGNN_variant(hidden_layer_sizes=layers_cfg, gate_mode=gate,
                         blend=blend, random_state=seed)
    model.fit_supervised(X_tr, y_tr)
    sc = model.decision_function(X_te)
    auc = float(roc_auc_score(y_te, sc)) if len(np.unique(y_te)) > 1 else float("nan")
    g2 = te.copy(); g2["score"] = sc
    grp = g2.groupby("cycle_idx").agg(score_mean=("score", "mean"),
                                       cap=("capacity_ah", "first"))
    sp = float(spearmanr(grp["score_mean"], grp["cap"]).statistic)
    return auc, sp, abs(sp) if sp == sp else 0.0


def _summarise(vals):
    """Return (mean, std, min, max) tolerating NaN."""
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, float(v.min()), float(v.max())


def sweep():
    if not os.path.exists(os.path.join(REAL_DIR, "B0005.mat")):
        raise SystemExit(f"Real NASA .mat files missing in {REAL_DIR}")
    b5 = load_battery("B0005")
    print("=" * 96)
    print(f"PINN MULTI-SEED ROBUSTNESS SWEEP  |  B0005: {len(b5)} samples, "
          f"{b5['cycle_idx'].nunique()} cycles  |  seeds {SEEDS}  |  "
          f"{len(TOP5)} configs")
    print("=" * 96)

    # Honor the per-seed train sub-sampling (the existing scan already varies
    # the train subsample by seed). Each config is run for *every* seed so the
    # comparison sweeps across all sources of randomness.
    results = {}  # label -> [(auc, sp, |sp|), ...] in seed order
    for label, layers_cfg, gate, blend in TOP5:
        aucs, sps, fabss = [], [], []
        detail_lines = []
        for s in SEEDS:
            X_tr, y_tr, X_te, te, y_te = _split_data(b5, seed=s)
            auc, sp, fabs = _score_one(layers_cfg, gate, blend,
                                       X_tr, y_tr, X_te, te, y_te, s)
            aucs.append(auc); sps.append(sp); fabss.append(fabs)
            detail_lines.append(f"  seed={s:>3d}: AUC={auc:.3f}  Sp={sp:+.3f}  |Sp|={fabs:.3f}")
        results[label] = (aucs, sps, fabss)
        print(f"\n[{label}]")
        for line in detail_lines:
            print(line)
        am, asd, amin, amax = _summarise(aucs)
        sm, ssd, smin, smax = _summarise(sps)
        xm, xsd, xmin, xmax = _summarise(fabss)
        print(f"  AUC   mean={am:.3f} std={asd:.3f} range=[{amin:.3f}, {amax:.3f}]")
        print(f"  |Sp|  mean={xm:.3f} std={xsd:.3f} range=[{xmin:.3f}, {xmax:.3f}]")
        print(f"  Sp    mean={sm:+.3f} std={ssd:.3f} range=[{smin:+.3f}, {smax:+.3f}]")

    # ---- Comparison table ----
    print("\n" + "=" * 96)
    print("WINNER TABLE  (lower std = more robust; higher mean = stronger)")
    print("=" * 96)
    hdr = (f"{'config':<38s} {'mean(AUC)':>9s} {'std(AUC)':>9s} {'min':>6s} "
           f"{'mean(|Sp|)':>10s} {'std(|Sp|)':>10s} {'min':>6s} "
           f"{'mean(min)':>9s} {'#wins':>6s}")
    print(hdr)
    print("-" * len(hdr))
    min_means = []
    for label, _layers_cfg, _gate, _blend in TOP5:
        aucs, _sps, fabss = results[label]
        am, asd, amin, amax = _summarise(aucs)
        xm, xsd, xmin, xmax = _summarise(fabss)
        min_per_seed = [min(a, b) for a, b in zip(aucs, fabss)]
        mm = float(np.mean(min_per_seed))
        min_means.append((label, mm, am, asd, xm, xsd))
        print(f"{label:<38s} {am:>9.3f} {asd:>9.3f} {amin:>6.3f} "
              f"{xm:>10.3f} {xsd:>10.3f} {xmin:>6.3f} {mm:>9.3f}")

    # #wins column = count of seeds where this config had the highest min(AUC, |Sp|)
    print()
    win_counts = {label: 0 for label, *_ in TOP5}
    for seed_i in range(len(SEEDS)):
        per_seed = []
        for label, _layers_cfg, _gate, _blend in TOP5:
            aucs, _sps, fabss = results[label]
            per_seed.append((label, min(aucs[seed_i], fabss[seed_i])))
        per_seed.sort(key=lambda t: t[1], reverse=True)
        win_counts[per_seed[0][0]] += 1
    print("WIN COUNT PER CONFIG (across ", len(SEEDS), " seeds):", sep="")
    for label, _layers_cfg, _gate, _blend in TOP5:
        print(f"  {label:<38s} -> {win_counts[label]} / {len(SEEDS)} seeds")

    # ---- Verdict ----
    best_label = max(win_counts, key=lambda k: (win_counts[k],
                                                min_means[[l for l, *_ in TOP5].index(k)][1]))
    best_auc_mean = next(((am, asd) for (label, *_), (am, asd, *_rest) in
                          zip([(l,) for l, *_ in TOP5],
                              [tuple(_summarise(results[l][0])[:2])
                               for l, *_ in TOP5])
                          if label == best_label), (None, None))
    print()
    print("=" * 96)
    print(f"ROBUSTNESS VERDICT")
    print("=" * 96)
    print(f"  Highest win-count config: {best_label}")
    print(f"  Wins: {win_counts[best_label]}/{len(SEEDS)}  |  "
          f"mean(min(AUC,|Sp|)): {min_means[[l for l, *_ in TOP5].index(best_label)][1]:.3f}")
    # Stability classification
    for label, _layers_cfg, _gate, _blend in TOP5:
        aucs, _sps, fabss = results[label]
        am, asd, amin, amax = _summarise(aucs)
        xm, xsd, xmin, xmax = _summarise(fabss)
        cov = am / asd if asd > 0 else float("inf")
        verdict = "ROBUST" if (asd <= 0.05 and xsd <= 0.10) else (
                  "MODERATE" if (asd <= 0.10 and xsd <= 0.20) else "FRAGILE")
        print(f"  {label:<38s} {verdict:<8s}  AUC={am:.3f} ± {asd:.3f}  "
              f"|Sp|={xm:.3f} ± {xsd:.3f}")

    # Persist structured results for offline inspection.
    out = {}
    for label, _layers_cfg, _gate, _blend in TOP5:
        aucs, sps, fabss = results[label]
        am, asd, amin, amax = _summarise(aucs)
        xm, xsd, xmin, xmax = _summarise(fabss)
        out[label] = {
            "layers": list(_layers_cfg), "gate": _gate, "blend": _blend,
            "per_seed": [{"seed": s, "auc": float(aucs[i]), "spearman": float(sps[i]),
                          "abs_spearman": float(fabss[i])}
                         for i, s in enumerate(SEEDS)],
            "auc_mean": am, "auc_std": asd, "auc_min": amin, "auc_max": amax,
            "abs_sp_mean": xm, "abs_sp_std": xsd, "abs_sp_min": xmin, "abs_sp_max": xmax,
            "win_count": win_counts[label],
        }
    out_path = os.path.join(os.path.dirname(__file__), "..", "models",
                            "pinn_seed_robustness.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sweep()

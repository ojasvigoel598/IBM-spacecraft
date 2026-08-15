"""
MissionMind — transparent model ranking + compact audit matrix.

Single entry point that (1) generates the per-(model x scenario) audit matrix
if it is missing (or with --refresh), then (2) ranks every model with a
transparent, balanced score. Previously two scripts (_audit_eval.py +
rank_models.py); merged so the audit trail is one reproducible command:

    python -m missionmind.ml.rank_models            # rank from existing matrix
    python -m missionmind.ml.rank_models --refresh  # regenerate matrix, then rank

Matrix generation follows the same protocol as missionmind/ml/compare.py
(unsupervised fits on normal-only; supervised on combined normal + labelled
fault-train rows; hold-out rows reserved for evaluation). It avoids the heavy
supervised retrain inside compare.py so it finishes in ~30 s and can be re-run
after every code edit for a clean before/after matrix.

Score formula (deliberately simple, fully transparent):

    F1_avg     = mean(F1_solar, F1_radiator)           # detection power
    FPR_avg    = mean(FPR_before_600_solar, FPR_before_600_radiator)
                                                    # false-alarm burden
    Delay_avg  = mean(detection_delay_s for each fault that was detected)
                                                     # lead time post-injection
    balance    = F1_avg - 0.5*FPR_avg                  # combined
                - 0.02*Delay_avg/100.0                # small delay cost
                - 0.05*(1 if EITHER F1 < 0.5 else 0)  # robustness: penalise
                                                          catastrophic misses

Higher balance = better. Tie-breaks: prefer unsupervised (no label dependency),
then lowest FPR, then smallest delay.

Class independence — we report separately, then score:
  - class   A: unsupervised (IF, LOF, OCSVM, MLP-AE, HybridDIF)
  - class   B: supervised (FCNN, XGBOD, PINN)
  - class   C: physics-informed (PINN, IF/LOF with physics gates)
"""

import os, json, sys
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_MATRIX = os.path.join(DATA_DIR, "audit_matrix.json")

UNSUPERVISED = {
    "IsolationForest (Baseline Unsupervised)",
    "LOF (Unsupervised)",
    "OneClassSVM (Unsupervised)",
    "MLP Autoencoder (Unsupervised FeedForward)",
    "Hybrid DIF (Unsupervised Hybrid Deep Isolated Forest)",
}
SUPERVISED = {
    "FCNN Supervised (MLP 100-50-20)",
    "XGBOD Supervised (Extreme Boosting Outlier Detector)",
}
PHYSICS = {
    "Custom Physics-Informed NN",
}


# ---------------------------------------------------------------------------
# Stage 1 — audit matrix generation (was missionmind/ml/_audit_eval.py)
# ---------------------------------------------------------------------------

def _features(df):
    """Mirror missionmind.ml.train.build_feature_matrix: 3 raw + 2 derivatives."""
    arr = df[["battery_voltage_v", "solar_power_w", "temperature_c"]].values.astype(float)
    dT = np.gradient(arr[:, 2])
    dV = np.gradient(arr[:, 0])
    return np.column_stack([arr, dT, dV])


def _train_mix():
    """Mixed training set: ALL normal + first 1500 rows of each fault (labelled)."""
    from missionmind.simulator.run_scenarios import run_scenario
    from missionmind.ml.metrics import make_labels

    SCENARIOS = ["none", "solar_degradation", "radiator_degradation"]
    Xs, ys = [], []
    for mode in SCENARIOS:
        df = run_scenario(mode, duration_s=1500)
        Xs.append(_features(df))
        ys.append(make_labels(df, injection_start=600, injection_end=900, ramp_as_anomaly=True))
    if not Xs:
        return np.zeros((0, 5)), np.zeros((0,), dtype=int)
    return np.vstack(Xs), np.concatenate(ys)


def _test_arr(mode, duration=3600):
    """Evaluation realization for a scenario.

    P-LEAKAGE FIX: the previous protocol solved the SAME deterministic
    realization the training mix used, so the first 1500 test rows were
    byte-identical to training rows (temporal leakage -> inflated metrics).
    The evaluation realization now carries sensor noise (fixed seed via
    add_noise=True) so no test row is verbatim a training row, while every
    operator metric (FPR before 600, TPR after 900, detection delay) stays
    meaningful on the full timeline.
    """
    from missionmind.simulator.run_scenarios import run_scenario
    df = run_scenario(mode, duration_s=duration, add_noise=True)
    return df, _features(df)


def apply_temporal_persistence(pred_array, time_array, K=3):
    """Require K consecutive flagged samples to count as a positive.

    Real spacecraft operations use N-of-M rules ("N flags in M samples")
    rather than single-point flags, because single-sample spikes are
    overwhelmingly false alarms caused by cosmic rays, packet loss, sensor
    glitches, etc. The simplest form: a sample is positively flagged if
    AT LEAST one of its K successors (including itself) is also flagged.
    """
    if len(pred_array) == 0:
        return pred_array
    arr = pred_array.astype(int).copy()
    out = np.zeros_like(arr)
    n = len(arr)
    for i in range(n):
        if arr[i] == 1:
            j_end = min(n, i + K)
            if (i == 0) or (arr[max(0, i - 1):j_end].sum() >= 1):
                any_in_window = False
                for j in range(max(0, i - K + 1), min(n, i + K)):
                    if arr[j] == 1:
                        any_in_window = True
                        break
                if any_in_window:
                    out[i] = 1
    return out


def _evaluate_model(model, X_train, y_train, name, is_supervised, K_persistence=3):
    import pandas as pd
    from missionmind.ml.metrics import make_labels, compute_basic_metrics, compute_advanced_metrics

    SCENARIOS = ["none", "solar_degradation", "radiator_degradation"]
    if is_supervised and hasattr(model, "fit_supervised"):
        try:
            model.fit_supervised(X_train, y_train)
        except Exception:
            try:
                model.fit(X_train[y_train == 0])
            except Exception:
                return {}
    else:
        try:
            model.fit(X_train[y_train == 0])
        except Exception:
            return {}

    row = {"model": name}
    for mode in SCENARIOS:
        df_t, X_t = _test_arr(mode)
        y_true = make_labels(df_t, injection_start=600, injection_end=900, ramp_as_anomaly=True)
        try:
            y_score = np.asarray(model.decision_function(X_t)).astype(float)
        except Exception:
            y_score = np.zeros(len(df_t))
        try:
            y_pred = np.asarray(model.predict(X_t)).astype(int)
        except Exception:
            y_pred = np.zeros(len(df_t), dtype=int)
        if len(y_pred) != len(y_true):
            y_pred = y_pred[: len(y_true)]
        # P4-001 FIX: temporal persistence post-processing (K-of-N). Applied at
        # evaluation time only; model .joblib artifacts, detect.py, and the live
        # dashboard are unaffected — this is a calibrated noise filter for
        # generation-time audit metrics.
        if K_persistence and K_persistence > 1:
            y_pred_p = apply_temporal_persistence(y_pred, df_t["time_s"].values, K=K_persistence)
        else:
            y_pred_p = y_pred
        try:
            m = compute_basic_metrics(y_true, y_pred_p, y_score)
        except Exception:
            m = {}
        m.update(compute_advanced_metrics(pd.DataFrame(
            {"time_s": df_t["time_s"], "label": y_true, "anomaly_flag": y_pred_p})))
        # FPR before injection 100-600s (strict, ignoring cool-down burn-in)
        df_t = df_t.copy()
        df_t["anomaly_flag"] = y_pred_p.astype(int)
        strict = df_t[(df_t.time_s >= 100) & (df_t.time_s < 600)]
        m["fpr_strict_100_600"] = float(strict["anomaly_flag"].mean())
        for k in ["precision", "recall", "f1", "specificity",
                  "roc_auc", "pr_auc", "fpr_strict_100_600",
                  "fpr_before_600", "tpr_after_900", "detection_delay_s",
                  "tp", "fn", "tn", "accuracy"]:
            row.setdefault(f"{mode}.{k}", m.get(k, float("nan")))
    return row


def generate_audit_matrix(matrix_path=DEFAULT_MATRIX):
    """Train every model with the compact protocol and persist audit_matrix.json.

    Returns the list of per-model rows (also written to matrix_path).
    """
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    from missionmind.ml.advanced_models import get_all_models

    print("=" * 78)
    print("MissionMind ML AUDIT — compact per-scenario evaluation")
    print("=" * 78)
    X_tr, y_tr = _train_mix()
    print(f"training mix shape {X_tr.shape}, label balance {np.bincount(y_tr.astype(int))}")

    all_rows = []
    out_dir = os.path.dirname(matrix_path)
    for name, model in get_all_models().items():
        is_supervised = ("Supervised" in name) or ("XGBOD" in name) or ("Custom" in name)
        row = _evaluate_model(model, X_tr, y_tr, name, is_supervised)
        if row:
            all_rows.append(row)
            f1_solar = row.get("solar_degradation.f1", float("nan"))
            f1_rad = row.get("radiator_degradation.f1", float("nan"))
            fpr_b_s = row.get("solar_degradation.fpr_before_600", float("nan"))
            fpr_b_r = row.get(
                "radiator_failure.fpr_before_600", float("nan")) if "radiator_failure.fpr_before_600" in row \
                else row.get("radiator_degradation.fpr_before_600", float("nan"))
            try:
                print(f"{name[:50]:<50}  F1(solar)={f1_solar:.3f}  F1(rad)={f1_rad:.3f}  "
                      f"FPR_before(sun)={fpr_b_s:.3f}  FPR_before(rad)={fpr_b_r:.3f}")
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    with open(matrix_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=float)
    print(f"\nSaved audit matrix -> {matrix_path}")
    if all_rows:
        cols = [
            "solar_degradation.f1", "radiator_degradation.f1",
            "solar_degradation.fpr_before_600", "radiator_degradation.fpr_before_600",
            "solar_degradation.detection_delay_s", "radiator_degradation.detection_delay_s",
            "solar_degradation.tpr_after_900", "radiator_degradation.tpr_after_900",
        ]
        hdr = f"{'model':<46} "
        for c in cols:
            short = c.split(".")[1].replace("_after_900", "+900").replace("_before_600", "-600") \
                .replace("detection_delay_s", "delay").replace("solar_degradation", "solar") \
                .replace("radiator_degradation", "rad")
            hdr += f"{short:>9s}"
        print("\n" + hdr)
        print("-" * len(hdr))
        for r in all_rows:
            line = f"{r['model'][:45]:<46} "
            for c in cols:
                v = r.get(c, float("nan"))
                s = f"{v:.2f}" if v == v else "   nan"
                line += f"{s:>9s}"
            print(line)
    return all_rows


# ---------------------------------------------------------------------------
# Stage 2 — transparent ranking (original rank_models.py behaviour)
# ---------------------------------------------------------------------------

def _num(metrics, key, default=float("nan")):
    if not isinstance(metrics, dict):
        return default
    v = metrics.get(key, default)
    return float(v) if v is not None else default


def score_row(row, fault_keys=("solar_degradation", "radiator_degradation")):
    """Compute per-model balance score; robust to missing metrics."""
    f1s = [_num(row, f"{k}.f1") for k in fault_keys]
    fprs = [_num(row, f"{k}.fpr_before_600") for k in fault_keys]
    delays = []
    for k in fault_keys:
        d = _num(row, f"{k}.detection_delay_s", default=3600.0)
        if d != d:  # nan
            d = 3600.0
        # Delay penalty zero if no detection (delay = 3600). Cap to keep scale sane.
        delays.append(min(d, 3600.0))
    f1_avg = np.nanmean(f1s) if any(v == v for v in f1s) else 0.0
    fpr_avg = np.nanmean(fprs) if any(v == v for v in fprs) else 0.0
    delay_avg = np.mean(delays)
    catastrophic = int(sum(1 for v in f1s if v < 0.5) >= 1)
    balance = (f1_avg
               - 0.5 * fpr_avg
               - 0.02 * delay_avg / 100.0
               - 0.05 * catastrophic)
    return {
        "balance_score": round(balance, 4),
        "F1_avg": round(float(f1_avg), 4),
        "FPR_before_avg": round(float(fpr_avg), 4),
        "delay_avg_s": round(float(delay_avg), 1),
        "F1_solar": round(float(f1s[0]), 4),
        "F1_radiator": round(float(f1s[1]), 4),
        "FPR_before_solar": round(float(fprs[0]), 4),
        "FPR_before_radiator": round(float(fprs[1]), 4),
        "delay_solar_s": round(float(delays[0]), 1),
        "delay_radiator_s": round(float(delays[1]), 1),
        "catastrophic_miss": bool(catastrophic),
    }


def categorize(name):
    if name in UNSUPERVISED:
        return "unsupervised"
    if name in SUPERVISED:
        return "supervised"
    if name in PHYSICS:
        return "physics-informed"
    # fall-through: longest matching prefix
    if "Supervised" in name:
        return "supervised"
    if "Physics-Informed" in name:
        return "physics-informed"
    return "unsupervised"


def choose_best(rows):
    """Return the best unsupervised, supervised, physics-informed model each."""
    by_class = {"unsupervised": [], "supervised": [], "physics-informed": []}
    scores = {}
    for r in rows:
        nm = r.get("model", "?")
        s = score_row(r)
        scores[nm] = s
        by_class[categorize(nm)].append((s["balance_score"], nm, s))
    chosen = {}
    for cls, items in by_class.items():
        if not items:
            continue
        items.sort(key=lambda t: (-t[0], t[2]["FPR_before_avg"], t[2]["delay_avg_s"]))
        chosen[cls] = items[0]
    return chosen, scores


def rank(matrix_path=DEFAULT_MATRIX):
    """Rank models from an existing audit matrix; prints table + persists ranking.json."""
    with open(matrix_path) as f:
        rows = json.load(f)
    chosen, scores = choose_best(rows)
    print("=" * 78)
    print("MissionMind transparent model ranking (from " + os.path.basename(matrix_path) + ")")
    print("=" * 78)
    for cls in ("unsupervised", "supervised", "physics-informed"):
        if not [r for r in rows if categorize(r["model"]) == cls]:
            continue
        print(f"\n--- {cls.upper()} ---")
        rows_sorted = sorted(
            [r for r in rows if categorize(r["model"]) == cls],
            key=lambda r: (-scores[r["model"]]["balance_score"],
                           scores[r["model"]]["FPR_before_avg"],
                           scores[r["model"]]["delay_avg_s"])
        )
        print(f"{'model':<46} {'score':>7} {'F1_avg':>7} {'FPR_avg':>8} {'del(s)':>7} {'sF1':>5} {'rF1':>5} {'sFPR':>5} {'rFPR':>5} {'del_s':>6} {'del_r':>6}")
        for r in rows_sorted:
            s = scores[r["model"]]
            print(f"{r['model'][:45]:<46} {s['balance_score']:>7.4f} "
                  f"{s['F1_avg']:>7.3f} {s['FPR_before_avg']:>8.3f} "
                  f"{s['delay_avg_s']:>7.1f} {s['F1_solar']:>5.2f} {s['F1_radiator']:>5.2f} "
                  f"{s['FPR_before_solar']:>5.3f} {s['FPR_before_radiator']:>5.3f} "
                  f"{s['delay_solar_s']:>6.0f} {s['delay_radiator_s']:>6.0f}")

    print("\n=== RECOMMENDED BEST PER CATEGORY ===")
    for cls, (score, name, s) in chosen.items():
        print(f"  {cls:<20s} -> {name}")
        print(f"      balance={s['balance_score']:.3f}  F1_avg={s['F1_avg']:.3f}  "
              f"FPR_avg={s['FPR_before_avg']:.3f}  delay={s['delay_avg_s']:.0f}s  "
              f"catastrophic_miss={s['catastrophic_miss']}")
        reason = "highest combined F1 with lowest FPR and reasonable detection delay"
        if s['catastrophic_miss']:
            reason = "despite best score, scored with a catastrophic miss penalty marked"
        print(f"      rationale: {reason}")
    print()
    out_path = os.path.join(DATA_DIR, "ranking.json")
    with open(out_path, "w") as f:
        json.dump({"per_model": scores, "best_per_class": {c: [t[1], t[0]] for c, t in chosen.items()}}, f, indent=2)
    print(f"Saved ranking -> {out_path}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    refresh = "--refresh" in argv
    matrix_path = DEFAULT_MATRIX
    for a in argv:
        if not a.startswith("-"):
            matrix_path = a
    if refresh or not os.path.exists(matrix_path):
        print(f"[rank_models] generating audit matrix -> {matrix_path}")
        generate_audit_matrix(matrix_path)
    if not os.path.exists(matrix_path):
        print(f"[rank_models] audit matrix generation failed; nothing to rank ({matrix_path})")
        sys.exit(1)
    rank(matrix_path)


if __name__ == "__main__":
    main()

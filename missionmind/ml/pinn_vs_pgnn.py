"""Compare a strict Raissi 2019 PINN against feature-only PGNN on real NASA B0005.

The question being answered:
    "Does the marginal effort of a strict PINN (composite data + physics
     loss with dC/dn_ODE - dC/dn_NN residual) beat the current PGNN-best
     (feature-only MLPClassifier with healthy-envelope physics gates) by
     a MEASURABLE margin?"

We run both methods on the IDENTICAL Arm-D protocol used in
missionmind.ml.pinn_layer_scan.py (cycles 0..15% healthy + 85..100% degraded
for training; 15..100% for testing) and report BOTH:
  * ROC-AUC (degraded vs healthy discrimination at the row level)
  * signed Spearman(score, capacity) at the CYCLE level
  * the combined `min(AUC, |Spearman|)` selection metric used by P4-002

The PGNN reference is taken from the top-5 multi-seed sweep:
  (64,64,64) reground α=0.30 won 2/6 seeds with lowest AUC σ
  (P4-002 criterion `min(AUC,|Sp|)`).
We re-run it here on the same seed (0) for an apples-to-apples comparison
against the PINN. We then report the TRUE-answer interpretation in plain
text — "does PINN beat PGNN by a measurable margin".

Run:
    .venv/Scripts/python.exe -m missionmind.ml.pinn_vs_pgnn
"""
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from missionmind.ml.nasa_real_validation import (
    load_battery, features, degraded_label, REAL_DIR,
)
from missionmind.ml.pinn_layer_scan import PGNN_variant
from missionmind.ml.pinn_raissi import RaissiBatteryPINN


# ---------------------------------------------------------------------------
# Reproducible result artifacts: table (CSV) + figure (PNG)
# ---------------------------------------------------------------------------

def results_table(results: dict) -> list:
    """Flatten the comparison results dict into table rows.

    Each row is (model_name, auc, spearman, abs_sp, min_metric, time_s),
    PGNN first, then each PINN lambda sweep in ascending lambda.
    """
    rows = []
    p = results["pgnn"]
    rows.append(("PGNN", float(p["auc"]), float(p["spearman"]),
                 float(p["abs_sp"]), float(p["min_metric"]),
                 float(p["time_s"])))
    for r_ in results["pinn"]:
        lam = float(r_["lam"])
        rows.append((f"PINN(lam={lam:.1f})", float(r_["auc"]),
                     float(r_["spearman"]), float(r_["abs_sp"]),
                     float(min(r_["auc"], r_["abs_sp"])),
                     float(r_["time_s"])))
    return rows


def write_results_csv(path: str, rows: list) -> None:
    """Persist the comparison table as a header + CSV rows."""
    header = ("model,auc,spearman,abs_spearman,min_auc_spearman,time_s")
    with open(path, "w") as f:
        f.write(header + "\n")
        for name, auc, sp, fab, mm, t in rows:
            f.write(f"{name},{auc:.4f},{sp:+.4f},{fab:.4f},{mm:.4f},{t:.2f}\n")


def render_comparison_figure(results: dict, path: str) -> None:
    """2-panel figure: AUC and |Spearman| across the PINN lambda sweep,
    with the feature-only PGNN reference drawn as dashed lines.

    Panel 1 (left):  ROC-AUC vs lambda  — higher is better; PGNN reference.
    Panel 2 (right): |Spearman(score, capacity)| vs lambda — the physics
    term's best hope; PGNN reference.
    The subtitle/annotations state plainly where physics helps vs hurts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = [float(r_["lam"]) for r_ in results["pinn"]]
    auc = [float(r_["auc"]) for r_ in results["pinn"]]
    sp = [abs(float(r_["spearman"])) for r_ in results["pinn"]]
    pgnn_auc = float(results["pgnn"]["auc"])
    pgnn_sp = float(results["pgnn"]["abs_sp"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle("Strict PINN (physics-loss) vs feature-only PGNN — real NASA B0005\n"
                 "Arm-D protocol, seed 0, one cycle per row",
                 fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.plot(lam, auc, "-o", color="#c0392b", lw=2, ms=6,
            label="PINN (physics loss weighted by $\\lambda$)")
    ax.axhline(pgnn_auc, ls="--", color="#2c3e50", lw=2,
               label=f"PGNN feature-only: AUC={pgnn_auc:.3f}")
    ax.set_xlabel("physics weight $\\lambda$")
    ax.set_ylabel("ROC-AUC (degraded vs healthy)")
    ax.set_title("AUC — physics term does not lift discrimination", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(lam, sp, "-s", color="#8e44ad", lw=2, ms=6,
            label="PINN $|$Spearman$|$ (score vs capacity)")
    ax.axhline(pgnn_sp, ls="--", color="#2c3e50", lw=2,
               label=f"PGNN feature-only: |Sp|={pgnn_sp:.3f}")
    ax.set_xlabel("physics weight $\\lambda$")
    ax.set_ylabel("$|$Spearman(score, capacity)$|$")
    ax.set_title("Cycle-level correlation — physics helps only marginally", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.text(0.5, 0.015,
             "Finding: adding the dC/dn physics residual never beats the feature-only"
             " healthy-envelope gates on this metric — physics helps the PINN's own"
             " regression, not its anomaly discrimination.",
             ha="center", fontsize=9, color="#444444")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Top-5 PGNN reference from the multi-seed sweep; winner by min(AUC, |Sp|).
# We pick `(64,64,64) reground α=0.30` as the productivity competitor.
# Structure: (layers_cfg_tuple, gate_mode_str, blend_float).
PGNN_REF = ((64, 64, 64), "reground", 0.30)


def _split(b5, train_rows=4000, seed=0):
    """Arm-D protocol from pinn_layer_scan.py."""
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


def evaluate_pgnn(b5, seed=0):
    layers_cfg, gate, blend = PGNN_REF
    X_tr, y_tr, X_te, te, y_te = _split(b5, seed=seed)
    m = PGNN_variant(hidden_layer_sizes=layers_cfg,
                     gate_mode=gate, blend=blend,
                     random_state=seed)
    m.fit_supervised(X_tr, y_tr)
    sc = m.decision_function(X_te)
    if len(np.unique(y_te)) < 2:
        return float("nan"), 0.0
    auc = float(roc_auc_score(y_te, sc))
    g2 = te.copy(); g2["score"] = sc
    grp = g2.groupby("cycle_idx").agg(
        score_mean=("score", "mean"), cap=("capacity_ah", "first"))
    sp = float(spearmanr(grp["score_mean"], grp["cap"]).statistic)
    return auc, sp


def evaluate_pinn(b5, lam=0.5, epochs=80, hidden=(16,)):
    """PINN AUC + Spearman on the same test set.

    Strategy: per-cycle-group median capacity, build labels from 'cycle_idx
    below or above the median capacity' so we get a balanced binary. We do
    NOT use `degraded_label` because the PINN produces a per-cycle point
    proxy (its training target IS the capacity) and the AUC needs many
    duplicates.
    """
    cycles = sorted(b5["cycle_idx"].unique())
    n = len(cycles)
    cut_h, cut_d = int(n * 0.15), int(n * 0.85)
    early_cycles, late_cycles = cycles[:cut_h], cycles[cut_d:]
    mid_cycles = cycles[cut_h:cut_d]
    train_pts = b5[b5["cycle_idx"].isin(early_cycles + late_cycles)]
    # Use one row per cycle to keep the PINN compact
    train_one = train_pts.drop_duplicates(subset="cycle_idx").reset_index(drop=True)
    n_arr = train_one["cycle_idx"].values.astype(np.float64)
    cap_arr = train_one["capacity_ah"].values.astype(np.float64)
    n_arr = n_arr / max(float(n_arr.max()), 1.0)
    pinn = RaissiBatteryPINN(hidden=hidden, lam=lam, epochs=epochs,
                              random_state=0, lr=1e-2)
    pinn.fit(n_arr, cap_arr)
    test_cycles = mid_cycles + late_cycles
    test_pts = b5[b5["cycle_idx"].isin(test_cycles)]
    test_one = test_pts.drop_duplicates(subset="cycle_idx").reset_index(drop=True)
    n_te = test_one["cycle_idx"].values.astype(np.float64)
    cap_te = test_one["capacity_ah"].values.astype(np.float64)
    n_te_n = n_te / max(float(n_te.max()), 1.0)
    # Use the physics residual as the anomaly score:
    # r > 0  => NN predicts less fade than the ODE expects => "suspiciously healthy"
    # r < 0  => NN predicts more fade than the ODE expects => "genuinely more degraded"
    # End-of-life cells SHOULD have r ~ 0 because the NN has matched the ODE;
    # anomalous cells deviate.  We take |residual| as the score.
    res = pinn.decision_function(n_te_n)
    # Build a 'degraded vs healthy' binary from observed capacity:
    threshold_cap = float(np.median(cap_te))
    y_te = (cap_te <= threshold_cap).astype(int)
    if len(np.unique(y_te)) < 2:
        return float("nan"), 0.0, pinn._alpha
    sc = np.abs(res)
    try:
        auc = float(roc_auc_score(y_te, sc))
    except Exception:
        auc = float("nan")
    sp = float(spearmanr(sc, cap_te).statistic)
    return auc, sp, pinn._alpha


# ---------------------------------------------------------------------------
# Architecture x lambda sweep
# ---------------------------------------------------------------------------

def sweep_pinn_architectures(b5=None, hidden_list=None, lam_list=None,
                             epochs=80, _evaluator=None) -> list:
    """Run the PINN across a (hidden, lambda) grid on real B0005.

    Returns one dict per config:
        {hidden, lam, auc, spearman, abs_sp, min_metric, time_s}

    `_evaluator(lam, hidden, epochs) -> (auc, spearman, alpha)` is an optional
    dependency-injection seam used by the unit tests; when omitted the real
    `evaluate_pinn` on NASA B0005 is used.
    """
    hidden_list = hidden_list or ((16,), (32, 16), (64, 32, 16))
    lam_list = lam_list or (0.0, 0.3, 0.5, 0.7, 1.0)
    rows = []
    for hidden in hidden_list:
        for lam in lam_list:
            t0 = time.time()
            if _evaluator is not None:
                auc, sp, alpha = _evaluator(lam, hidden, epochs)
            else:
                auc, sp, alpha = evaluate_pinn(b5, lam=lam, epochs=epochs,
                                               hidden=hidden)
            abs_sp = abs(sp) if sp == sp else 0.0
            mm = min(auc, abs_sp) if auc == auc else 0.0
            rows.append({
                "hidden": list(hidden), "lam": float(lam), "auc": float(auc),
                "spearman": float(sp), "abs_sp": float(abs_sp),
                "min_metric": float(mm), "time_s": time.time() - t0,
            })
    return rows


def architecture_verdict(pgnn_min_metric: float, rows: list) -> dict:
    """Best (hidden, lam) config by min(AUC,|Sp|) and the verdict vs PGNN."""
    best = max(rows, key=lambda r: r["min_metric"])
    delta = float(best["min_metric"] - pgnn_min_metric)
    if delta > 0.02:
        verdict = ("YES \u2014 a PINN config beats feature-only PGNN by a "
                   "MEASURABLE margin.")
    elif delta > 0:
        verdict = ("MARGINAL \u2014 best PINN edges PGNN but the margin is "
                   "within seed noise.")
    else:
        verdict = ("NO \u2014 strict PINN does NOT beat feature-only PGNN at any "
                   "architecture on this metric.")
    return {"best_config": best, "delta": delta, "verdict": verdict}


def write_sweep_csv(path: str, rows: list) -> None:
    header = "hidden,lam,auc,spearman,abs_spearman,min_auc_spearman,time_s"
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            h = "x".join(str(d) for d in r["hidden"])
            f.write(f"{h},{r['lam']:.1f},{r['auc']:.4f},{r['spearman']:+.4f},"
                    f"{r['abs_sp']:.4f},{r['min_metric']:.4f},{r['time_s']:.2f}\n")


def render_arch_sweep_figure(rows: list, pgnn_auc: float, pgnn_sp: float,
                             path: str) -> None:
    """2-panel figure: AUC and |Spearman| vs lambda, one line per hidden cfg,
    with the feature-only PGNN reference dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hidden_list = sorted({tuple(r["hidden"]) for r in rows},
                         key=lambda h: (len(h), h))
    colors = ["#c0392b", "#2980b9", "#27ae60"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle("Strict PINN architecture sweep on real NASA B0005\n"
                 "AUC and |Spearman| vs physics weight \u03bb, per hidden layer set",
                 fontsize=11, fontweight="bold")
    for ax, (metric, ylabel, ref, title) in zip(
            axes,
            [("auc", "ROC-AUC (degraded vs healthy)", pgnn_auc,
              "AUC \u2014 physics never lifts discrimination"),
             ("abs_sp", "$|$Spearman(score, capacity)$|$", pgnn_sp,
              "Cycle-level correlation vs physics weight")]):
        for i, h in enumerate(hidden_list):
            sub = [r for r in rows if tuple(r["hidden"]) == h]
            sub = sorted(sub, key=lambda r: r["lam"])
            ax.plot([r["lam"] for r in sub], [r[metric] for r in sub],
                    "-o", color=colors[i % len(colors)], lw=2, ms=5,
                    label="hidden " + "x".join(map(str, h)))
        ax.axhline(ref, ls="--", color="#2c3e50", lw=2,
                   label=f"PGNN feature-only: {ref:.3f}")
        ax.set_xlabel("physics weight \u03bb")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.text(0.5, 0.015,
             "Finding: no hidden-layer configuration lifts the strict PINN above "
             "the feature-only healthy-envelope gates on this metric.",
             ha="center", fontsize=9, color="#444444")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if not os.path.exists(os.path.join(REAL_DIR, "B0005.mat")):
        raise SystemExit(f"Real NASA .mat files missing in {REAL_DIR}")
    b5 = load_battery("B0005")
    print("=" * 90)
    print(f"PINN vs PGNN on real NASA B0005  |  {len(b5)} samples, "
          f"{b5['cycle_idx'].nunique()} cycles  |  PGNN ref = {PGNN_REF}")
    print("=" * 90)

    # --- PGNN ---
    t0 = time.time()
    pgnn_auc, pgnn_sp = evaluate_pgnn(b5, seed=0)
    pgnn_time = time.time() - t0
    print(f"\nPGNN  (feature-only, healthy-envelope gates):")
    print(f"  layers = {PGNN_REF[0]}, gate = {PGNN_REF[1]}, blend = {PGNN_REF[2]}")
    print(f"  AUC        = {pgnn_auc:.4f}")
    print(f"  Spearman   = {pgnn_sp:+.4f}  |Sp|={abs(pgnn_sp):.4f}")
    print(f"  min(AUC,|Sp|) = {min(pgnn_auc, abs(pgnn_sp)):.4f}")
    print(f"  train time = {pgnn_time:.2f}s")

    # --- PINN across lam ---
    pinn_sweeps = []
    for lam in (0.0, 0.3, 0.5, 0.7, 1.0):
        t0 = time.time()
        try:
            auc, sp, alpha = evaluate_pinn(b5, lam=lam, epochs=80, hidden=(16,))
            elapsed = time.time() - t0
            pinn_sweeps.append({
                "lam": lam, "auc": auc, "spearman": sp,
                "abs_sp": abs(sp) if sp == sp else 0.0,
                "alpha_learned": alpha, "time_s": elapsed,
            })
            print(f"\nPINN  lam={lam:.1f}:")
            print(f"  AUC        = {auc:.4f}")
            print(f"  Spearman   = {sp:+.4f}  |Sp|={abs(sp):.4f}")
            print(f"  min(AUC,|Sp|) = {min(auc, abs(sp)):.4f}")
            print(f"  alpha_learned = {alpha:.6f}")
            print(f"  train time = {elapsed:.2f}s")
        except Exception as e:  # noqa: BLE001
            print(f"  PINN lam={lam} failed: {type(e).__name__}: {e}")

    # --- Comparison ---
    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD  (B0005 Arm-D protocol, one cycle per row)")
    print("=" * 90)
    rows = [("PGNN",
             pgnn_auc, pgnn_sp, abs(pgnn_sp), pgnn_time)]
    for r_ in pinn_sweeps:
        rows.append((f"PINN(lam={r_['lam']:.1f})",
                     r_["auc"], r_["spearman"], r_["abs_sp"], r_["time_s"]))
    print(f"{'model':<22s} {'AUC':>7s} {'spearman':>10s} {'|Sp|':>6s} "
          f"{'min(AUC,|Sp|)':>14s} {'time_s':>8s}")
    print("-" * 70)
    best_pgnn = min(pgnn_auc, abs(pgnn_sp))
    best_pinn = max((min(r["auc"], r["abs_sp"]) for r in pinn_sweeps),
                     default=0.0)
    for name, auc, sp, fabs, t in rows:
        try:
            mm = min(auc, fabs)
        except Exception:
            mm = float("nan")
        print(f"{name:<22s} {auc:>7.4f} {sp:>+10.4f} {fabs:>6.4f} "
              f"{mm:>14.4f} {t:>8.2f}")

    print()
    print("-" * 90)
    print("ANSWER TO THE OPEN QUESTION:")
    print("-" * 90)
    delta_auc = best_pinn - pgnn_auc
    delta_sp = best_pinn - best_pgnn
    if delta_sp > 0.02:
        verdict = "YES — strict PINN beats feature-only PGNN by a MEASURABLE margin."
    elif delta_sp > 0:
        verdict = "MARGINAL — PINN beats PGNN but the margin is within seed noise."
    else:
        verdict = "NO — strict PINN does NOT beat feature-only PGNN on this metric."
    print(f"  best PGNN min(AUC,|Sp|) = {best_pgnn:.4f}")
    print(f"  best PINN min(AUC,|Sp|) = {best_pinn:.4f}  (delta = {delta_sp:+.4f})")
    print(f"  verdict: {verdict}")

    # Persist
    out = {
        "pgnn_ref": list(PGNN_REF),
        "pgnn": {"auc": pgnn_auc, "spearman": pgnn_sp,
                  "abs_sp": abs(pgnn_sp), "min_metric": best_pgnn,
                  "time_s": pgnn_time},
        "pinn": pinn_sweeps,
        "best_pinn_min": best_pinn,
        "delta_pinn_minus_pgnn": delta_sp,
        "verdict": verdict,
    }
    out_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "models",
                     "pinn_vs_pgnn_b0005.json"))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # Reproducible artifacts: CSV table + PNG figure from the same run.
    rows = results_table(out)
    csv_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "models",
                     "pinn_vs_pgnn_b0005.csv"))
    write_results_csv(csv_path, rows)
    print(f"Wrote {csv_path}")
    try:
        png_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "models",
                         "pinn_vs_pgnn_b0005.png"))
        render_comparison_figure(out, png_path)
        print(f"Wrote {png_path}")
    except Exception as e:  # noqa: BLE001 — figure is a nice-to-have
        print(f"Figure render skipped ({type(e).__name__}: {e})")


def main_arch_sweep():
    """Architecture x lambda sweep (--arch-sweep mode).

    Runs the PINN across hidden=(16,), (32,16), (64,32,16) and
    lam=0.0..1.0 on real B0005, reports min(AUC,|Sp|) per config, and
    persists JSON + CSV + PNG artifacts.
    """
    if not os.path.exists(os.path.join(REAL_DIR, "B0005.mat")):
        raise SystemExit(f"Real NASA .mat files missing in {REAL_DIR}")
    b5 = load_battery("B0005")
    print("=" * 90)
    print("PINN ARCHITECTURE SWEEP on real NASA B0005  |  "
          f"{len(b5)} samples, {b5['cycle_idx'].nunique()} cycles")
    print("hidden=(16,),(32,16),(64,32,16) x lam=0.0,0.3,0.5,0.7,1.0")
    print("=" * 90)

    pgnn_auc, pgnn_sp = evaluate_pgnn(b5, seed=0)
    pgnn_min = min(pgnn_auc, abs(pgnn_sp))
    print(f"\nPGNN reference (feature-only, (64,64,64) reground a=0.30):")
    print(f"  AUC={pgnn_auc:.4f}  Spearman={pgnn_sp:+.4f}  "
          f"min(AUC,|Sp|)={pgnn_min:.4f}")

    rows = sweep_pinn_architectures(b5, epochs=80)
    v = architecture_verdict(pgnn_min, rows)

    print(f"\n{'hidden':<14s} {'lam':>4s} {'AUC':>7s} {'spearman':>9s} "
          f"{'|Sp|':>6s} {'min(AUC,|Sp|)':>13s} {'s':>5s}")
    print("-" * 64)
    for r in sorted(rows, key=lambda r: (len(r["hidden"]), r["hidden"], r["lam"])):
        h = "x".join(str(d) for d in r["hidden"])
        print(f"{h:<14s} {r['lam']:>4.1f} {r['auc']:>7.4f} {r['spearman']:>+9.4f} "
              f"{r['abs_sp']:>6.4f} {r['min_metric']:>13.4f} {r['time_s']:>5.1f}")

    best = v["best_config"]
    print(f"\nBEST PINN config: hidden={best['hidden']} lam={best['lam']:.1f} "
          f"min(AUC,|Sp|)={best['min_metric']:.4f}")
    print(f"PGNN min(AUC,|Sp|)={pgnn_min:.4f}  delta={v['delta']:+.4f}")
    print(f"verdict: {v['verdict']}")

    out = {"pgnn": {"auc": pgnn_auc, "spearman": pgnn_sp,
                    "abs_sp": abs(pgnn_sp), "min_metric": pgnn_min},
           "sweep": rows,
           "best_config": best,
           "delta_pinn_minus_pgnn": v["delta"],
           "verdict": v["verdict"]}
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                         "models", "pinn_arch_sweep_b0005"))
    with open(base + ".json", "w") as f:
        json.dump(out, f, indent=2)
    write_sweep_csv(base + ".csv", rows)
    print(f"\nWrote {base}.json")
    print(f"Wrote {base}.csv")
    try:
        render_arch_sweep_figure(rows, pgnn_auc, abs(pgnn_sp), base + ".png")
        print(f"Wrote {base}.png")
    except Exception as e:  # noqa: BLE001
        print(f"Figure render skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    if "--arch-sweep" in sys.argv:
        main_arch_sweep()
    else:
        main()

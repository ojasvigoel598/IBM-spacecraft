"""
Metrics review (user request): add the missing threshold-dependent metric
(specificity) and two evaluation modes the current validation lacks:

1. cycle_level_metrics — aggregate row-level scores/flags per cycle and
   compute both threshold-independent (ROC-AUC, PR-AUC) and
   threshold-dependent (precision, recall, F1, specificity, confusion)
   metrics on per-cycle units. Row-level metrics over ~50k rows from 168
   cycles inflate apparent statistical power ~300x; cycle level is the
   honest unit.

2. predictive_horizon_metrics — the future-event experiment that must exist
   before claiming "failure prediction": at each healthy cycle c, does the
   detector's score (using only data up to c) predict that the battery
   reaches EOL within the next H cycles? This is "healthy telemetry at time
   t -> predict failure at t+dt", not "classify current degradation".

All expected values are hand-derived literals.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.ml.metrics import compute_basic_metrics, cycle_level_metrics, predictive_horizon_metrics


def test_specificity_in_basic_metrics():
    # y_true=[0,0,1,1], y_pred=[0,1,1,1] -> tn=1, fp=1, fn=0, tp=2
    m = compute_basic_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert m["tn"] == 1 and m["fp"] == 1 and m["fn"] == 0 and m["tp"] == 2
    assert m["specificity"] == 0.5, m["specificity"]   # tn/(tn+fp) = 1/2
    assert m["recall"] == 1.0                          # tp/(tp+fn) = 2/2
    assert m["precision"] == 2 / 3                      # tp/(tp+fp) = 2/3


def test_cycle_level_metrics_perfect_separation():
    # 4 cycles x 3 rows; cycles 2-3 degraded with high scores
    df = pd.DataFrame({
        "cycle": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "score": [0.1, 0.2, 0.15, 0.3, 0.4, 0.5, 0.8, 0.7, 0.9, 0.6, 0.65, 0.7],
        "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "flag":  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    })
    res = cycle_level_metrics(df, score_col="score", label_col="label",
                              cycle_col="cycle", flag_col="flag")
    # per-cycle means: [0.15, 0.4, 0.8, 0.65]; labels: [0,0,1,1]
    assert res["n_cycles"] == 4
    assert res["roc_auc"] == 1.0, res["roc_auc"]
    assert res["pr_auc"] == 1.0, res["pr_auc"]
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["specificity"] == 1.0
    assert res["accuracy"] == 1.0


def test_cycle_level_metrics_imperfect():
    # same data but cycle-1's flag flips to 1 (false positive at cycle level)
    df = pd.DataFrame({
        "cycle": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "score": [0.1, 0.2, 0.15, 0.3, 0.4, 0.5, 0.8, 0.7, 0.9, 0.6, 0.65, 0.7],
        "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "flag":  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })
    res = cycle_level_metrics(df, score_col="score", label_col="label",
                              cycle_col="cycle", flag_col="flag")
    # per-cycle flags: [0,1,1,1], labels [0,0,1,1] -> tp=2, fp=1, fn=0, tn=1
    assert res["precision"] == 2 / 3
    assert res["recall"] == 1.0
    assert res["specificity"] == 0.5   # tn/(tn+fp) = 1/2
    assert res["f1"] == 0.8            # 2*2/3*1/(2/3+1) = 4/5
    assert res["accuracy"] == 3 / 4


def test_predictive_horizon_hand_derived():
    # Monotonic capacity fade: cap(c) = 2.0 - 0.1c -> EOL=1.5 crossed at c=6.
    # Score rises as failure approaches: score(c) = 1/(d-c+1) with d=6.
    # Evaluate only healthy cycles c<6. H=2 -> events at c=4,5 (d-c<=2).
    cycles = np.arange(20)
    cap = 2.0 - 0.1 * cycles
    d = 6  # first cycle with cap < 1.5
    score = np.array([1.0 / (d - c + 1) for c in cycles])
    df = pd.DataFrame({"cycle": cycles, "capacity": cap, "score": score})
    res = predictive_horizon_metrics(df, score_col="score", capacity_col="capacity",
                                     cycle_col="cycle", horizons=(2,), eol_fraction=0.75,
                                     score_threshold=0.3)
    r = res[2]
    # Healthy cycles 0..5; events at c=4,5; scores [0.143,0.167,0.2,0.25,0.333,0.5]
    # all events outscore all non-events -> AUC=1.0. threshold 0.3 -> flags at 4,5.
    assert r["n_healthy"] == 6
    assert r["n_events"] == 2
    assert r["roc_auc"] == 1.0, r
    assert r["pr_auc"] == 1.0, r
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f1"] == 1.0
    assert r["specificity"] == 1.0


def test_predictive_horizon_needs_fresh_initial_capacity():
    # Regression: when df starts mid-life (test split), EOL must come from the
    # battery's FRESH initial capacity, not the first cycle in df. Using the
    # first df cycle as reference would silently find the wrong degradation
    # horizon (the real Arm E bug: n_events=0 because EOL was anchored to the
    # already-faded test-set capacity).
    cycles = np.arange(100, 120)  # mid-life test split, cycle 100..119
    # Fresh capacity 2.0 -> EOL 1.5. Test-set capacity is already 1.4..1.2
    # (all BELOW EOL: the battery is degraded throughout this window).
    cap = 1.4 - 0.01 * (cycles - 100)  # 1.4 at cycle 100, ~1.2 at 119
    score = np.array([0.1] * 10 + [0.9] * 10)
    df = pd.DataFrame({"cycle": cycles, "capacity": cap, "score": score})
    # Wrong (no fresh capacity): EOL anchored to 0.75*1.4=1.05 -> finds a fake
    # "first degraded" cycle near the end and reports events that don't exist.
    res_bad = predictive_horizon_metrics(df, score_col="score", capacity_col="capacity",
                                         cycle_col="cycle", horizons=(5,),
                                         eol_fraction=0.75, score_threshold=0.5)
    # Right: with fresh capacity 2.0, EOL=1.5, every cycle is already degraded
    # -> zero healthy cycles remain to predict from (honest "already degraded").
    res_good = predictive_horizon_metrics(df, score_col="score", capacity_col="capacity",
                                          cycle_col="cycle", horizons=(5,),
                                          eol_fraction=0.75, score_threshold=0.5,
                                          initial_capacity=2.0)
    assert res_good[5]["n_healthy"] == 0 and "note" in res_good[5], res_good
    assert res_bad[5]["n_healthy"] >= 1, res_bad  # wrong reference invents events


def test_predictive_horizon_imperfect_threshold():
    # Same data, looser threshold 0.24 -> c=3 (score 0.25) also flags.
    cycles = np.arange(20)
    cap = 2.0 - 0.1 * cycles
    d = 6
    score = np.array([1.0 / (d - c + 1) for c in cycles])
    df = pd.DataFrame({"cycle": cycles, "capacity": cap, "score": score})
    res = predictive_horizon_metrics(df, score_col="score", capacity_col="capacity",
                                     cycle_col="cycle", horizons=(2,), eol_fraction=0.75,
                                     score_threshold=0.24)
    r = res[2]
    # flags at c=3,4,5 -> tp=2, fp=1, fn=0, tn=3
    assert r["n_healthy"] == 6 and r["n_events"] == 2
    assert r["precision"] == 2 / 3
    assert r["recall"] == 1.0
    assert r["specificity"] == 3 / 4
    assert r["f1"] == 4 / 5


if __name__ == "__main__":
    tests = [test_specificity_in_basic_metrics,
             test_cycle_level_metrics_perfect_separation,
             test_cycle_level_metrics_imperfect,
             test_predictive_horizon_hand_derived,
             test_predictive_horizon_imperfect_threshold,
             test_predictive_horizon_needs_fresh_initial_capacity]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print("\nAll ML-metrics tests PASS")

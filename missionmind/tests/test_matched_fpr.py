"""Matched-FPR cycle-level metrics (user request).

The first question an ML reviewer asks about an anomaly detector is "what
precision / recall do you get at a matched false-positive rate?" - i.e.
choose the decision threshold FROM THE HEALTHY-CLASS SCORE DISTRIBUTION so
that a target fraction (here 5%) of healthy units are flagged, then report
precision/recall/F1/specificity at THAT operating point. A threshold tuned
to a contamination prior (e.g. 0.07) is not the same thing and reviewers
know it.

These tests cover:
  1. matched_fpr_metrics      - threshold + confusion at target FPR
  2. perfect separation       - clean FPR=0, P/R/F1 = 1
  3. cycle_matched_fpr_metrics- cycle-level wrapper (per-cycle mean score,
                                max label), the unit used in Arm B

All expected values are hand-derived literals.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from missionmind.ml.metrics import matched_fpr_metrics, cycle_matched_fpr_metrics


def test_matched_fpr_hand_derived():
    # healthy scores [1,2,3,4], degraded [5,6,7,8], target FPR 0.5
    # tau = 50th percentile of healthy = 2.5 ; flag = score > tau
    # -> fp=2 (3,4), tn=2, tp=4, fn=0 ; achieved FPR = 2/4 = 0.5
    res = matched_fpr_metrics(
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        fpr_target=0.5,
    )
    assert res["threshold"] == 2.5, res
    assert res["achieved_fpr"] == 0.5, res
    assert res["tp"] == 4 and res["fp"] == 2 and res["fn"] == 0 and res["tn"] == 2, res
    assert res["precision"] == 4 / 6, res   # tp/(tp+fp) = 4/6 = 2/3
    assert res["recall"] == 1.0, res        # tp/(tp+fn)
    assert res["specificity"] == 0.5, res   # tn/(tn+fp) = 2/4
    assert res["f1"] == 0.8, res            # 2*P*R/(P+R) = 2*(2/3)/(5/3)


def test_matched_fpr_perfect_separation():
    # healthy all at 0.1, degraded all at 0.9: nothing healthy can exceed
    # the 95th percentile of healthy scores -> FPR 0, perfect P/R/F1.
    y_true = np.array([0] * 10 + [1] * 5)
    y_score = np.array([0.1] * 10 + [0.9] * 5)
    res = matched_fpr_metrics(y_true, y_score, fpr_target=0.05)
    assert res["threshold"] == 0.1, res
    assert res["achieved_fpr"] == 0.0, res
    assert res["precision"] == 1.0 and res["recall"] == 1.0, res
    assert res["specificity"] == 1.0 and res["f1"] == 1.0, res


def test_cycle_matched_fpr_wrapper():
    # 8 cycles x 2 rows. Per-cycle mean scores: [1.25,2.25,3.25,4.25,
    # 5.25,6.25,7.25,8.25]; per-cycle label = max(row labels):
    # [0,0,0,0,1,1,1,1] -> healthy [1.25,2.25,3.25,4.25], target 0.5
    # tau = median = 2.75 ; flag = score > 2.75 -> healthy flagged: 3.25,
    # 4.25 (fp=2), all 4 degraded flagged (tp=4).
    df = pd.DataFrame({
        "cycle_idx": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
        "score":     [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5],
        "label":     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    })
    res = cycle_matched_fpr_metrics(df, score_col="score", label_col="label",
                                    cycle_col="cycle_idx", fpr_target=0.5)
    assert res["n_cycles"] == 8, res
    assert res["n_degraded_cycles"] == 4, res
    assert res["threshold"] == 2.75, res
    assert res["achieved_fpr"] == 0.5, res
    assert res["tp"] == 4 and res["fp"] == 2 and res["fn"] == 0 and res["tn"] == 2, res
    assert res["precision"] == 4 / 6, res
    assert res["recall"] == 1.0, res
    assert res["f1"] == 0.8, res


def test_matched_fpr_no_healthy_cycles():
    # No healthy reference -> cannot match FPR; must not crash.
    res = matched_fpr_metrics(np.array([1, 1, 1]), np.array([5, 6, 7]), fpr_target=0.05)
    assert res is None, res


if __name__ == "__main__":
    tests = [test_matched_fpr_hand_derived,
             test_matched_fpr_perfect_separation,
             test_cycle_matched_fpr_wrapper,
             test_matched_fpr_no_healthy_cycles]
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
    print("\nAll matched-FPR tests PASS")

"""TDD test suite for the PINN architecture x lambda sweep.

What we are asserting (one test per behaviour):

  test_sweep_rows_shape      : `sweep_pinn_architectures` returns one row per
                               (hidden, lam) config with the documented keys
                               and a correct min(AUC,|Sp|) field.
  test_sweep_covers_grid     : a 2-hidden x 2-lambda grid yields exactly 4
                               rows and the expected (hidden, lam) pairs.
  test_sweep_on_real_b0005   : end-to-end on real NASA B0005 (small grid)
                               produces finite AUC/Spearman rows — proving the
                               sweep path actually runs against the real data.
  test_arch_verdict_matches  : `architecture_verdict` picks the best
                               min(AUC,|Sp|) config and reproduces the
                               PGNN-vs-PINN verdict sign from the stored
                               reference JSON.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from missionmind.ml.pinn_vs_pgnn import (
    sweep_pinn_architectures, architecture_verdict,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def test_sweep_rows_shape():
    rows = sweep_pinn_architectures(
        hidden_list=[(16,), (32, 16)], lam_list=[0.0, 0.5],
        _evaluator=lambda lam, hidden, epochs: (0.9, -0.8, 0.35),
    )
    assert len(rows) == 4
    for r in rows:
        assert "hidden" in r and "lam" in r and "auc" in r
        assert "spearman" in r and "abs_sp" in r and "min_metric" in r
        # min(AUC, |Sp|) must be computed exactly
        expected = min(r["auc"], abs(r["spearman"]))
        assert abs(r["min_metric"] - expected) < 1e-12, f"min_metric wrong: {r}"
    print("  PASS sweep row shape + min_metric")


def test_sweep_covers_grid():
    rows = sweep_pinn_architectures(
        hidden_list=[(16,), (64, 32, 16)], lam_list=[0.0, 1.0],
        _evaluator=lambda lam, hidden, epochs: (0.5, 0.1, 0.2),
    )
    pairs = {(tuple(r["hidden"]), r["lam"]) for r in rows}
    expected = {((16,), 0.0), ((16,), 1.0), ((64, 32, 16), 0.0), ((64, 32, 16), 1.0)}
    assert pairs == expected, f"grid mismatch: {pairs} != {expected}"
    print("  PASS grid coverage")


def test_sweep_on_real_b0005():
    from missionmind.ml.nasa_real_validation import load_battery
    b5 = load_battery("B0005")
    rows = sweep_pinn_architectures(
        b5, hidden_list=[(16,), (32, 16)], lam_list=[0.0, 0.5],
        epochs=30,  # lighter epochs so the test stays fast
    )
    assert len(rows) == 4
    for r in rows:
        assert np.isfinite(r["auc"]) and np.isfinite(r["spearman"]), r
        assert 0.0 <= r["auc"] <= 1.0, r
    print("  PASS real B0005 sweep (finite, in-range)")


def test_arch_verdict_matches():
    # Stored reference: PGNN min(AUC,|Sp|)=0.7892, best PINN=0.2854, verdict NO.
    path = os.path.join(MODELS_DIR, "pinn_vs_pgnn_b0005.json")
    if not os.path.exists(path):
        print("  SKIP stored reference absent")
        return
    with open(path) as f:
        ref = json.load(f)
    rows = sweep_pinn_architectures(
        hidden_list=[(16,)], lam_list=[0.3],
        _evaluator=lambda lam, hidden, epochs: (ref["pinn"][1]["auc"],
                                                ref["pinn"][1]["spearman"], 0.35),
    )
    v = architecture_verdict(ref["pgnn"]["min_metric"], rows)
    assert "best_config" in v and "delta" in v and "verdict" in v
    assert v["delta"] < 0, "stored reference must show PINN losing"
    assert "NOT beat" in v["verdict"]
    print(f"  PASS verdict reproduces stored sign: delta={v['delta']:+.4f}")


if __name__ == "__main__":
    print("=" * 76)
    print("PINN ARCHITECTURE SWEEP — TDD TEST SUITE")
    print("=" * 76)
    test_sweep_rows_shape()
    test_sweep_covers_grid()
    test_sweep_on_real_b0005()
    test_arch_verdict_matches()
    print("ALL 4 ASSERTIONS PASS")

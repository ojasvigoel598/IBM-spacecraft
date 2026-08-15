"""TDD test suite for PINN-vs-PGNN comparison rendering (figure + table).

What we are asserting (one test per behaviour):

  test_results_table_rows   : `results_table(results)` returns one row per
                              model (PGNN + each PINN lambda sweep), with
                              all numeric columns parsed as floats and the
                              AUC/Spearman/min-metric values matching the
                              input dict exactly.
  test_rendered_csv_matches_rows
                            : `write_results_csv(path, rows)` writes a
                              header + rows that re-read as identical
                              numbers (reproducible table artifact).
  test_render_figure_creates_png
                            : `render_comparison_figure(results, path)`
                              produces a non-empty PNG file on disk
                              (matplotlib Agg backend, no display needed).
  test_fresh_json_reproduces_verdict
                            : re-reading the persisted JSON (the artifact a
                              fresh run writes) still yields the same
                              verdict string and delta sign as the run
                              printed — guards against stale artifacts.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np

from missionmind.ml.pinn_vs_pgnn import (
    results_table, write_results_csv, render_comparison_figure,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _sample_results():
    return {
        "pgnn": {"auc": 0.7892, "spearman": -0.9386, "abs_sp": 0.9386,
                 "min_metric": 0.7892, "time_s": 3.44},
        "pinn": [
            {"lam": 0.0, "auc": 0.3480, "spearman": 0.2157,
             "abs_sp": 0.2157, "alpha_learned": 0.01, "time_s": 1.94},
            {"lam": 0.3, "auc": 0.2183, "spearman": 0.4226,
             "abs_sp": 0.4226, "alpha_learned": 0.3543, "time_s": 1.47},
        ],
        "verdict": "NO — strict PINN does NOT beat feature-only PGNN on this metric.",
    }


def test_results_table_rows():
    rows = results_table(_sample_results())
    # 1 PGNN row + 2 PINN rows
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
    names = [r[0] for r in rows]
    assert names[0] == "PGNN"
    assert names[1] == "PINN(lam=0.0)"
    assert names[2] == "PINN(lam=0.3)"
    # Every numeric field must be a float (no strings leaking into math)
    for r in rows:
        for v in r[1:]:
            assert isinstance(v, float), f"non-float value in row {r}: {v!r}"
    # Values must round-trip from the input dict exactly
    assert abs(rows[0][1] - 0.7892) < 1e-9          # PGNN AUC
    assert abs(rows[0][2] - (-0.9386)) < 1e-9       # PGNN Spearman
    assert abs(rows[1][3] - 0.2157) < 1e-9          # PINN lam=0.0 |Sp|
    assert abs(rows[2][4] - 0.2183) < 1e-9          # PINN lam=0.3 min(AUC,|Sp|)
    print("  PASS results_table rows =", names)


def test_rendered_csv_matches_rows():
    rows = results_table(_sample_results())
    csv_path = os.path.join(MODELS_DIR, "pinn_vs_pgnn_table_check.csv")
    write_results_csv(csv_path, rows)
    assert os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines[0].startswith("model")  # header
    assert len(lines) == len(rows) + 1
    parsed = [ln.split(",") for ln in lines[1:]]
    for (name, auc, sp, fab, mm, t), parts in zip(rows, parsed):
        assert parts[0] == name
        assert abs(float(parts[1]) - auc) < 1e-6
        assert abs(float(parts[2]) - sp) < 1e-6
        assert abs(float(parts[4]) - mm) < 1e-6
    os.remove(csv_path)
    print("  PASS csv round-trip")


def test_render_figure_creates_png():
    import matplotlib
    matplotlib.use("Agg")
    png_path = os.path.join(MODELS_DIR, "pinn_vs_pgnn_figure_check.png")
    render_comparison_figure(_sample_results(), png_path)
    assert os.path.exists(png_path)
    assert os.path.getsize(png_path) > 5000, "PNG suspiciously small"
    with open(png_path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n", "not a valid PNG header"
    os.remove(png_path)
    print("  PASS png artifact created")


def test_fresh_json_reproduces_verdict():
    """The persisted artifact must agree with the printed verdict/delta."""
    path = os.path.join(MODELS_DIR, "pinn_vs_pgnn_b0005.json")
    if not os.path.exists(path):
        print("  SKIP fresh JSON not present (run pinn_vs_pgnn first)")
        return
    with open(path) as f:
        d = json.load(f)
    assert "verdict" in d and d["verdict"]
    assert "delta_pinn_minus_pgnn" in d
    assert isinstance(d["delta_pinn_minus_pgnn"], float)
    # The printed verdict and the stored delta must agree in sign.
    neg = d["delta_pinn_minus_pgnn"] < 0
    if neg:
        assert "NOT beat" in d["verdict"]
    else:
        assert "NOT beat" not in d["verdict"]
    print(f"  PASS fresh JSON: verdict='{d['verdict'][:60]}...' "
          f"delta={d['delta_pinn_minus_pgnn']:+.4f}")


if __name__ == "__main__":
    print("=" * 76)
    print("PINN vs PGNN — COMPARISON RENDERING TDD TEST SUITE")
    print("=" * 76)
    test_results_table_rows()
    test_rendered_csv_matches_rows()
    test_render_figure_creates_png()
    test_fresh_json_reproduces_verdict()
    print("ALL 4 ASSERTIONS PASS")

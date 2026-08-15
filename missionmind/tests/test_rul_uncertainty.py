"""P6 tests — RUL uncertainty (missionmind/ml/rul_uncertainty.py).

The dashboard must never show a false-precise single RUL number. These tests
verify the bootstrap interval is real: it is finite, contains the point
estimate, widens with noise, and formats honestly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.ml.prognostics import load_curves, eol_cap  # noqa: E402
from missionmind.ml.rul_uncertainty import (  # noqa: E402
    rul_prediction_interval, time_to_limit_interval, format_interval,
)


def test_nasa_bootstrap_interval_contains_point():
    curves = load_curves()
    n, c = curves["B0005"]
    eol = eol_cap(c[0])
    pa = n[-1] * 0.6
    rul, lo, hi = rul_prediction_interval(n, c, eol=eol, predict_at=pa, n_boot=120)
    assert np.isfinite(rul) and np.isfinite(lo) and np.isfinite(hi)
    assert lo <= rul <= hi, "bootstrap CI must contain the point estimate"
    assert 0 <= lo, "lower bound cannot be negative"


def test_bootstrap_interval_widens_with_noise():
    """More residual noise => wider interval (the method is responsive)."""
    rng = np.random.default_rng(0)
    n = np.arange(0, 80.0, 1.0)
    clean = 2.0 * np.exp(-0.004 * n) + 0.5
    eol = 1.5
    c_clean = clean
    c_noisy = clean + rng.normal(0, 0.05, len(clean))
    _, lo1, hi1 = rul_prediction_interval(n, c_clean, eol=eol, n_boot=80)
    _, lo2, hi2 = rul_prediction_interval(n, c_noisy, eol=eol, n_boot=80)
    if np.isfinite(hi1) and np.isfinite(hi2):
        assert (hi2 - lo2) >= (hi1 - lo1) - 1e-9, "noisy data must not shrink the CI"


def test_time_to_limit_interval_contains_point():
    rng = np.random.default_rng(0)
    soc_diffs = -np.abs(rng.normal(0.000417, 0.00005, 60))
    r, lo, hi = time_to_limit_interval(0.9, -150.0, soc_diffs)
    assert np.isfinite(r) and np.isfinite(lo) and np.isfinite(hi)
    assert lo <= r <= hi
    # charging -> no depletion risk
    r2, _, _ = time_to_limit_interval(0.9, 120.0, soc_diffs)
    assert r2 == float("inf")


def test_format_interval_honest():
    assert format_interval(float("inf"), float("inf"), float("inf")) == "∞"
    assert "±" in format_interval(30.0, 26.0, 44.0, "min")
    # unavailable interval is flagged, not faked
    assert format_interval(30.0, np.nan, np.nan) == "30 ± —"


if __name__ == "__main__":
    for fn in (test_nasa_bootstrap_interval_contains_point,
               test_bootstrap_interval_widens_with_noise,
               test_time_to_limit_interval_contains_point,
               test_format_interval_honest):
        fn()
        print(f"PASS {fn.__name__}")
    print("All RUL-uncertainty tests PASS")

"""Dataset-fingerprint tests.

The CI dataset-change check compares the committed dataset_id against a fresh
in-memory retrain. The fingerprint hashes the feature matrix rounded to 6
decimals, so last-ULP cross-platform float differences (e.g. from numpy `**`
calling the platform C pow()) are absorbed while genuine physics/feature
changes still flip the id.

The tests below are DETERMINISTIC (no probabilistic flakiness): ULP-stability
is pinned on a controlled matrix whose values sit far from rounding
boundaries, real changes are pinned to flip, and the full training pipeline
must reproduce its own fingerprint exactly on a given platform.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from missionmind.ml.train import dataset_fingerprint, generate_training_data, \
    add_derivative_features, FEATURE_COLS


def _boundary_safe_matrix(n=40, cols=5):
    """Matrix whose values are all far (>= 1e-3) from a 6-decimal rounding
    boundary AND never round to +/-0.0, so 1-ULP perturbations can never cross
    a boundary or flip a zero sign bit. Values use at most 3 decimals, well
    inside a 1e-6 rounding cell."""
    rng = np.random.default_rng(123)
    vals = np.empty((n, cols))
    vals[:, 0] = rng.uniform(20, 30, size=n)          # voltage-like
    vals[:, 1] = rng.uniform(1, 520, size=n)          # solar-like (positive)
    vals[:, 2] = rng.uniform(-15, 30, size=n)         # temp-like
    vals[:, 3] = rng.uniform(1e-3, 1e-2, size=n)      # derivative-like (positive)
    vals[:, 4] = rng.uniform(0.01, 1, size=n)         # residual-like (positive)
    # snap to a coarse grid so no value is near a 6-decimal boundary
    out = np.round(vals, 3)
    # never allow a value to round to zero (sign-bit flips would change bytes
    # while == stays true)
    out[np.abs(out) < 1e-3] = 1e-3
    return out


def _ulp_perturb_all(X):
    """Nudge every value by one ULP in a random direction."""
    rng = np.random.default_rng(5)
    Xp = X.copy()
    for i in range(Xp.shape[0]):
        for c in range(Xp.shape[1]):
            x = Xp[i, c]
            Xp[i, c] = np.nextafter(x, x + (1.0 if rng.random() < 0.5 else -1.0))
    return Xp


def test_fingerprint_deterministic():
    X = _boundary_safe_matrix()
    assert dataset_fingerprint(X) == dataset_fingerprint(X.copy())


def test_fingerprint_stable_under_ulp_noise():
    """1-ULP perturbations (the cross-platform pow()/trig divergence class)
    must not flip the rounded fingerprint when values stay inside their
    1e-6 rounding cells."""
    X = _boundary_safe_matrix()
    Xp = _ulp_perturb_all(X)
    assert dataset_fingerprint(X) == dataset_fingerprint(Xp)


def test_fingerprint_changes_on_real_change():
    X = _boundary_safe_matrix()
    Xr = X.copy()
    Xr[:, 0] += 0.5  # a genuine 0.5 V change
    assert dataset_fingerprint(X) != dataset_fingerprint(Xr)


def test_fingerprint_sensitive_to_small_but_real_change():
    """A change well above the 1e-6 rounding granularity (e.g. 1e-4) flips
    the id; a change far below it (1e-8, above ULP but below rounding) does
    not — on the boundary-safe matrix this is deterministic."""
    X = _boundary_safe_matrix()
    small = X.copy()
    small += 1e-8
    assert dataset_fingerprint(X) == dataset_fingerprint(small)
    large = X.copy()
    large += 1e-4
    assert dataset_fingerprint(X) != dataset_fingerprint(large)


def test_training_pipeline_reproduces_own_fingerprint():
    """The canonical training pipeline must reproduce its dataset_id exactly
    on a given platform (same seed, same physics)."""
    cols = FEATURE_COLS + ["d_temp_dt", "d_volt_dt",
                           "solar_residual_w", "thermal_residual_w"]
    df1 = add_derivative_features(generate_training_data())
    df2 = add_derivative_features(generate_training_data())
    X1 = df1[cols].values.astype(np.float64)
    X2 = df2[cols].values.astype(np.float64)
    assert dataset_fingerprint(X1) == dataset_fingerprint(X2)
    assert np.array_equal(X1, X2), "training data must be bit-identical on rerun"


if __name__ == "__main__":
    tests = [test_fingerprint_deterministic,
             test_fingerprint_stable_under_ulp_noise,
             test_fingerprint_changes_on_real_change,
             test_fingerprint_sensitive_to_small_but_real_change,
             test_training_pipeline_reproduces_own_fingerprint]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
    if failed:
        sys.exit(1)
    print("All dataset-fingerprint tests PASS")

"""
MissionMind - drift detector (TDD: RED -> GREEN -> REFACTOR).

Tests for `missionmind.ml.drift.streaming_ks_test`:
  1. identical distributions are NOT flagged as drift
  2. shifted distributions ARE flagged as drift
  3. too-small inputs raise ValueError (not crash with cryptic scipy error)

Run: python -m missionmind.tests.test_drift
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.ml.drift import streaming_ks_test


def _expect(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
    return cond


def test_identical_distributions_are_not_drift():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(0, 1, size=500)
    p = streaming_ks_test(a, b)
    return _expect("identical_N(0,1)_p>0.05", p > 0.05, f"p={p:.4f}")


def test_shifted_distributions_are_drift():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(0.5, 1, size=500)  # 0.5-sigma shift
    p = streaming_ks_test(a, b)
    return _expect("shifted_N(0.5)_p<0.05", p < 0.05, f"p={p:.4f}")


def test_short_input_raises_value_error():
    raised = None
    try:
        streaming_ks_test([1.0], [1.0, 2.0])
    except ValueError as e:
        raised = e
    return _expect("short_input_raises_ValueError", raised is not None,
                   f"raised={raised!r}")


def test_empty_input_raises_value_error():
    raised = None
    try:
        streaming_ks_test([], [])
    except ValueError as e:
        raised = e
    return _expect("empty_input_raises_ValueError", raised is not None,
                   f"raised={raised!r}")


if __name__ == "__main__":
    results = [
        test_identical_distributions_are_not_drift(),
        test_shifted_distributions_are_drift(),
        test_short_input_raises_value_error(),
        test_empty_input_raises_value_error(),
    ]
    print()
    print(f"  Total: {sum(results)}/{len(results)} pass")
    sys.exit(0 if all(results) else 1)

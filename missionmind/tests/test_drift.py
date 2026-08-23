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


def test_identical_distributions_are_not_drift():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(0, 1, size=500)
    p = streaming_ks_test(a, b)
    assert p > 0.05, f"identical distributions should not show drift (p={p:.4f})"


def test_shifted_distributions_are_drift():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(0.5, 1, size=500)  # 0.5-sigma shift
    p = streaming_ks_test(a, b)
    assert p < 0.05, f"shifted distributions should show drift (p={p:.4f})"


def test_short_input_raises_value_error():
    try:
        streaming_ks_test([1.0], [1.0, 2.0])
        assert False, "short input should raise ValueError"
    except ValueError:
        pass


def test_empty_input_raises_value_error():
    try:
        streaming_ks_test([], [])
        assert False, "empty input should raise ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

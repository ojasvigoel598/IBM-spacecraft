"""
Prognostics evaluation protocol tests.

Scientific-validity bug: true RUL was computed inline as
`eol_cycle - predict_at`, which goes NEGATIVE when the prediction point is
past EOL (B0006 reaches EOL at cycle 72 of 168, so F=60%/80% predict after
the battery is dead). RUL is by definition >= 0; a model correctly reporting
0 for an already-failed battery must not be penalized by abs() of a negative
label. The label is clamped at 0 via the shared true_rul_at() helper used by
both eval functions.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.ml.prognostics import true_rul_at


def test_true_rul_healthy_prediction_point():
    # EOL at cycle 72, predicting at cycle 66.8 -> 5.2 cycles remaining.
    assert abs(true_rul_at(eol_cycle=72, predict_at=66.8) - 5.2) < 1e-9


def test_true_rul_exact_eol_is_zero():
    assert true_rul_at(eol_cycle=72, predict_at=72) == 0.0


def test_true_rul_past_eol_clamped_to_zero():
    # The bug: this used to return -28.2 / -61.6 and inflate abs() error.
    assert true_rul_at(eol_cycle=72, predict_at=100.2) == 0.0
    assert true_rul_at(eol_cycle=72, predict_at=133.6) == 0.0
    assert true_rul_at(eol_cycle=125, predict_at=133.6) == 0.0


if __name__ == "__main__":
    tests = [test_true_rul_healthy_prediction_point,
             test_true_rul_exact_eol_is_zero,
             test_true_rul_past_eol_clamped_to_zero]
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
    print("All prognostics protocol tests PASS")

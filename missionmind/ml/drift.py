"""MissionMind - streaming drift detection helpers.

Currently exposes one function:

  streaming_ks_test(a, b) -> float
      Two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp). Returns
      the two-sided p-value; callers compare against the chosen alpha
      (commonly 0.05). Used to detect drift between a held-out training
      window and a recent window of streaming telemetry.

This module was added with TDD discipline: see missionmind/tests/test_drift.py
for the failing-first-then-pass suite.
"""

import numpy as np
from scipy.stats import ks_2samp


def streaming_ks_test(a, b):
    """Two-sample KS test between `a` (e.g. training window) and `b`
    (e.g. recent streaming window). Returns the p-value.

    Raises ValueError if either window has fewer than 2 samples, which
    is what scipy requires; we surface a domain-specific error message
    instead of the cryptic 'data must be at least size 2' from scipy.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError(
            f"streaming_ks_test requires >=2 samples per window; "
            f"got a.size={a.size}, b.size={b.size}"
        )
    result = ks_2samp(a, b)
    return float(result.pvalue)

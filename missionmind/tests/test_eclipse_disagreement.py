"""P5 tests — ML-vs-physics disagreement exposure (adaptive layer).

If the ML ensemble flags a solar dip while Kepler physics says the satellite
is in eclipse, the decision layer must expose the disagreement (strategy
ECLIPSE_EXPLAINED, flag suppressed) rather than hide it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd  # noqa: E402

from missionmind.simulator.run_scenarios import run_scenario  # noqa: E402
from missionmind.physics_rules.rules import check_eclipse  # noqa: E402
from missionmind.ml.adaptive import decide  # noqa: E402


def _eclipse_window():
    """Deterministic eclipse window: in-eclipse telemetry with low solar."""
    df = run_scenario(failure_mode="none", duration_s=120, add_orbit=True)
    win = df.head(40).copy()
    win["solar_power_w"] = 10.0
    win["in_eclipse"] = 1
    return win


def test_check_eclipse_fires_on_synthetic_eclipse():
    win = _eclipse_window()
    res = check_eclipse(win)
    assert res is not None and res[0] == "eclipse"
    assert 0.5 < res[1] <= 0.95


def test_check_eclipse_none_on_nominal():
    df = run_scenario(failure_mode="none", duration_s=120, add_orbit=True)
    assert check_eclipse(df.head(40)) is None


def test_adaptive_exposes_disagreement():
    win = _eclipse_window()
    d = decide(win)
    assert d["strategy"] == "ECLIPSE_EXPLAINED"
    assert d["adaptive_flag"] == 0, "expected transient must not flag as fault"
    joined = " ".join(d["reasoning"]).lower()
    assert "eclipse" in joined and "not a fault" in joined


def test_eclipse_rule_requires_orbit_column():
    # a window without the orbit column must not crash, just return None
    df = run_scenario(failure_mode="none", duration_s=120, add_orbit=False)
    win = df.head(40).copy()
    win["solar_power_w"] = 10.0
    assert check_eclipse(win) is None


if __name__ == "__main__":
    for fn in (test_check_eclipse_fires_on_synthetic_eclipse,
               test_check_eclipse_none_on_nominal,
               test_adaptive_exposes_disagreement,
               test_eclipse_rule_requires_orbit_column):
        fn()
        print(f"PASS {fn.__name__}")
    print("All eclipse-disagreement tests PASS")

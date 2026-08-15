"""TDD suite — adaptive decision layer (missionmind/ml/adaptive.py).

Seam under test: the public `decide()` interface — situation-aware fusion that
sits on top of the production 3-forest ensemble + physics rule layer. The
decision layer must pick a strategy per situation (rule-first when physics
confirms a subsystem, early-detection-sensitive during the fault ramp,
silent in nominal ops) and always return an explainable decision.

Expected to FAIL (ImportError) until the module is implemented.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.simulator.run_scenarios import run_scenario
from missionmind.ml.detect import score_dataframe


def _window(mode, t, w=120):
    df = run_scenario(failure_mode=mode, duration_s=3600)
    df = score_dataframe(df)
    win = df[(df["time_s"] >= t - w) & (df["time_s"] <= t)]
    return win


def main():
    from missionmind.ml.adaptive import decide

    # 1. nominal stays silent after burn-in (no false alarm burden)
    d = decide(_window("none", 500))
    assert d["adaptive_flag"] == 0, f"nominal window flagged: {d}"

    # 2. solar fault at t=1500 -> flagged, and routed via a real strategy
    d = decide(_window("solar_degradation", 1500))
    assert d["adaptive_flag"] == 1, f"solar window not flagged: {d}"
    assert d["strategy"] in ("RULE_POWER", "RULE_THERMAL", "AGREE_2OF3",
                            "RAMP_LEAD", "DETECTOR_CONSENSUS"), d["strategy"]

    # 3. radiator fault at t=3000 -> flagged
    d = decide(_window("radiator_degradation", 3000))
    assert d["adaptive_flag"] == 1, f"radiator window not flagged: {d}"

    # 4. explainable decision: strategy + reasoning lines + a score
    assert "strategy" in d and "adaptive_score" in d and "adaptive_flag" in d
    assert isinstance(d.get("reasoning"), list) and len(d["reasoning"]) >= 2, d
    assert all(isinstance(line, str) and line for line in d["reasoning"]), d

    # 5. fault ramp (600-900 s) is early-detection sensitive -> RAMP_LEAD
    d = decide(_window("solar_degradation", 750))
    assert d["strategy"] == "RAMP_LEAD", d["strategy"]

    print("All adaptive-layer tests PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

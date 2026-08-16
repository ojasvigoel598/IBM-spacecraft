"""Shared pytest fixtures for the MissionMind test suite.

The suite's CSV-dependent tests read the three scenario telemetry files
(run_normal.csv, run_solar_failure.csv, run_radiator_failure.csv). Those
files are gitignored by design (regenerable via
`python -m missionmind.simulator.run_scenarios`), so a fresh checkout does
not contain them. This module regenerates any that are missing so the
suite stays green on a fresh clone and in CI. Generation is fast
(~0.3 s per scenario) and writes only to gitignored paths, so the repo's
phantom-diff guard stays clean.
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(TESTS_DIR, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "missionmind", "data")

# Same mapping as missionmind/simulator/run_scenarios.py::main().
_SCENARIOS = {
    "none": "run_normal.csv",
    "solar_degradation": "run_solar_failure.csv",
    "radiator_degradation": "run_radiator_failure.csv",
}


def _generate_scenario_csvs() -> None:
    """Regenerate the gitignored scenario CSVs (same calls as the simulator CLI)."""
    from missionmind.simulator.run_scenarios import run_scenario

    for mode, fname in _SCENARIOS.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            continue
        df = run_scenario(failure_mode=mode, duration_s=3600)
        df.to_csv(path, index=False)


def pytest_sessionstart(session) -> None:
    missing = [f for f in _SCENARIOS.values()
               if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        _generate_scenario_csvs()

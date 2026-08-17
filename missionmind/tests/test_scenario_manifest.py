"""Scenario CSV versioning tests (simulator/run_scenarios.py).

The scenario CSVs are gitignored, regenerable artifacts. The manifest records
which simulator version generated them so a stale CSV (old physics) is
detectable instead of silently replaying into the pipeline. These tests cover
the signature, the write/check round trip, tamper detection and the
no-manifest case.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.simulator.run_scenarios import (
    generator_signature, write_scenario_manifest, check_stale_scenarios,
    SCENARIOS,
)


def test_signature_stable_and_bounded():
    assert generator_signature() == generator_signature()
    assert len(generator_signature()) == 16


def test_manifest_current_not_stale(tmp_path):
    path = str(tmp_path / "manifest.json")
    n_rows = {fname: 3600 for fname in SCENARIOS.values()}
    cur = write_scenario_manifest(n_rows, add_noise=False, duration_s=3600, path=path)
    assert cur == generator_signature()
    assert check_stale_scenarios(path) == []


def test_tampered_manifest_reports_stale(tmp_path):
    path = str(tmp_path / "manifest.json")
    n_rows = {fname: 3600 for fname in SCENARIOS.values()}
    write_scenario_manifest(n_rows, add_noise=False, duration_s=3600, path=path)
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    for fname in SCENARIOS.values():
        m[fname]["simulator_hash"] = "0" * 16
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f)
    stale = check_stale_scenarios(path)
    assert sorted(stale) == sorted(SCENARIOS.values())


def test_missing_manifest_not_stale(tmp_path):
    # nothing recorded -> nothing to be stale about (fresh checkout)
    assert check_stale_scenarios(str(tmp_path / "does-not-exist.json")) == []


if __name__ == "__main__":
    tests = [test_signature_stable_and_bounded,
             test_manifest_current_not_stale,
             test_tampered_manifest_reports_stale,
             test_missing_manifest_not_stale]
    failed = []
    for t in tests:
        try:
            t(__import__("tempfile").TemporaryDirectory().__enter__())
            print(f"PASS {t.__name__}")
        except TypeError:
            # tmp_path fixture unavailable in __main__; run with a temp dir
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                t(d)
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print("\nAll scenario-manifest tests PASS")

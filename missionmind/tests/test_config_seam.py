"""
Architecture review candidate 1 — collapse the constants seam.

Target architecture:
- simulator/config.py is the single source of truth for physics constants.
- power.py, thermal.py, failures.py bind config's objects directly and carry
  NO private fallback copies. If config cannot be imported, the failure is
  loud (ImportError), never a silent stale copy.
- thermal.py does not shadow imported constants with local re-assignment
  (the previous dead seam: it imported ETA/EPSILON/... then overwrote them).
- physics_rules/rules.py imports P_SOLAR_MAX and the tuned thresholds from
  config instead of hardcoding them.

The identity assertions below fail on the current code (each consumer owns
its own float object instead of config's), which is exactly the drift bug
this seam collapse removes.
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from missionmind.simulator import config  # noqa: E402
from missionmind.simulator import power, thermal, failures  # noqa: E402
from missionmind.physics_rules import rules  # noqa: E402


def test_power_binds_config_constants():
    # power.py must use config's objects, not its own copies.
    assert power.P_SOLAR_MAX is config.P_SOLAR_MAX
    assert power.P_LOAD is config.P_LOAD
    assert power.E_CAP_WH is config.E_CAP_WH
    assert power.V_MIN is config.V_MIN
    assert power.V_MAX is config.V_MAX
    assert power.SOC_0 is config.SOC_0
    assert power.DT_S is config.DT_S  # currently re-assigned to 1.0 locally


def test_thermal_binds_config_constants():
    # thermal.py imports these then unconditionally re-assigns them
    # (dead seam). Each `is` fails on the current code.
    assert thermal.ETA is config.ETA
    assert thermal.EPSILON is config.EPSILON
    assert thermal.AREA is config.AREA
    assert thermal.SIGMA is config.SIGMA
    assert thermal.T_SPACE_K is config.T_SPACE_K
    assert thermal.T0_C is config.T0_C
    assert thermal.Q_IN_NOMINAL is config.Q_IN_NOMINAL
    assert thermal.MC_P is config.MC_P


def test_failures_binds_config_constants():
    assert failures.T_RAMP_START is config.T_RAMP_START
    assert failures.T_RAMP_END is config.T_RAMP_END
    assert failures.SOLAR_FINAL_FACTOR is config.SOLAR_FINAL_FACTOR
    assert failures.RADIATOR_FINAL_FRACTION is config.RADIATOR_FINAL_FRACTION
    assert failures.EPSILON_A_NOMINAL is config.EPSILON_A_NOMINAL
    assert failures.EPSILON_A_FINAL is config.EPSILON_A_FINAL


def test_rules_binds_config_thresholds():
    # rules.py currently hardcodes P_SOLAR_MAX and the tuned thresholds.
    assert rules.P_SOLAR_MAX is config.P_SOLAR_MAX
    assert rules.SOC_SLOPE_THRESHOLD_TUNED is config.SOC_SLOPE_THRESHOLD_TUNED
    assert rules.TEMP_SLOPE_THRESHOLD_TUNED is config.TEMP_SLOPE_THRESHOLD_TUNED
    assert rules.SOC_SLOPE_THRESHOLD_SPEC is config.SOC_SLOPE_THRESHOLD_SPEC
    assert rules.TEMP_SLOPE_THRESHOLD_SPEC is config.TEMP_SLOPE_THRESHOLD_SPEC


def test_config_failure_is_loud_not_silent():
    # A missing config must propagate ImportError; consumers must not
    # silently bind stale fallback copies.
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "import missionmind.simulator\n"
        "sys.modules['missionmind.simulator.config'] = None\n"
        "try:\n"
        "    import missionmind.simulator.power\n"
        "    import missionmind.simulator.thermal\n"
        "    import missionmind.simulator.failures\n"
        "except ImportError:\n"
        "    print('LOUD_FAIL')\n"
        "else:\n"
        "    print('SILENT_FALLBACK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=ROOT)
    assert "LOUD_FAIL" in r.stdout, (
        f"expected loud ImportError, got: {r.stdout!r} {r.stderr!r}")


def test_physics_values_agree_with_config():
    # Behavior guard: effective values must equal config exactly even if
    # the binding style changes later.
    assert power.P_SOLAR_MAX == config.P_SOLAR_MAX
    assert power.P_LOAD == config.P_LOAD
    assert power.E_CAP_WH == config.E_CAP_WH
    assert thermal.ETA == config.ETA
    assert thermal.EPSILON == config.EPSILON
    assert thermal.Q_IN_NOMINAL == config.Q_IN_NOMINAL
    assert failures.SOLAR_FINAL_FACTOR == config.SOLAR_FINAL_FACTOR
    assert failures.EPSILON_A_NOMINAL == config.EPSILON_A_NOMINAL


if __name__ == "__main__":
    tests = [
        test_power_binds_config_constants,
        test_thermal_binds_config_constants,
        test_failures_binds_config_constants,
        test_rules_binds_config_thresholds,
        test_config_failure_is_loud_not_silent,
        test_physics_values_agree_with_config,
    ]
    failures_list = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures_list.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
    if failures_list:
        print(f"\n{len(failures_list)} FAILED: {failures_list}")
        sys.exit(1)
    print("\nAll config-seam tests PASS")

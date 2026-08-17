"""
Prognostics evaluation protocol tests.

Scientific-validity bug: true RUL was computed inline as
`eol_cycle - predict_at`, which goes NEGATIVE when the prediction point is
past EOL (B0006 reaches EOL at cycle 72 of 168, so F=60%/80% predict after
the battery is dead). RUL is by definition >= 0; a model correctly reporting
0 for an already-failed battery must not be penalized by abs() of a negative
label. The label is clamped at 0 via the shared true_rul_at() helper used by
both eval functions.

Cycle definition: RUL is in EQUIVALENT FULL CYCLES (EFC), and the
prognostics->calendar-time conversion uses the EPS-measured EFC/orbit rate,
NOT the old assumption that one eclipse = one full cycle. The tests below pin
the shared EFC convention (accumulated discharge depth) and the measured-rate
conversion.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.ml.prognostics import (
    true_rul_at,
    equivalent_full_cycles_from_soc,
    efc_rate_per_orbit,
    estimate_dod_per_orbit,
    cycles_to_days,
    orbital_period_s,
)


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


def test_efc_full_discharge_is_one():
    # One full 1.0 -> 0.0 discharge = 1.0 equivalent full cycle.
    assert abs(equivalent_full_cycles_from_soc([1.0, 0.9, 0.5, 0.0]) - 1.0) < 1e-12


def test_efc_partial_cycles_accumulate():
    # Two 50% discharges = 1.0 EFC (standard equivalent-full-cycle convention).
    soc = [1.0, 0.5, 1.0, 0.5, 1.0]
    assert abs(equivalent_full_cycles_from_soc(soc) - 1.0) < 1e-12


def test_efc_ignores_charge_steps():
    # Charge steps (positive dSOC) never count toward EFC.
    soc = [0.0, 0.5, 0.9, 1.0, 1.0, 1.0]
    assert equivalent_full_cycles_from_soc(soc) == 0.0


def test_efc_rate_per_orbit_measures_actual_eps():
    # 0.5 EFC accumulated over half an orbit -> 1.0 EFC/orbit measured rate.
    t = [0.0, 100.0, 200.0, 300.0]
    soc = [1.0, 0.5, 0.5, 1.0]  # discharge 0.5, recharge; net 0.5 EFC
    period = 600.0
    assert abs(efc_rate_per_orbit(soc, t, period) - 1.0) < 1e-12


def test_estimate_dod_per_orbit_capped_at_one():
    # A 400 W load through a 2089 s eclipse on a 100 Wh (360 kJ) pack is a
    # 2.3x deficit -> capped at 1.0 (bus trips at SOC 0 under the EPS policy).
    assert estimate_dod_per_orbit(400.0, 2089.0, 360000.0) == 1.0


def test_estimate_dod_per_orbit_partial():
    # A light 20 W load through a 2000 s eclipse drains only 11% of the pack.
    dod = estimate_dod_per_orbit(20.0, 2000.0, 360000.0)
    assert abs(dod - (40000.0 / 360000.0)) < 1e-12


def test_cycles_to_days_uses_measured_rate_not_assumed_one():
    # 100 EFC at a measured 0.5 EFC/orbit -> 200 orbits -> 200 * T / 86400 days.
    T = orbital_period_s(550.0)
    days = cycles_to_days(100.0, altitude_km=550.0, efc_per_orbit=0.5)
    assert abs(days - (200.0 * T / 86400.0)) < 1e-9
    # The old assumption (1.0 EFC/orbit) would give half that time.
    days_old = cycles_to_days(100.0, altitude_km=550.0, efc_per_orbit=1.0)
    assert days_old == days / 2.0


def test_cycles_to_days_zero_rate_is_infinite():
    # No cycling -> no calendar-time conversion (never divide by zero).
    assert cycles_to_days(50.0, altitude_km=550.0, efc_per_orbit=0.0) == float("inf")


def test_cycles_to_days_fallback_is_finite_estimate():
    # Without a measured rate, the first-principles estimate still converts.
    days = cycles_to_days(50.0, altitude_km=550.0)
    assert 0.0 < days < 1e6


if __name__ == "__main__":
    tests = [test_true_rul_healthy_prediction_point,
             test_true_rul_exact_eol_is_zero,
             test_true_rul_past_eol_clamped_to_zero,
             test_efc_full_discharge_is_one,
             test_efc_partial_cycles_accumulate,
             test_efc_ignores_charge_steps,
             test_efc_rate_per_orbit_measures_actual_eps,
             test_estimate_dod_per_orbit_capped_at_one,
             test_estimate_dod_per_orbit_partial,
             test_cycles_to_days_uses_measured_rate_not_assumed_one,
             test_cycles_to_days_zero_rate_is_infinite,
             test_cycles_to_days_fallback_is_finite_estimate]
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

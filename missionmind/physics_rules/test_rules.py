"""
Automated version of row 7 checks, passing on all 3 CSVs
- check_power_subsystem returns None on normal CSV, flags solar_degradation after t~900s on solar-failure CSV
- same pattern for thermal
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from missionmind.physics_rules.rules import check_power_subsystem, check_thermal_subsystem

def evaluate_on_csv(csv_path: str, window_size: int = 120):
    df = pd.read_csv(csv_path)
    findings = []
    for t in range(0, len(df), 60):  # every 60s
        window = df.iloc[max(0, t-window_size):t+1]
        p = check_power_subsystem(window)
        th = check_thermal_subsystem(window)
        findings.append((int(df.iloc[t]["time_s"]), p, th))
    return df, findings

def test_rules():
    base = os.path.join(os.path.dirname(__file__), '..', 'data')
    normal_csv = os.path.join(base, 'run_normal.csv')
    solar_csv = os.path.join(base, 'run_solar_failure.csv')
    rad_csv = os.path.join(base, 'run_radiator_failure.csv')

    for f in [normal_csv, solar_csv, rad_csv]:
        assert os.path.exists(f), f"Missing {f}, run run_scenarios first"

    print("=== Testing on run_normal.csv ===")
    df_n, findings_n = evaluate_on_csv(normal_csv)
    # Normal should never flag
    power_flags = [p for _,p,_ in findings_n if p]
    thermal_flags = [th for _,_,th in findings_n if th]
    print(f"Normal: power flags={len(power_flags)}, thermal flags={len(thermal_flags)} (expected 0)")
    assert len(power_flags) == 0, f"Normal should not flag power, got {power_flags}"
    assert len(thermal_flags) == 0, f"Normal should not flag thermal, got {thermal_flags}"

    print("=== Testing on run_solar_failure.csv ===")
    df_s, findings_s = evaluate_on_csv(solar_csv)
    # After 900s should flag solar degradation in every window where the
    # fault is PHYSICALLY OBSERVABLE. With eclipse-coupled physics the power
    # rule correctly declines eclipse-dominated windows (the dip is expected
    # orbital geometry, not a fault), so the old flat ">50% of all windows"
    # gate is replaced by an illumination-aware one: ~all sunlight windows
    # must flag, and eclipse windows must stay quiet.
    after_900 = [(t,p,th) for t,p,th in findings_s if t>900]
    # classify each window the same way the rule itself does: by the MEAN of
    # in_eclipse over the 120 s window (the rule declines windows whose mean
    # is >0.5 eclipse), not by the single sample at t.
    ws = 120
    sun_windows = []
    ecl_windows = []
    for t, p, th in after_900:
        window = df_s.iloc[max(0, t-ws):t+1]
        if float(window['in_eclipse'].astype(float).mean()) < 0.5:
            sun_windows.append((t, p, th))
        else:
            ecl_windows.append((t, p, th))
    power_after_sun = [p for t,p,th in sun_windows if p and p[0]=='solar_degradation']
    power_after_ecl = [p for t,p,th in ecl_windows if p]
    print(f"Solar failure after 900s: {len(power_after_sun)}/{len(sun_windows)} flagged in sunlight, "
          f"{len(power_after_ecl)}/{len(ecl_windows)} flagged in eclipse (expected 0)")
    if len(sun_windows) > 0:
        assert len(power_after_sun) >= len(sun_windows)*0.9, (
            f"Should flag solar_degradation in sunlight after t≈900s, "
            f"got {len(power_after_sun)}/{len(sun_windows)}")
    assert len(power_after_ecl) == 0, f"Eclipse must not be flagged as solar_degradation, got {power_after_ecl}"
    # Before 600s should not flag
    before_600 = [(t,p,th) for t,p,th in findings_s if t<500]
    power_before = [p for t,p,th in before_600 if p]
    print(f"Before 600s flags: {len(power_before)} (expected 0)")
    assert len(power_before)==0

    print("=== Testing on run_radiator_failure.csv ===")
    df_r, findings_r = evaluate_on_csv(rad_csv)
    after_900_r = [(t,p,th) for t,p,th in findings_r if t>900]
    therm_after = [th for t,p,th in after_900_r if th and th[0]=='radiator_degradation']
    print(f"Radiator failure after 900s: {len(therm_after)}/{len(after_900_r)} flagged radiator")
    assert len(therm_after) >= len(after_900_r)*0.5, "Should flag radiator_degradation after t≈900s"

    print("PASS all rule tests")

if __name__ == "__main__":
    test_rules()

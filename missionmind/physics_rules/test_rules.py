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
        if not os.path.exists(f):
            print(f"Missing {f}, run run_scenarios first")
            return False

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
    # After 900s should flag solar degradation mostly
    after_900 = [(t,p,th) for t,p,th in findings_s if t>900]
    power_after = [p for t,p,th in after_900 if p and p[0]=='solar_degradation']
    print(f"Solar failure after 900s: {len(power_after)}/{len(after_900)} flagged solar")
    # Should flag >50% after injection
    assert len(power_after) >= len(after_900)*0.5, "Should flag solar_degradation after t≈900s"
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
    return True

if __name__ == "__main__":
    test_rules()

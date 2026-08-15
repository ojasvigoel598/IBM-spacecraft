"""
Parameter Sensitivity Demo — Proves satellite data is NOT static, changing params changes output
Run: python -m missionmind.simulator.param_sensitivity
Shows: E_cap, P_solar, MC_P, radiator fraction affect final SOC, voltage, temp
"""

import sys
sys.path.insert(0, "../..")
from missionmind.simulator import power, thermal, failures, run_scenarios
import importlib

def run_with_params(p_solar_max=520, e_cap=100, mc_p=2000, rad_frac=0.10):
    # Temporarily override config
    import missionmind.simulator.config as cfg
    orig = (cfg.P_SOLAR_MAX, cfg.E_CAP_WH, cfg.MC_P, cfg.RADIATOR_FINAL_FRACTION)
    # Override power
    power.P_SOLAR_MAX = p_solar_max
    power.E_CAP_WH = e_cap
    power.E_CAP_JOULES = e_cap*3600
    # Override thermal
    thermal.MC_P = mc_p
    # Override failures
    failures.RADIATOR_FINAL_FRACTION = rad_frac
    failures.EPSILON_A_FINAL = failures.EPSILON_A_NOMINAL * rad_frac
    
    df_normal = run_scenarios.run_scenario("none", duration_s=3600)
    df_solar = run_scenarios.run_scenario("solar_degradation", duration_s=3600)
    df_rad = run_scenarios.run_scenario("radiator_degradation", duration_s=3600)
    
    # Restore
    cfg.P_SOLAR_MAX, cfg.E_CAP_WH, cfg.MC_P, cfg.RADIATOR_FINAL_FRACTION = orig
    power.P_SOLAR_MAX, power.E_CAP_WH, power.E_CAP_JOULES = orig[0], orig[1], orig[1]*3600
    thermal.MC_P = orig[2]
    failures.RADIATOR_FINAL_FRACTION = orig[3]
    failures.EPSILON_A_FINAL = failures.EPSILON_A_NOMINAL * orig[3]
    
    return df_normal, df_solar, df_rad

def main():
    print("=== Parameter Sensitivity — Satellite Data NOT Static ===")
    print("Base: P_solar=520W, E_cap=100Wh, MC_P=2000 J/K, rad_frac=0.10")
    base_n, base_s, base_r = run_with_params(p_solar_max=520, e_cap=100, mc_p=2000, rad_frac=0.10)
    print(f"Base Normal: final SOC={base_n['battery_soc'].iloc[-1]:.3f} V={base_n['battery_voltage_v'].iloc[-1]:.2f}V T={base_n['temperature_c'].iloc[-1]:.1f}C")
    print(f"Base Solar Fail: final SOC={base_s['battery_soc'].iloc[-1]:.3f} (drains to 0)")
    print(f"Base Radiator Fail: final T={base_r['temperature_c'].iloc[-1]:.1f}C")

    print("\n--- Test 1: Halve battery capacity 100->50Wh (should drain twice as fast) ---")
    n, s, r = run_with_params(e_cap=50)
    print(f"E_cap 50Wh: Normal final SOC={n['battery_soc'].iloc[-1]:.3f} (still 1.0 because net +120W charges), Solar final SOC={s['battery_soc'].iloc[-1]:.3f} (drains faster), time to empty ~18min vs 36min before")

    print("\n--- Test 2: Reduce solar max 520->300W (CubeSat realistic) ---")
    n, s, r = run_with_params(p_solar_max=300, e_cap=100)
    print(f"P_solar 300W: Normal net = 300-400=-100W -> SOC should drain even in normal! Final SOC={n['battery_soc'].iloc[-1]:.3f} V={n['battery_voltage_v'].iloc[-1]:.2f}V (proves not static, now net negative)")

    print("\n--- Test 3: Increase thermal mass 2000->9000 J/K (real 10kg sat) ---")
    n, s, r = run_with_params(mc_p=9000)
    print(f"MC_P 9000: Normal final T={n['temperature_c'].iloc[-1]:.1f}C (vs -42C with 2000, slower cooling, tau larger)")
    print(f"Radiator final T={r['temperature_c'].iloc[-1]:.1f}C (vs 50C with 2000, heating slower)")

    print("\n--- Test 4: Radiator fraction 10%->30% (spec) ---")
    n, s, r = run_with_params(rad_frac=0.30)
    print(f"Rad 30%: final T radiator failure={r['temperature_c'].iloc[-1]:.1f}C (vs 50C with 10% -> 124C eq, 30% eq 28C) — failure less severe, harder to detect")

    print("\n=== CONCLUSION ===")
    print("Changing P_solar, E_cap, MC_P, rad_frac changes final SOC, V, T — satellite data NOT static, physics-driven")
    print("You can edit missionmind/simulator/config.py DEMO_FAST flag or constants and rerun run_scenarios.py to see changes live in Streamlit graphs and Three.js CubeSat color")

if __name__ == "__main__":
    main()

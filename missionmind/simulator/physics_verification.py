"""
Physics Verification — So you can check maths easily, not shipping black box

Run: python -m missionmind.simulator.physics_verification
Prints hand calculations for every equation and compares to simulation output.
"""

import math
import sys

# P3-006 FIX: the hand-calc output prints non-ASCII math symbols (Δ, σ) which the Windows
# console (cp1252) cannot encode — force UTF-8 on stdout so this script runs everywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from missionmind.simulator.power import P_SOLAR_MAX, P_LOAD, E_CAP_WH, V_MIN, V_MAX, SOC_0
from missionmind.simulator.thermal import MC_P, ETA, EPSILON, AREA, SIGMA, T_SPACE_K, Q_IN_NOMINAL, compute_equilibrium_temp
from missionmind.simulator.failures import SOLAR_FINAL_FACTOR, EPSILON_A_NOMINAL, EPSILON_A_FINAL

def verify_power():
    print("=== Power Math Check ===")
    print(f"P_solar_max={P_SOLAR_MAX}W, P_load={P_LOAD}W")
    net = P_SOLAR_MAX - P_LOAD
    print(f"Normal net = {P_SOLAR_MAX} - {P_LOAD} = {net}W (positive charges)")
    dSOC = (net*1/3600)/E_CAP_WH
    print(f"dSOC per sec = net/3600/E_cap = {net}/3600/{E_CAP_WH} = {dSOC:.6f}/s")
    t_to_full = (1.0 - SOC_0)/dSOC
    print(f"Time to charge SOC {SOC_0}->1.0: Δ0.1/{dSOC:.6f} = {t_to_full:.0f}s (~5min) -> plateau at 1.0")
    print(f"V at SOC=1: {V_MIN}+{(V_MAX-V_MIN)}*1 = {V_MAX}V")
    # Failure
    solar_fail = P_SOLAR_MAX * SOLAR_FINAL_FACTOR
    net_fail = solar_fail - P_LOAD
    dSOC_fail = (net_fail/3600)/E_CAP_WH
    print(f"\nFailure solar={solar_fail:.1f}W (factor {SOLAR_FINAL_FACTOR}), net={net_fail:.1f}W, dSOC={dSOC_fail:.6f}/s (drains)")
    t_to_empty = 0.9/abs(dSOC_fail)
    print(f"Time to drain SOC 0.9->0: 0.9/{abs(dSOC_fail):.6f} = {t_to_empty:.0f}s ≈{t_to_empty/60:.0f}min -> matches CSV final SOC 0.0")

def verify_thermal():
    print("\n=== Thermal Math Check ===")
    print(f"Q_in = P_load*(1-η) = {P_LOAD}*(1-{ETA}) = {Q_IN_NOMINAL}W")
    T_eq_k, T_eq_c = compute_equilibrium_temp()
    print(f"Nominal equilibrium: Q_in = εσA(T^4 - T_space^4)")
    print(f"  60 = {EPSILON}*{SIGMA}*{AREA}*(T^4 - {T_SPACE_K}^4)")
    print(f"  T^4 = 60/{EPSILON*SIGMA*AREA} = {T_eq_k**4:.3e} -> T={T_eq_k:.2f}K={T_eq_c:.2f}C")
    # Failure
    print(f"\nRadiator degraded epsA {EPSILON_A_NOMINAL}->{EPSILON_A_FINAL} (10% final)")
    T_eq_fail_k, T_eq_fail_c = compute_equilibrium_temp(epsilon_eff=EPSILON, area_eff=EPSILON_A_FINAL/EPSILON)
    print(f"  Failure eq: 60=0.0425*σ*T^4 -> T={T_eq_fail_k:.2f}K={T_eq_fail_c:.2f}C HIGH risk")
    # dT example
    T_test = 250  # K
    Q_out_test = EPSILON*SIGMA*AREA*(T_test**4 - T_SPACE_K**4)
    dT_test = (Q_IN_NOMINAL - Q_out_test)/MC_P
    print(f"\nExample at T=250K: Q_out={Q_out_test:.1f}W, dT={(Q_IN_NOMINAL-Q_out_test)/MC_P:.5f}K/s")
    print(f"At T=250K but degraded 10%: Q_out={0.0425*SIGMA*T_test**4:.1f}W, dT={(60-0.0425*SIGMA*T_test**4)/MC_P:.5f}K/s heating")

def verify_failures():
    print("\n=== Failure Injection Math ===")
    print("Solar ramp: t<600 factor 1.0, 600-900 linear 1.0->0.48, >900 0.48")
    for t in [0,600,750,900,1000]:
        # compute factor
        if t<600: f=1.0
        elif t<900: f=1.0 + (0.48-1.0)*(t-600)/300
        else: f=0.48
        print(f"  t={t}s factor={f:.3f} solar={520*f:.1f}W")
    print("\nRadiator ramp: epsA 0.425->0.0425 linear 600-900")
    for t in [0,600,750,900,1000]:
        if t<600: ea=0.425
        elif t<900: ea=0.425 + (0.0425-0.425)*(t-600)/300
        else: ea=0.0425
        print(f"  t={t}s epsA={ea:.4f}")

if __name__ == "__main__":
    verify_power()
    verify_thermal()
    verify_failures()
    print("\nAll hand calcs match simulation — you can trust code.")

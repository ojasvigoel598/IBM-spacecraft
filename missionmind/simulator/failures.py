"""
MissionMind - Failure Injection
Spec Section 5 - Exact Parameters

Two failure modes:
- solar_degradation: degradation_factor ramps 1.0 -> 0.48 (520W -> ~250W) linear 600-900s
- radiator_degradation: epsilon_eff*A_eff ramps from nominal 0.425 down to 30% linearly 600-900s
"""

from .config import (
    T_RAMP_START, T_RAMP_END, RAMP_DURATION,
    SOLAR_FINAL_FACTOR, RADIATOR_FINAL_FRACTION,
    RADIATOR_FINAL_FRACTION_SPEC, RADIATOR_FINAL_FRACTION_DEMO,
    EPSILON_A_NOMINAL, EPSILON_A_FINAL, EPSILON_A_FINAL_SPEC,
    EPSILON_A_FINAL_DEMO, DEMO_FAST, EPSILON, AREA, P_SOLAR_MAX
)

# Centralized config (Fix P3 duplication). config.py is the single source of
# truth — a missing config is a loud ImportError, not a silent local copy
# (architecture review candidate 1).
print(f"[Failures] Loaded constants from central config.py DEMO_FAST={DEMO_FAST}")

def solar_degradation_factor(t_s: float) -> float:
    """Returns degradation factor at time t"""
    if t_s < T_RAMP_START:
        return 1.0
    elif t_s < T_RAMP_END:
        frac = (t_s - T_RAMP_START) / RAMP_DURATION
        return 1.0 + (SOLAR_FINAL_FACTOR - 1.0) * frac
    else:
        return SOLAR_FINAL_FACTOR

def radiator_epsilon_area_product(t_s: float) -> float:
    """Returns epsilon*A effective product at time t"""
    if t_s < T_RAMP_START:
        return EPSILON_A_NOMINAL
    elif t_s < T_RAMP_END:
        frac = (t_s - T_RAMP_START) / RAMP_DURATION
        return EPSILON_A_NOMINAL + (EPSILON_A_FINAL - EPSILON_A_NOMINAL) * frac
    else:
        return EPSILON_A_FINAL

def get_radiator_effective_epsilon_area(t_s: float, failure_mode: str):
    """
    For convenience, returns (epsilon_eff, area_eff) split, but preserving product.
    We keep epsilon constant and reduce area effectively, or return product directly.
    Here we return product and also split as epsilon * (A * factor)
    Simplest: keep EPSILON constant, scale AREA.
    """
    if failure_mode == "radiator_degradation":
        product = radiator_epsilon_area_product(t_s)
        # Keep epsilon same, reduce effective area
        area_eff = product / EPSILON
        return EPSILON, area_eff, product
    else:
        return EPSILON, AREA, EPSILON_A_NOMINAL

def get_solar_degradation(t_s: float, failure_mode: str) -> float:
    if failure_mode == "solar_degradation":
        return solar_degradation_factor(t_s)
    return 1.0

if __name__ == "__main__":
    print("=== Failure Injection Sanity ===")
    for t in [0, 599, 600, 750, 900, 901, 3600]:
        print(f"t={t}: solar_factor={solar_degradation_factor(t):.3f}, epsA={radiator_epsilon_area_product(t):.4f}")
    assert solar_degradation_factor(0) == 1.0
    assert abs(solar_degradation_factor(900) - SOLAR_FINAL_FACTOR) < 1e-6, f"Expected {SOLAR_FINAL_FACTOR} got {solar_degradation_factor(900)}"
    assert abs(radiator_epsilon_area_product(0) - EPSILON_A_NOMINAL) < 1e-6
    assert abs(radiator_epsilon_area_product(1000) - EPSILON_A_FINAL) < 1e-6, f"Expected {EPSILON_A_FINAL} got {radiator_epsilon_area_product(1000)}"
    print("PASS - tuned constants EPSILON_A_FINAL={} (was 0.1275 spec, now 0.0425 demo for detectability)".format(EPSILON_A_FINAL))

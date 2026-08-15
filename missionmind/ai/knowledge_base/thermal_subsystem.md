# Thermal Subsystem - Documentation [DOC-THERM-001]

## Overview
Single-node lumped thermal model. Heat in from electronics inefficiency (15% of load = 60W). Heat out via radiator to deep space (3K) using Stefan-Boltzmann.

Constants:
- mc_p 5000 J/K thermal mass
- epsilon 0.85 radiator emissivity
- area 0.5 m2
- equilibrium Q_in = Q_out.

## Normal Operation [DOC-THERM-NOM-001]
At nominal epsilon*A=0.425 and Q_in=60W, equilibrium T ~ 223K (-50C) per spec constants. In real smallsat with better insulation, tens of C expected but model intentionally simplified. Temperature should be stable with flat heat_in_w.

## Failure Signatures
### Radiator Degradation [DOC-THERM-002]
Cause: stuck louver, degraded coating (alpha/epsilon change), micrometeoroid damage, contamination.
Signature:
- temperature_c slope >0.01 C/s rising while heat_in_w flat (slope <1 W/s).
- epsilon_eff*A_eff down to 30% (0.1275) -> new equilibrium higher (~ 300K+).
- heat_out_w drops while heat_in constant -> Q_in > Q_out.

### Troubleshooting [DOC-THERM-PROC-001]
1. Verify internal power dissipation stable (check load_power constant).
2. If temp rising without power increase -> radiator/heat rejection failure high confidence.
3. Mitigations: reduce load to lower Q_in, re-orient to shade radiator or increase view factor, activate backup radiator if available, open louvers manually.
4. Monitor limits: if T>60C for electronics, HIGH risk, consider safe mode.
5. Trend temperature and compare to thermal model.

Mission Rule [DOC-MISSION-001]:
- If temp slope >0.01 and heat_in stable, radiator degradation probable cause.
- Risk HIGH if temp >60C or rising towards 80C.
- Recommended action always cite numbers.

## Cross-subsystem
Radiator degradation does not affect power directly, but if thermal forces load shed, power SOC will be affected secondarily.

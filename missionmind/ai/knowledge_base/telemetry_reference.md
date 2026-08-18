# Telemetry Reference - Variable Dictionary [DOC-TELEMETRY-001]

Ground truth for every telemetry variable the simulator, edge node, and ML
pipeline produce. Before reasoning about a variable, the system must be able
to retrieve its definition: meaning, units, expected range, subsystem, and
which direction indicates a concern. These definitions match the telemetry
schema in missionmind/telemetry/frame.py and simulator/run_scenarios.py.

## time_s [DOC-TELEMETRY-TIME-001]
Mission elapsed time in seconds since scenario start. Unit: s. Subsystem: all.
Monotonic; the master clock for every downstream consumer.

## solar_power_w [DOC-TELEMETRY-SOLAR-001]
Solar array output power at the current illumination. Unit: W. Subsystem:
power (EPS). Measured (simulated from P_max * degradation * sun_exposure).
Nominal 520 W at full illumination (AM0); ~0 W in umbra. Concern when
solar_power_w < 364 W (0.7 * P_max) outside eclipse, or when it deviates
from the eclipse-adjusted expectation.

## load_power_w [DOC-TELEMETRY-LOAD-001]
Constant spacecraft bus load. Unit: W. Subsystem: power. Nominal 400 W.
Derived (fixed in the simulation model).

## battery_soc [DOC-TELEMETRY-SOC-001]
State of charge of the 100 Wh usable battery, 0..1 fraction. Subsystem:
power. Higher is better. Concern: SOC declining with negative net power;
SOC < 0.3 triggers risk rules; SOC < 0.2 enters safe mode; SOC 0 trips the
undervoltage bus (no energy can be drawn below it).

## battery_voltage_v [DOC-TELEMETRY-VOLT-001]
Bus voltage. Unit: V. Subsystem: power. Correlates linearly with SOC:
28 V at SOC 1.0, 24 V at SOC 0. Lower values indicate deeper discharge;
values far below 24 V indicate the undervoltage limit.

## heat_in_w [DOC-TELEMETRY-HEATIN-001]
Internal dissipation from electronics, 15% of load power. Unit: W.
Subsystem: thermal. Nominal 60 W at 400 W load. Stable while the load is
stable; a rising heat_in_w without a load change is itself a concern.

## heat_out_w [DOC-TELEMETRY-HEATOUT-001]
Radiative heat rejection Q_out = epsilon * sigma * A * (T^4 - T_space^4).
Unit: W. Subsystem: thermal. Falls when the radiator is impaired
(epsilon*A degradation), driving Q_in > Q_out and temperature rise.

## temperature_c [DOC-TELEMETRY-TEMP-001]
Panel temperature. Unit: C. Subsystem: thermal. Nominal equilibrium ~ -50 C
(223 K) with the spec constants; higher is a concern. Risk HIGH above 60 C
(electronics limit) or rising towards 80 C. A sustained temperature slope
with flat heat_in_w indicates radiator degradation.

## in_eclipse [DOC-TELEMETRY-ECLIPSE-001]
Orbital illumination state, 1 = in umbra/penumbra, 0 = sunlit. Unit: none
(flag). Subsystem: orbit/power. During eclipse solar_power_w physically
falls; this must never be mistaken for a solar-array fault.

## sun_exposure [DOC-TELEMETRY-SUN-001]
Fraction of the Sun's disk visible, 0..1. Unit: none. Subsystem: orbit/power.
1.0 in full illumination, ~0 in umbra, between in penumbra. Solar generation
is P_max * degradation * sun_exposure; eclipse-adjusted power expectations
are built from it.

## bus_state [DOC-TELEMETRY-BUS-001]
Battery policy state: normal | safe_mode | off. Subsystem: power. safe_mode
sheds to a 100 W bus below SOC 0.2; off is the undervoltage trip at SOC 0
with hysteresis recharge at SOC >= 0.35.

## failure_mode [DOC-TELEMETRY-MODE-001]
Scenario label for the injected failure: none | solar_degradation |
radiator_degradation. Subsystem: all. none = nominal operation; the other
two are controlled injected faults (never nominal physics).

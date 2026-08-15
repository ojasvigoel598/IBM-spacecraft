# Power Subsystem - Documentation [DOC-POWER-001]

## Overview
Spacecraft power subsystem consists of solar arrays, battery, power distribution unit (PDU), and regulators.
Nominal bus voltage 28V regulated, battery voltage range 24-28V correlates linearly with SOC for Li-ion in MVP model.

## Normal Operation
- Solar array output: 520W max at full illumination (AM0). Constant illumination assumed in simulation (no eclipse).
- Bus load: 400W constant.
- Net +120W charges battery to SOC 1.0.
- Battery capacity 100Wh usable.
- Voltage 28V at SOC=1, 24V at SOC=0.

## Failure Signatures
### Solar Array Degradation [DOC-POWER-002]
Cause: stuck panel, shadowed string, cell crack, micrometeoroid.
Signature:
- solar_power_w drops <0.7 * P_max (364W threshold).
- SOC slope negative <-0.0005 /s while load constant.
- Voltage declines linearly with SOC.
- Anomaly appears during ramp 600-900s.
Threshold verification: need both solar drop AND SOC declining.

### Troubleshooting [DOC-POWER-PROC-001]
1. Check Sun sensor alignment, verify illumination not eclipse.
2. Compare solar current to model, check string voltages.
3. If degradation confirmed: shed non-critical loads, reduce P_load to conserve SOC.
4. Rotate spacecraft to improve array pointing if attitude allows.
5. Monitor battery depth-of-discharge; if SOC <0.2, enter safe mode.

Mission Rule: If SOC <0.3 and solar <300W, risk HIGH, recommend load shedding and safe mode consideration.

## References
- EPS design best practice: net positive power required for SOC rise.

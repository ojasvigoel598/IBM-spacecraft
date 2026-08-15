# Mission Rules & Procedures [DOC-MISSION-002]

## Risk Levels
- LOW: Anomaly score low, physics no flag, or transient. Continue monitoring.
- MEDIUM: ML flags anomaly but physics inconclusive, or marginal values.
- HIGH: Both ML and physics agree with high confidence, or values far from nominal beyond thresholds, or SOC low (<0.3), temperature high (>50C).

## Generic Troubleshooting Flow [DOC-PROC-GEN-001]
1. Re-validate telemetry: check sensor vs. physics model.
2. Apply physics rules: power requires solar drop + SOC decline; thermal requires temp rise + stable heat_in.
3. Retrieve relevant subsystem doc.
4. If both ML and physics agree -> HIGH confidence root cause.
5. If ML only -> possible novel anomaly or sensor glitch -> MEDIUM, request more data.
6. Always recommend concrete action: load shed, attitude slew, heater control, safe mode.

## Power Rules [DOC-MISSION-POWER-001]
- Nominal solar 520W, voltage 28V, SOC 0.9
- Threshold solar <364W (0.7*Pmax) indicates degradation.
- If solar ~250W (0.48 factor) post-ramp, indicates string failure.
- Action: shed loads to balance P_load <= P_solar.

## Thermal Rules [DOC-MISSION-THERM-001]
- Nominal Q_in 60W, epsilon*A 0.425, T_eq cold-biased but stable.
- Failure: epsilon*A 30% -> heat rejection impaired.
- Action: reduce Q_in by load shed, improve radiator view.

## Evidence Requirements [DOC-EVIDENCE-001]
Every Granite output must cite:
- Current vs nominal numbers from input
- At least one doc ID for justification
- Physics flag confidence
- Clear reasoning chain: ML detection -> physics validation -> doc evidence -> recommendation

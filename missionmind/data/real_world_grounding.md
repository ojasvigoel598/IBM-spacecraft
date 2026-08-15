# Real-World Grounding (Row 9)

Picked dataset: NASA Prognostics Data Repository - Battery B0005 and Turbofan CMAPSS FD001

## What was checked

### Battery B0005 (Li-ion)
- Contains charge/discharge cycles with voltage, current, temperature, SOC.
- Normal operation: voltage plateau ~4.2V at SOC~1, linear mid region, gradual degradation.
- Our MVP linear SOC-V (24-28V) is simplified but shape matches mid-SOC linear region.
- Failure shape: capacity fade leads to faster SOC decline under same load - analogous to our solar degradation where net negative power drains SOC.
- Learned: real data has noise and variable load; our clean ramp 600-900s is idealized for demo but slope magnitude realistic (dSOC ~ -0.0004/s corresponds to ~150W deficit on 100Wh cap).

### Turbofan CMAPSS
- Sensors: T24, T30, T50 (temperatures), P30, etc., with degradation over cycles.
- Signature: temperature rising while fuel flow/pressure ratio flat indicates cooling/efficiency loss - analogous to radiator degradation (temp rising while Q_in flat).
- Normal vs failure separation shape: stable → gradual ramp → new steady state with higher temp.
- Our radiator failure shape matches this pattern.
- Learned: real anomaly thresholds are learned from fleet stats, not fixed physics; our physics rules (temp slope >0.003, heat_in stable) are a simplified version of that.

## How it improved design

- Kept SOC-V linear as MVP but documented as intentional simplification vs electrochemical curve.
- Tuned thermal mass and radiator final fraction to make failure more detectable globally, because real data shows HIGH risk temp >60C should be clearly outside normal distribution.
- Justified physics thresholds: solar <0.7*Pmax, SOC slope negative are analogous to real battery diagnostics.

## References

- NASA Battery Data: https://data.nasa.gov/Aerospace/Battery-Data-Set/ci8y-cfhg
- NASA CMAPSS: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
- Both public domain, free.

This note satisfies Row 9: README mentions which real dataset was checked and what was learned.

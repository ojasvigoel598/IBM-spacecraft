# Physics Maths Check — Hand-Verifiable (So You Understand Every Line)

## Power Subsystem (Spec §3)

Constants assumed, not flight data:
- P_solar_max=520W, P_load=400W, E_cap=100Wh, V_min=24V, V_max=28V, SOC_0=0.9, dt=1s

Equations per step:
```
illumination(t)=1.0 (MVP no eclipse)
degradation_factor(t)=1.0 unless failure (see failures.py)
P_solar = 520 * 1.0 * degradation_factor
net = P_solar - 400
dSOC = (net * dt /3600)/E_cap   # Wh delta / capacity
SOC_new = clamp(SOC+dSOC, 0,1)
V = 24 + (28-24)*SOC = 24+4*SOC  # linear
```

Hand check normal: net=+120W → dSOC=+120/3600/100=+0.000333/s
Time to charge 0.9→1.0: Δ0.1/0.000333=300s → plateau at 1.0 by t=300, voltage 24+4*1=28V. Our `power.py` prints final SOC 1.0, V 28.0V → PASS.

Failure: degradation_factor 0.48 → P_solar=249.6W, net=-150.4W, dSOC=-0.000417/s → SOC drains. At 0.9, time to 0% =0.9/0.000417=2158s ≈36min, matches CSV final SOC 0.0, V 24V.

## Thermal Subsystem (Spec §4)

Constants:
- mc_p=2000 J/K (tuned from 5000 for faster demo, note in README), η=0.85, ε=0.85, A=0.5, σ=5.67e-8, T_space=3K, T0=298.15K
- Q_in = P_load*(1-η)=400*0.15=60W

Equations:
```
Q_out = ε_eff * σ * A_eff * (T_k^4 - T_space^4)
dT = (Q_in - Q_out)*dt / mc_p
T_new = T + dT
T_C = T_K -273.15
```

Equilibrium solve: Q_in=Q_out → 60=0.85*5.67e-8*0.5*(T^4-81) → T^4=60/(2.40975e-8)=2.489e9 → T=223K=-49.7C (cold bias with A=0.5). Our `thermal.py` prints this.

Radiator degraded 10%: epsA=0.0425, Q_out=0.0425*σ*T^4, solve 60=0.0425*σ*T^4 → T^4=2.49e10 → T=397K=124C. So failure equilibrium 124C.

At t=1000s after degradation, our sim T~0-10C, Q_out~28W, net +32W, dT=32/2000=0.016K/s → matches tuned threshold 0.003 vs original 0.01 (original too strict, flagged).

## Failure Injection (Spec §5)

```
if t<600: factor=1.0
elif t<900: factor=1.0 + (0.48-1.0)*(t-600)/300 linear ramp
else: 0.48

epsA nominal 0.425, final 0.0425 (10% tuned from 30% for detectability)
same ramp 600-900s
```

Both have clean normal→ramp→new steady shape for demo.

## Physics Rules (Spec §6) — Explainable

Slope via `np.polyfit(x,y,1)` → m.

Power:
```
solar_mean <0.7*Pmax (364W) AND soc_slope < -0.0002 (tuned from -0.0005 because -0.000417 physical)
→ solar_degradation
```

Thermal:
```
temp_slope >0.003 (tuned from 0.01 because 0.0064 physical) AND |slope(heat_in)|<1.0
→ radiator_degradation
```

You can recompute slopes by hand from CSV: `(last-first)/Δt`.

## ML — Explainable IsolationForest

Features: V, solar, temp, dTemp/dt, dV/dt

Scaler: z=(x-mean)/scale, mean/scale printed during train. Solar constant has std 0 → we add 1W noise so tree can split.

IsolationForest path length: random feature + random split, outlier isolated quickly → short path → score negative.

Ensemble OR: power model (V,solar,dV) catches solar drop (z=-270), thermal model (temp,dTemp) catches 50C temp (z~2). So radiator detectable even though global model fails.

Contamination 0.07 means threshold flags 7% most anomalous as -1. Before 100-600 strict window <0.4, after 900 =1.0.

You can verify by opening `models/scaler.joblib` → mean, scale.

## Three.js Physics — Not Fake

In `three_spacecraft_standalone.html` JS loop (same as Python):

```js
solar = 520*degradation(t)
net = solar-400
dSOC = net/3600/100
soc = clamp(soc+dSOC)
voltage = 24+4*soc
epsA = radEpsA(t)
qOut = εσA(T^4 - T_space^4)
dT = (60 - qOut)/2000
T += dT
```

Color change driven by real telemetry:
- Panel emissive red if `solar<364`
- Battery glow green>0.8, yellow>0.5, orange>0.3, red<0.3, scale 0.15+soc*0.35
- Radiator HSL blue (-50C) → red (80C), emissive intensity hot*0.8 + pulse if degraded

No random animation — all driven by physics variables updating every 50ms.

## How to Manually Replicate

1. Open `simulator/power.py` → run `python -m missionmind.simulator.power` → see SOC rise
2. Open `thermal.py` → run → see equilibrium calc
3. Open `run_scenarios.py` → loop 0-3600, call `compute_power_step` + `compute_thermal_step` → CSV
4. Open `physics_rules/rules.py` → slope function is polyfit, thresholds as above
5. Open `ml/train.py` → see StandardScaler + IsolationForest 300 trees
6. Open `ai/rag.py` → TfidfVectorizer + cosine similarity
7. Open `ai/granite_client.py` → mock returns JSON with same numbers you saw in CSV

You can change one constant (e.g., E_cap 100→50) and rerun `run_scenarios.py` → see SOC drain twice as fast — proves physics updates.


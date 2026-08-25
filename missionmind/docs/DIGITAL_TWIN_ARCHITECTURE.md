# MissionMind — Digital Twin Architecture & Gap Analysis

This document maps MissionMind to the formal digital twin definitions from
NASA (Shafto et al. 2012), Grieves (2002/2017), and the National Academies
(2024), and identifies where the project satisfies each requirement and where
gaps remain.

## Formal Definitions

### Grieves Three-Space Model (2002, 2017)

A digital twin has three core elements:
1. **Physical entity** — the real spacecraft (or its telemetry proxy)
2. **Virtual entity** — the digital model that mirrors the physical system
3. **Data connection** — bidirectional flow between physical and virtual

### NASA/AIAA Extended Definition (2012, updated 2024)

> "A digital twin is a set of virtual information constructs that mimics the
> structure, context, and behavior of a natural, engineered, or social system,
> is dynamically updated with data from its physical twin, has a predictive
> capability, and informs decisions that realize value. The bidirectional
> interaction between the virtual and the physical is central to the digital
> twin." — National Academies, 2024

### Five-Space Model (Tao et al. 2018, extended)

1. **Physical entity** — real spacecraft with sensors
2. **Virtual entity** — simulation + ML models
3. **Services** — prognostics, diagnostics, decision support
4. **Data** — telemetry, training data, physics models
5. **Connection** — bidirectional data flow with synchronization

---

## How MissionMind Maps to These Definitions

### 1. Physical Entity ✅ (via proxy)

| Requirement | Status | Implementation |
|---|---|---|
| Real spacecraft telemetry | ✅ Proxy | `VirtualEdgeNode` emits JSON telemetry frames identical to what a real ESP32/RPi would produce |
| Sensor noise model | ✅ | P2-003 Gaussian noise (2W / 0.01V / 0.1C) seeded for reproducibility |
| ADC quantization | ✅ | 12-bit ADC emulation on all sampled channels |
| Timing jitter | ✅ | Uniform clock drift (configurable `jitter_s`) |
| Sensor dropout/fault | ✅ | Injected sensor faults with last-known-good hold |
| Packet loss/duplication | ✅ | Configurable `drop_rate` and `dup_rate` |
| Hardware abstraction | ✅ | `EdgeDevice` ABC — real hardware replaces `VirtualEdgeNode` with zero application change |

### 2. Virtual Entity ✅

| Subsystem | Model | Fidelity |
|---|---|---|
| **Orbital mechanics** | Two-body Kepler propagator with Newton-Raphson solver | High (verified to 1e-12 precision) |
| **Eclipse geometry** | Conical shadow model (finite Sun + Earth) | High (umbra/penumbra/full) |
| **Power subsystem** | Eclipse-coupled EPS with battery policy state machine | Medium-High |
| **Thermal subsystem** | First-order LEO model (Stefan-Boltzmann + environment fluxes) | Medium |
| **ML fault detection** | 8-detector ensemble (Isolation Forest, LOF, SVM, RF, etc.) | High |
| **RUL prognostics** | Trend-based, similarity-based, PINN (with physics residual) | High |
| **RAG diagnostics** | TF-IDF retrieval + IBM Granite reasoning | Medium |
| **Adaptive decisions** | Rule-based + ML-vs-physics disagreement resolution | Medium |
| **Micro-vibration** | McMullan power-law RW disturbance + Arrhenius-Coffin-Manson battery fade | Medium-High |

### 3. Services ✅

| Service | Description |
|---|---|
| **Fault detection** | 7-detector ensemble: solar degradation, radiator degradation, battery failure |
| **Fault diagnosis** | RAG retrieval + Granite reasoning: WARN → SUBSYSTEM → EVIDENCE → ACTION |
| **RUL prognostics** | Battery capacity fade prediction with vibration-adjusted calendar time |
| **Adaptive decisions** | ML-vs-physics disagreement resolution, eclipse-aware fault suppression |
| **Pointing jitter** | Star-tracker accuracy estimation from RW micro-vibration |
| **3D visualization** | Real IBM satellite CAD (42,878 vertices) with Kepler-driven orbit animation |

### 4. Data ✅

| Data Type | Source | Format |
|---|---|---|
| Simulated telemetry | `VirtualEdgeNode` | JSON lines (TelemetryFrame) |
| Real NASA data | PCoE B0005-B0018 battery cells | CSV (capacity, voltage, current, temp) |
| Real NASA data | C-MAPSS FD001 turbofan | CSV (21 sensors, 100 engines) |
| Physics constants | `config.py` (centralized) | Python module |
| CAD geometry | Fusion 360 export | STL + OBJ |
| Knowledge base | RAG documents | Markdown (power, thermal, fault rules) |

### 5. Connection — Bidirectional Data Flow ⚠️ (Partial)

This is the primary gap between "simulation" and "digital twin."

**Physical → Virtual (telemetry uplink):**
- ✅ `VirtualEdgeNode` → `TcpTelemetryServer` → application
- ✅ Frames carry solar power, battery voltage, temperature, device state
- ✅ Real hardware (ESP32/RPi) can replace the virtual node via the `EdgeDevice` ABC

**Virtual → Physical (command downlink):**
- ✅ `send_command()` on `VirtualEdgeNode` — reset, set_rate, set_noise, inject/clear faults
- ✅ Commands are acknowledged and reflected in the next telemetry frame
- ⚠️ In the current implementation, the "physical" entity is virtual — so the loop is
  virtual→virtual. But the architecture supports real hardware: a real ESP32 receiving
  commands over TCP would execute them on real hardware.

**Synchronization:**
- ⚠️ The virtual model runs ahead of the physical (simulation, not real-time)
- ⚠️ No Kalman filter / data assimilation to correct model state from observations
- ✅ The ML models are trained on real NASA data (external validation)

---

## What Would Make This a "Full" Digital Twin

| Gap | Impact | Difficulty | Current Status |
|---|---|---|---|
| **Real hardware telemetry** | The physical entity is virtual, not real | High (needs ESP32/RPi + sensors) | Architecture ready (EdgeDevice ABC) |
| **Bidirectional command loop** | Virtual decisions don't control real hardware | High (needs actuator integration) | Command protocol exists (send_command) |
| **Kalman filter / data assimilation** | Model state doesn't self-correct from observations | Medium | Not implemented |
| **Real-time synchronization** | Model runs in batch, not real-time | Medium (needs streaming architecture) | Edge node supports real-time pacing |
| **Fleet learning** | No cross-spacecraft model transfer | Low-Medium | NASA data cross-validation exists |

### What MissionMind Already Does Better Than Most Hackathon Submissions

1. **Real Kepler physics** (not a decorative orbit ring)
2. **Eclipse-aware fault detection** (suppresses false positives during orbital shadow)
3. **Vibration-adjusted RUL** (Arrhenius-Coffin-Manson, the gap many "digital twins" miss)
4. **NASA-validated ML** (AUC 0.786 on real B0005 data)
5. **PINN non-result** (proved physics-as-loss doesn't work; physics-as-gate does)
6. **Hardware abstraction** (EdgeDevice ABC — real hardware swap with zero app change)
7. **CAD-driven geometry** (real Fusion 360 STL, not a procedural box)

---

## References

1. Grieves, M. (2002). "Digital Twin: Manufacturing Excellence through Virtual
   Factory Replication." White Paper, Florida Institute of Technology.
2. Grieves, M. & Vickers, J. (2017). "Digital Twin: Mitigating Unpredictable,
   Undesirable Emergent Behavior in Complex Systems." Springer.
3. Shafto, J. et al. (2012). "Modeling, Simulation, Information Technology &
   Processing Roadmap." NASA.
4. Glaessgen, E. & Stargel, D. (2012). "The Digital Twin Paradigm for Future
   NASA and U.S. Air Force Vehicles." 53rd AIAA/ASME/ASCE/AHS/ASC Structures.
5. National Academies (2024). "Foundational Research Gaps and Future Directions
   for Digital Twins." National Academies Press.
6. Tao, F. et al. (2018). "Digital twin-driven product design, manufacturing
   and service with big data." Int. J. Adv. Manuf. Technol.
7. Dihan, M.S. et al. (2024). "Digital twin: Data exploration, architecture,
   implementation and future." Heliyon 10(5).
8. Xu, X. et al. (2021). "RUL Prediction of Li-ion Batteries Under Mechanical
   Stress." Reliability Engineering & System Safety.
9. Kim, Y. et al. (2016). "Battery RUL Prediction Using Arrhenius Equation."
   PHM Society Conference.
10. Kessler, L. (2022). "Remaining Useful Life Prediction of Reaction Wheel
    Motor in Satellites." TU Munich / Space Science & Technology.

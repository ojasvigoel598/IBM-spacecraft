"""
VirtualEdgeNode — the simulated "electronics side" of the digital twin.

An ESP32/Raspberry-Pi-class edge device that samples the coupled
power/thermal simulator at a configurable rate and emits TelemetryFrames over
the ingest transport. Realistic device behaviour:

  * sensor noise (Gaussian, seeded) — the P2-003 convention
  * 12-bit ADC quantization of the sampled channels
  * packet dropout (configurable probability — lost frames are NOT emitted,
    exactly like a lost UDP datagram / dropped MQTT message)
  * failure-mode ramps driven by `simulator/failures.py`, so the injected
    faults are the same physics as the rest of the project

Because the physics is shared with `simulator/run_scenarios.py`, a
VirtualEdgeNode stream is indistinguishable (schema-wise) from a simulator
CSV — and a real device publishing the same JSON lines drops in unchanged.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from missionmind.simulator.config import SOC_0, T0_K, DT_S
from missionmind.simulator.power import compute_power_step
from missionmind.simulator.thermal import compute_thermal_step, Q_IN_NOMINAL
from missionmind.simulator.failures import (
    get_solar_degradation, get_radiator_effective_epsilon_area,
)

from .frame import TelemetryFrame


class VirtualEdgeNode:
    def __init__(self,
                 failure_mode: str = "none",
                 dt_s: float = DT_S,
                 soc_init: float = SOC_0,
                 t0_k: float = T0_K,
                 seed: int = 42,
                 noise: bool = True,
                 adc_bits: int = 12,
                 drop_rate: float = 0.0,
                 source: str = "virtual-edge-01"):
        """
        failure_mode: 'none' | 'solar_degradation' | 'radiator_degradation'
        dt_s:         sampling interval in seconds (1 Hz by default)
        seed:         RNG seed for sensor noise + dropout (deterministic)
        noise:        enable the P2-003 Gaussian sensor-noise model
        adc_bits:     ADC resolution for quantization (12-bit default, 0=off)
        drop_rate:    probability a frame is lost in transit (0..1)
        """
        assert failure_mode in ("none", "solar_degradation", "radiator_degradation")
        assert 0.0 <= drop_rate < 1.0
        self.failure_mode = failure_mode
        self.dt_s = float(dt_s)
        self.soc = soc_init
        self.t_k = t0_k
        self.noise = noise
        self.adc_bits = int(adc_bits) if adc_bits else 0
        self.drop_rate = float(drop_rate)
        self.source = source
        self.rng = np.random.default_rng(seed)
        self._frame_id = 0

    # -- device-level sampling helpers ------------------------------------- #
    def _quantize(self, value: float, vmin: float, vmax: float) -> float:
        """12-bit ADC quantization: round to the nearest ADC step."""
        if self.adc_bits <= 0:
            return value
        levels = float(2 ** self.adc_bits)
        span = vmax - vmin
        step = span / levels
        return vmin + round((value - vmin) / step) * step

    # -- one telemetry sample ----------------------------------------------- #
    def step(self) -> Optional[TelemetryFrame]:
        """Advance the device by one sampling interval and return the frame,
        or None if this sample was dropped in transit."""
        t = self._frame_id * self.dt_s

        # power side
        deg_factor = get_solar_degradation(t, self.failure_mode)
        solar_w, load_w, soc_new, voltage_v, _net_w = compute_power_step(
            t, self.soc, deg_factor)

        # thermal side
        eps_eff, area_eff, _epsA = get_radiator_effective_epsilon_area(
            t, self.failure_mode)
        t_new, q_in, q_out, _dT = compute_thermal_step(
            t, self.t_k, eps_eff, area_eff, q_in=Q_IN_NOMINAL)

        # sensor noise (P2-003 convention: 2 W / 0.01 V / 0.1 C)
        if self.noise:
            solar_w += self.rng.normal(0, 2.0)
            voltage_v += self.rng.normal(0, 0.01)
            temp_c = (t_new - 273.15) + self.rng.normal(0, 0.1)
            q_out += self.rng.normal(0, 0.5)
        else:
            temp_c = t_new - 273.15

        # ADC quantization of the sampled channels
        if self.adc_bits > 0:
            solar_w = self._quantize(solar_w, 0.0, 600.0)
            voltage_v = self._quantize(voltage_v, 20.0, 30.0)
            temp_c = self._quantize(temp_c, -60.0, 150.0)

        self.soc = soc_new
        self.t_k = t_new
        frame_id = self._frame_id
        self._frame_id += 1

        # packet dropout: a lost sample is not emitted at all
        if self.drop_rate > 0.0 and self.rng.random() < self.drop_rate:
            return None

        return TelemetryFrame(
            time_s=round(t, 4),
            solar_power_w=round(float(solar_w), 3),
            load_power_w=float(load_w),
            battery_soc=round(float(soc_new), 6),
            battery_voltage_v=round(float(voltage_v), 4),
            heat_in_w=float(q_in),
            heat_out_w=round(float(q_out), 3),
            temperature_c=round(float(temp_c), 3),
            failure_mode=self.failure_mode,
            frame_id=frame_id,
            source=self.source,
        )

    # -- stream -------------------------------------------------------------- #
    def stream(self, n_frames: Optional[int] = None,
               realtime: bool = False) -> Iterator[TelemetryFrame]:
        """Yield telemetry frames. n_frames=None streams indefinitely.

        realtime=True sleeps dt_s between samples so the stream paces like a
        live mission (useful for the CLI demo).
        """
        import time as _time
        produced = 0
        while n_frames is None or produced < n_frames:
            frame = self.step()
            if frame is not None:
                yield frame
                produced += 1
            if realtime:
                _time.sleep(self.dt_s)

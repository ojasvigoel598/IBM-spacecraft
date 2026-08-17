"""
VirtualEdgeNode — a constrained ESP32/Raspberry-Pi-class edge device, behind a
hardware abstraction so a real device can replace it with zero application
change.

What is modelled (and why):

  * sensor noise — Gaussian, seeded (P2-003 convention: 2 W / 0.01 V / 0.1 C).
    A real ADC measures a physical quantity plus transducer noise; the seed
    makes every failure mode reproducible in tests.
  * 12-bit ADC quantization of the sampled channels. A real converter has a
    finite step; downstream consumers must tolerate it.
  * sampling cadence with timing jitter — a real scheduler drifts by tens of
    ms; `jitter_s` bounds the uniform offset of each emitted timestamp.
  * boot / reboot behaviour — no samples are emitted while the device boots or
    reboots, the sample sequence (frame_id) resets after a reboot, and uptime
    counts from the last boot. Real devices do exactly this.
  * temporary sensor dropout with stale readings — during an injected sensor
    fault the channel holds its last-known-good value, `sensor_ok` goes 0 and
    the device enters `sensor_fault` state, then recovers. That is what a real
    sensor bus does on a failed read.
  * packet loss (drop_rate) — lost frames are NOT emitted, exactly like a
    full TX buffer on a constrained link or a dropped UDP datagram; the sample
    sequence still advances, leaving a gap in frame_id.
  * packet duplication (dup_rate) — retransmission on a lossy link: the same
    frame_id appears twice on the wire.
  * device state machine: boot -> nominal -> (sensor_fault) -> recovery ->
    nominal, and rebooting during a reset/reboot.
  * command -> hardware-action -> observable change: send_command() can reset
    the device, change the sampling rate, toggle sensor noise, or inject /
    clear a sensor fault, and the frame stream reflects it.
  * failure-mode ramps driven by `simulator/failures.py` — the injected faults
    are the same physics as the rest of the project.

Determinism: every source of randomness (noise, jitter, dropout, duplication)
comes from one seeded RNG, so two devices with the same seed produce
byte-identical streams — including with failures injected. The application
talks to the `EdgeDevice` contract, never to VirtualEdgeNode specifically:

    VirtualEdgeNode -> EdgeDevice (step/stream/send_command) -> application
    Real ESP32/RPi  -> EdgeDevice (same JSON-lines frames)      -> application
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional

import numpy as np

from missionmind.simulator.config import SOC_0, T0_K, DT_S
from missionmind.simulator.power import compute_power_step
from missionmind.simulator.thermal import compute_thermal_step, Q_IN_NOMINAL
from missionmind.simulator.failures import (
    get_solar_degradation, get_radiator_effective_epsilon_area,
)

from .frame import TelemetryFrame

# Device state machine (a constrained device, not the mission's fault state).
STATE_BOOT = "boot"
STATE_NOMINAL = "nominal"
STATE_SENSOR_FAULT = "sensor_fault"
STATE_RECOVERY = "recovery"
STATE_REBOOTING = "rebooting"

# Channels the device samples and can lose to a sensor fault.
SAMPLED_CHANNELS = ("solar_power_w", "battery_voltage_v", "temperature_c")

COMMAND_ACTIONS = frozenset({
    "reset", "set_rate", "set_noise",
    "inject_sensor_fault", "clear_sensor_fault",
})


class EdgeDevice(ABC):
    """The hardware seam: everything the application needs from an edge
    device, whether virtual or a real ESP32/RPi.

    A physical implementation publishes the same TelemetryFrame JSON lines
    (see telemetry/frame.py) and implements the same methods, so swapping
    VirtualEdgeNode for real hardware changes nothing downstream: the sink
    (TcpTelemetryServer / LiveScorer / run_edge_demo) only sees frames.
    """

    @abstractmethod
    def step(self) -> Optional[TelemetryFrame]:
        """Advance the device by one sampling interval.

        Returns the sample frame, or None when the sample was lost in transit
        (drop), the device is booting/rebooting, or the buffer overflowed.
        """

    @abstractmethod
    def stream(self, n_frames: Optional[int] = None,
               realtime: bool = False) -> Iterator[TelemetryFrame]:
        """Yield emitted frames. n_frames=None streams indefinitely.

        realtime=True paces at the device's sampling interval (a live demo).
        """

    @abstractmethod
    def send_command(self, command: Dict) -> Dict:
        """Command -> hardware action -> observable change.

        Returns an ack dict {ack, action, device_state, applied_at_t, ...}.
        Supported actions: reset, set_rate, set_noise, inject_sensor_fault,
        clear_sensor_fault.
        """

    @abstractmethod
    def reset(self) -> Dict:
        """Reboot the device (short boot window, sample sequence restarts)."""

    @property
    @abstractmethod
    def device_state(self) -> str:
        """One of boot / nominal / sensor_fault / recovery / rebooting."""

    @property
    @abstractmethod
    def uptime_s(self) -> float:
        """Seconds since the last boot (mission clock)."""

    @property
    @abstractmethod
    def sample_seq(self) -> int:
        """Total samples taken since the last boot (dropped samples included,
        so gaps in frame_id are visible to a sink)."""


class VirtualEdgeNode(EdgeDevice):
    def __init__(self,
                 failure_mode: str = "none",
                 dt_s: float = DT_S,
                 soc_init: float = SOC_0,
                 t0_k: float = T0_K,
                 seed: int = 42,
                 noise: bool = True,
                 adc_bits: int = 12,
                 drop_rate: float = 0.0,
                 source: str = "virtual-edge-01",
                 jitter_s: float = 0.0,
                 boot_s: float = 0.0,
                 reboot_at: Optional[float] = None,
                 reboot_dur_s: float = 3.0,
                 sensor_fault: Optional[Dict] = None,
                 dup_rate: float = 0.0):
        """
        failure_mode:   'none' | 'solar_degradation' | 'radiator_degradation'
        dt_s:           nominal sampling interval in seconds (1 Hz default)
        seed:           RNG seed for noise, jitter, dropout, duplication
        noise:          enable the P2-003 Gaussian sensor-noise model
        adc_bits:       12-bit ADC quantization (0 disables)
        drop_rate:      probability a sample is lost in transit (full TX
                        buffer / dropped datagram); the sample seq still
                        advances, leaving a gap in frame_id
        jitter_s:       max uniform offset of each emitted timestamp (s)
        boot_s:         seconds of boot silence before the first sample
        reboot_at:      mission time (s) at which the device reboots itself
        reboot_dur_s:   seconds of reboot silence (no frames emitted)
        sensor_fault:   {'channel': one of SAMPLED_CHANNELS,
                         't_start': s, 't_end': s} -> channel holds
                        last-known-good and sensor_ok goes 0 during the window
        dup_rate:       probability an emitted frame is retransmitted
                        (same frame_id twice on the wire)
        """
        assert failure_mode in ("none", "solar_degradation", "radiator_degradation")
        assert 0.0 <= drop_rate < 1.0
        assert 0.0 <= dup_rate <= 1.0
        assert jitter_s >= 0.0
        assert boot_s >= 0.0 and reboot_dur_s >= 0.0
        if sensor_fault is not None:
            assert sensor_fault["channel"] in SAMPLED_CHANNELS
            assert sensor_fault["t_end"] > sensor_fault["t_start"] >= 0.0

        self.failure_mode = failure_mode
        self.dt_s = float(dt_s)
        self.soc = soc_init
        self.t_k = t0_k
        self.noise = noise
        self.adc_bits = int(adc_bits) if adc_bits else 0
        self.drop_rate = float(drop_rate)
        self.dup_rate = float(dup_rate)
        self.source = source
        self.jitter_s = float(jitter_s)
        self.boot_s = float(boot_s)
        self.reboot_at = None if reboot_at is None else float(reboot_at)
        self.reboot_dur_s = float(reboot_dur_s)
        self.sensor_fault = sensor_fault

        self.rng = np.random.default_rng(seed)
        self._t = 0.0                      # mission clock (physics time)
        self._sample_seq = 0               # sample counter (frame_id source)
        self._last_boot_t = 0.0            # mission time of the last boot
        self._reboots = 0
        self._reboot_until = None
        self._boot_until = self.boot_s     # mission time at which boot ends
        self._recovery_until = None        # transient recovery state
        self._last_good = {c: None for c in SAMPLED_CHANNELS}
        self._fault_active = False

    # -- device-level sampling helpers ------------------------------------- #

    def _quantize(self, value: float, vmin: float, vmax: float) -> float:
        """ADC quantization: round to the nearest ADC step."""
        if self.adc_bits <= 0:
            return value
        levels = float(2 ** self.adc_bits)
        span = vmax - vmin
        step = span / levels
        return vmin + round((value - vmin) / step) * step

    def _in_boot(self) -> bool:
        return self._t < self._boot_until

    def _in_reboot(self) -> bool:
        if self._reboot_until is None:
            return False
        return self._t < self._reboot_until

    def _reboot(self) -> None:
        """Start a reboot: silence, then the sample sequence restarts."""
        self._reboot_until = self._t + self.reboot_dur_s
        self._last_boot_t = self._t
        self._reboots += 1
        self._sample_seq = 0        # a real device restarts its sample counter
        self._recovery_until = None

    # -- device state -------------------------------------------------------- #

    @property
    def device_state(self) -> str:
        if self._in_reboot():
            return STATE_REBOOTING
        if self._in_boot():
            return STATE_BOOT
        if self._recovery_until is not None and self._t < self._recovery_until:
            return STATE_RECOVERY
        if self._fault_active:
            return STATE_SENSOR_FAULT
        return STATE_NOMINAL

    @property
    def uptime_s(self) -> float:
        return max(0.0, self._t - self._last_boot_t)

    @property
    def sample_seq(self) -> int:
        return self._sample_seq

    # -- one telemetry sample ----------------------------------------------- #

    def step(self) -> Optional[TelemetryFrame]:
        """Advance the device one nominal sampling interval and return the
        frame, or None when the device is booting/rebooting or the sample was
        lost in transit (drop / full TX buffer)."""
        t = self._t

        # device-level silence (boot / reboot): no sample is taken (the
        # mission clock still advances, but no sequence number is consumed)
        if self._in_reboot() or self._in_boot():
            self._advance_clock()
            return None

        frame_id = self._sample_seq   # sequence counts samples, incl. drops
        self._sample_seq += 1

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

        # sensor dropout: during an injected fault the channel HOLDS its
        # last-known-good ADC reading (a real bus returns the last valid
        # conversion) and the frame is flagged stale.
        self._fault_active = False
        fault = self.sensor_fault
        if fault is not None and fault["t_start"] <= t < fault["t_end"]:
            ch = fault["channel"]
            if self._last_good[ch] is not None:
                if ch == "solar_power_w":
                    solar_w = self._last_good[ch]
                elif ch == "battery_voltage_v":
                    voltage_v = self._last_good[ch]
                else:
                    temp_c = self._last_good[ch]
            self._fault_active = True

        # remember the last good reading per channel (after noise+ADC)
        self._last_good["solar_power_w"] = solar_w
        self._last_good["battery_voltage_v"] = voltage_v
        self._last_good["temperature_c"] = temp_c

        self.soc = soc_new
        self.t_k = t_new

        # timing jitter: the emitted timestamp carries the scheduler's drift
        # (bounded uniform offset); the physics runs on the mission clock.
        emitted_t = t
        if self.jitter_s > 0.0:
            emitted_t = t + self.rng.uniform(-self.jitter_s, self.jitter_s)
            emitted_t = max(emitted_t, 0.0)

        # transient recovery after a sensor fault clears
        if self._fault_active:
            self._recovery_until = None
        elif self._recovery_until is None and self.sensor_fault is not None \
                and self.sensor_fault["t_end"] <= t < self.sensor_fault["t_end"] + self.dt_s:
            self._recovery_until = t + self.dt_s

        state = self.device_state
        up = self.uptime_s            # uptime at THIS sample's mission time
        self._advance_clock()

        # packet dropout: a lost sample is not emitted at all (seq gap stays)
        if self.drop_rate > 0.0 and self.rng.random() < self.drop_rate:
            return None

        return TelemetryFrame(
            time_s=round(emitted_t, 4),
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
            device_state=state,
            sensor_ok=0 if self._fault_active else 1,
            uptime_s=round(up, 3),
        )

    def _advance_clock(self) -> None:
        """Advance the mission clock one nominal interval and evaluate any
        scheduled self-reboot."""
        self._t += self.dt_s
        if self.reboot_at is not None and self._t >= self.reboot_at \
                and self._reboot_until is None and not self._in_boot():
            self._reboot()

    # -- stream -------------------------------------------------------------- #

    def stream(self, n_frames: Optional[int] = None,
               realtime: bool = False) -> Iterator[TelemetryFrame]:
        """Yield emitted frames. n_frames=None streams indefinitely.

        realtime=True sleeps dt_s between samples so the stream paces like a
        live mission. Duplicated frames (dup_rate) are retransmitted
        immediately after the original, with the same frame_id.
        """
        import time as _time
        produced = 0
        while n_frames is None or produced < n_frames:
            frame = self.step()
            if frame is not None:
                yield frame
                produced += 1
                if self.dup_rate > 0.0 and self.rng.random() < self.dup_rate:
                    yield frame          # retransmission: same frame_id
                    produced += 1
            if realtime:
                _time.sleep(self.dt_s)

    # -- commands ------------------------------------------------------------- #

    def send_command(self, command: Dict) -> Dict:
        """Command -> hardware action -> observable change in the frame stream.

        Actions:
          reset                    -> reboot (silence, frame_id restarts)
          set_rate {dt_s}          -> change the sampling interval
          set_noise {enabled}      -> toggle the sensor-noise model
          inject_sensor_fault {channel, t_start, t_end}
                                   -> schedule a sensor dropout
          clear_sensor_fault       -> remove any scheduled sensor fault
        """
        action = command.get("action")
        if action not in COMMAND_ACTIONS:
            return {"ack": False, "action": action,
                    "reason": f"unknown action {action!r}; expected one of "
                              f"{sorted(COMMAND_ACTIONS)}",
                    "device_state": self.device_state,
                    "applied_at_t": round(self._t, 3)}
        if action == "reset":
            self._reboot()
            detail = {"reboot_dur_s": self.reboot_dur_s}
        elif action == "set_rate":
            new_dt = float(command.get("dt_s", self.dt_s))
            if new_dt <= 0.0:
                return {"ack": False, "action": action,
                        "reason": "dt_s must be > 0", "device_state": self.device_state,
                        "applied_at_t": round(self._t, 3)}
            self.dt_s = new_dt
            detail = {"dt_s": new_dt}
        elif action == "set_noise":
            self.noise = bool(command.get("enabled", True))
            detail = {"noise": self.noise}
        elif action == "inject_sensor_fault":
            ch = command.get("channel")
            if ch not in SAMPLED_CHANNELS:
                return {"ack": False, "action": action,
                        "reason": f"channel must be one of {SAMPLED_CHANNELS}",
                        "device_state": self.device_state,
                        "applied_at_t": round(self._t, 3)}
            t0 = float(command.get("t_start", self._t))
            t1 = float(command.get("t_end", t0 + self.dt_s))
            if t1 <= t0:
                return {"ack": False, "action": action,
                        "reason": "t_end must be > t_start",
                        "device_state": self.device_state,
                        "applied_at_t": round(self._t, 3)}
            self.sensor_fault = {"channel": ch, "t_start": t0, "t_end": t1}
            detail = {"channel": ch, "t_start": t0, "t_end": t1}
        else:  # clear_sensor_fault
            self.sensor_fault = None
            self._fault_active = False
            detail = {}
        return {"ack": True, "action": action, "device_state": self.device_state,
                "applied_at_t": round(self._t, 3), **detail}

    def reset(self) -> Dict:
        return self.send_command({"action": "reset"})

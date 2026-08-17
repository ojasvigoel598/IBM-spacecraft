"""Tests for the hardened edge-device contract (telemetry/edge_node.py).

The device must behave like constrained real hardware behind a stable
abstraction: deterministic (seeded) sampling with jitter, a device state
machine (boot / nominal / sensor_fault / recovery / rebooting), sensor
dropout with stale last-known-good readings, packet duplication, reboot
semantics (silence + sequence restart + uptime reset), and a command channel
whose effects are observable in the frame stream.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.telemetry.edge_node import (
    EdgeDevice, VirtualEdgeNode, STATE_BOOT, STATE_NOMINAL, STATE_SENSOR_FAULT,
    STATE_REBOOTING, SAMPLED_CHANNELS,
)
from missionmind.telemetry.frame import TelemetryFrame


def test_virtual_node_implements_edge_device_contract():
    node = VirtualEdgeNode()
    assert isinstance(node, EdgeDevice)
    # the contract the application depends on
    assert hasattr(node, "step")
    assert hasattr(node, "stream")
    assert hasattr(node, "send_command")
    assert hasattr(node, "reset")
    assert isinstance(node.device_state, str)
    assert isinstance(node.uptime_s, float)
    f = node.step()
    assert isinstance(f, TelemetryFrame)


def test_timing_jitter_changes_timestamps_deterministically():
    n1 = VirtualEdgeNode(jitter_s=0.5, seed=7)
    n2 = VirtualEdgeNode(jitter_s=0.5, seed=7)
    ts1 = [f.time_s for f in n1.stream(20)]
    ts2 = [f.time_s for f in n2.stream(20)]
    assert ts1 == ts2, "same seed must reproduce the jittered timestamps"
    # jitter is bounded by +-0.5 s around the nominal sample times
    for t, i in zip(ts1, range(20)):
        assert abs(t - i) <= 0.5 + 1e-9
    # some sample must actually be off the nominal grid (jitter is real)
    assert any(abs(t - i) > 0.01 for t, i in zip(ts1, range(20)))
    # zero jitter keeps the exact clock (backward compatible)
    n0 = VirtualEdgeNode(jitter_s=0.0, seed=7)
    ts0 = [f.time_s for f in n0.stream(5)]
    assert ts0 == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_boot_silence_then_state_transitions():
    node = VirtualEdgeNode(boot_s=2.0, dt_s=1.0, seed=1)
    # no samples while booting, and the device reports the boot state
    assert node.step() is None
    assert node.device_state == STATE_BOOT
    assert node.step() is None
    assert node.device_state == STATE_NOMINAL   # t=2.0: boot ends exactly here
    # the sample after boot silence is emitted in the nominal state
    f = node.step()
    assert f is not None and f.device_state == STATE_NOMINAL
    assert node.device_state == STATE_NOMINAL


def test_sensor_fault_holds_last_known_good_and_recovers():
    node = VirtualEdgeNode(failure_mode="none", dt_s=1.0, seed=2,
                           sensor_fault={"channel": "solar_power_w",
                                         "t_start": 10.0, "t_end": 14.0})
    frames = list(node.stream(25))
    good = [f for f in frames if f.time_s < 10]
    stale = [f for f in frames if 10.0 <= f.time_s < 14.0]
    after = [f for f in frames if f.time_s >= 14.0]
    assert good and stale and after
    # during the window: sensor_ok=0, device_state=sensor_fault, and the
    # channel holds its last known good value (frozen, not dropping further)
    for f in stale:
        assert f.sensor_ok == 0
        assert f.device_state == STATE_SENSOR_FAULT
    assert stale[0].solar_power_w == stale[-1].solar_power_w
    # the frozen value is the LAST GOOD reading, i.e. within the nominal band
    assert 500.0 < stale[0].solar_power_w < 530.0
    # recovery: the device goes through a transient recovery state
    rec = [f for f in frames if 14.0 <= f.time_s < 15.0]
    assert rec and rec[0].device_state == "recovery"
    # back to nominal with live readings
    assert after[-1].device_state == STATE_NOMINAL
    assert after[-1].sensor_ok == 1


def test_packet_duplication_retransmits_same_frame_id():
    node = VirtualEdgeNode(dup_rate=1.0, seed=3)   # every frame duplicated
    frames = list(node.stream(6))
    # stream(n) yields n emitted frames; the duplicates arrive as consecutive
    # retransmissions with the same frame_id and identical payload
    assert len(frames) == 6
    for i in range(0, 6, 2):
        assert frames[i].frame_id == frames[i + 1].frame_id
        assert frames[i].to_json_line() == frames[i + 1].to_json_line()
    # no duplication by default
    node0 = VirtualEdgeNode(seed=3)
    f0 = list(node0.stream(6))
    assert len(f0) == 6 and len({f.frame_id for f in f0}) == 6


def test_reboot_silence_sequence_restart_uptime_reset():
    node = VirtualEdgeNode(dt_s=1.0, seed=4, reboot_at=10.0, reboot_dur_s=3.0)
    frames = list(node.stream(40))
    # no frame carries the reboot window (silence)
    assert not any(10.0 <= f.time_s < 13.0 for f in frames)
    before = [f for f in frames if f.time_s < 10.0]
    after = [f for f in frames if f.time_s >= 13.0]
    assert before and after
    # the sample sequence restarts from 0 after the reboot (real device
    # behaviour: a hardware sample counter restarts on boot)
    assert after[0].frame_id == 0
    assert before[-1].frame_id > 0
    # uptime resets: the frame right after reboot reports ~3 s of uptime
    assert after[0].uptime_s < 4.0
    assert before[-1].uptime_s >= 9.0
    # reboot is deterministic with the same seed
    node2 = VirtualEdgeNode(dt_s=1.0, seed=4, reboot_at=10.0, reboot_dur_s=3.0)
    assert [f.to_json_line() for f in node2.stream(40)] == \
        [f.to_json_line() for f in frames]


def test_commands_change_the_stream():
    node = VirtualEdgeNode(dt_s=1.0, seed=5)
    # unknown action -> ack False with a reason, no state change
    bad = node.send_command({"action": "warp_drive"})
    assert bad["ack"] is False and "unknown action" in bad["reason"]
    # set_rate changes the sampling cadence from the next sample on
    ok = node.send_command({"action": "set_rate", "dt_s": 2.0})
    assert ok["ack"] is True and ok["dt_s"] == 2.0
    f1 = node.step()          # t=0 (rate change applies from the next sample)
    f2 = node.step()          # t=2 (2 s cadence now in effect)
    assert f1 is not None and f2 is not None
    assert abs((f2.time_s - f1.time_s) - 2.0) < 1e-9
    # set_noise toggles the noise model off (deterministic, no RNG draw)
    node.send_command({"action": "set_noise", "enabled": False})
    # inject_sensor_fault schedules a dropout observable in the stream
    node.send_command({"action": "inject_sensor_fault",
                       "channel": "temperature_c",
                       "t_start": 4.0, "t_end": 6.0})
    frames = list(node.stream(8))
    stale = [f for f in frames if 4.0 <= f.time_s < 6.0]
    assert stale and all(f.sensor_ok == 0 for f in stale)
    # reset reboots: silence + sequence restart
    ack = node.send_command({"action": "reset"})
    assert ack["ack"] is True
    assert node.device_state == STATE_REBOOTING
    assert node.step() is None and node.step() is None   # reboot silence
    f2 = node.step()
    assert f2 is not None and f2.frame_id <= 1


def test_fault_injection_and_recovery_deterministic():
    """Full determinism with everything on: same seed -> identical streams."""
    kw = dict(failure_mode="solar_degradation", dt_s=1.0, seed=11,
              noise=True, adc_bits=12, drop_rate=0.2, dup_rate=0.1,
              jitter_s=0.1, boot_s=0.0,
              sensor_fault={"channel": "battery_voltage_v",
                            "t_start": 500.0, "t_end": 520.0})
    a = list(VirtualEdgeNode(**kw).stream(300))
    b = list(VirtualEdgeNode(**kw).stream(300))
    assert len(a) == len(b)
    assert [f.to_json_line() for f in a] == [f.to_json_line() for f in b]
    assert all(f.device_state in ("nominal", "recovery") for f in a)
    assert all(f.sensor_ok in (0, 1) for f in a)


def test_wire_round_trip_carries_device_header():
    f = TelemetryFrame(time_s=1.0, solar_power_w=520.0, load_power_w=400.0,
                       battery_soc=0.99, battery_voltage_v=27.9,
                       heat_in_w=60.0, heat_out_w=58.0, temperature_c=-40.0,
                       device_state="sensor_fault", sensor_ok=0, uptime_s=12.5)
    line = f.to_json_line()
    f2 = TelemetryFrame.from_json_line(line)
    assert f2.device_state == "sensor_fault" and f2.sensor_ok == 0
    assert abs(f2.uptime_s - 12.5) < 1e-9
    # an old device payload without the header fields still parses
    import json
    old = {k: v for k, v in f.to_dict().items()
           if k not in ("device_state", "sensor_ok", "uptime_s")}
    f3 = TelemetryFrame.from_json_line(json.dumps(old))
    assert f3.device_state == "nominal" and f3.sensor_ok == 1


if __name__ == "__main__":
    tests = [test_virtual_node_implements_edge_device_contract,
             test_timing_jitter_changes_timestamps_deterministically,
             test_boot_silence_then_state_transitions,
             test_sensor_fault_holds_last_known_good_and_recovers,
             test_packet_duplication_retransmits_same_frame_id,
             test_reboot_silence_sequence_restart_uptime_reset,
             test_commands_change_the_stream,
             test_fault_injection_and_recovery_deterministic,
             test_wire_round_trip_carries_device_header]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print("\nAll edge-device contract tests PASS")

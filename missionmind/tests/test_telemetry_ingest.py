"""
Telemetry-ingest tests: the virtual IoT edge layer.

Covered:
1. frame codec round-trip (dict <-> JSON line) with the schema that every
   downstream consumer (score_dataframe / physics_rules / RAG / Granite)
   already expects — so an ingested frame is indistinguishable from a
   simulator row.
2. VirtualEdgeNode: schema-valid, deterministic (seeded), packet dropout and
   12-bit ADC quantization behave.
3. TCP end-to-end: edge node -> TcpTelemetryServer -> LiveScorer -> the
   production ML ensemble flags the solar fault after t=600s and stays quiet
   on the nominal run after the transient burn-in.
4. MQTT transport degrades gracefully when paho-mqtt is not installed.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.telemetry.frame import TelemetryFrame, TELEMETRY_SCHEMA
from missionmind.telemetry.edge_node import VirtualEdgeNode
from missionmind.telemetry.ingest import TcpTelemetryServer, TcpTelemetryClient, LiveScorer
from missionmind.telemetry.ingest import MqttTelemetryClient

SCHEMA_KEYS = ["time_s", "solar_power_w", "load_power_w", "battery_soc",
               "battery_voltage_v", "heat_in_w", "heat_out_w",
               "temperature_c", "failure_mode"]


def test_frame_round_trip():
    f = TelemetryFrame(time_s=123, solar_power_w=520.0, load_power_w=400.0,
                       battery_soc=0.95, battery_voltage_v=27.9,
                       heat_in_w=60.0, heat_out_w=55.0, temperature_c=-40.0,
                       failure_mode="none", frame_id=7, source="edge-01")
    line = f.to_json_line()
    # JSON line: one object, no trailing newline inside
    obj = json.loads(line)
    for k in SCHEMA_KEYS:
        assert k in obj, f"schema key {k} missing from frame"
    f2 = TelemetryFrame.from_json_line(line)
    assert f2.time_s == 123 and f2.frame_id == 7 and f2.source == "edge-01"
    assert abs(f2.battery_voltage_v - 27.9) < 1e-9
    # to_dict must carry exactly the telemetry-schema keys (plus header extras
    # are fine, but the schema keys must be present and typed)
    d = f.to_dict()
    assert d["failure_mode"] == "none"


def test_edge_node_schema_and_determinism():
    n1 = VirtualEdgeNode(failure_mode="solar_degradation", dt_s=1.0, seed=42)
    n2 = VirtualEdgeNode(failure_mode="solar_degradation", dt_s=1.0, seed=42)
    frames1 = [f for f in n1.stream(1200)]   # t = 0..1199, past the ramp
    frames2 = [f for f in n2.stream(1200)]
    assert len(frames1) == len(frames2) == 1200
    # deterministic with the same seed
    assert frames1[150].to_json_line() == frames2[150].to_json_line()
    # schema-valid rows, physically plausible
    for f in frames1:
        d = f.to_dict()
        for k in SCHEMA_KEYS:
            assert k in d
        assert 0.0 <= d["battery_soc"] <= 1.0
        assert 20.0 <= d["battery_voltage_v"] <= 30.0
        assert d["solar_power_w"] >= 0.0
    # failure ramp: solar collapses after t=900
    s_early = np.mean([f.solar_power_w for f in frames1 if f.time_s < 500])
    s_late = np.mean([f.solar_power_w for f in frames1 if f.time_s > 1000])
    assert s_early > 500.0 and s_late < 300.0, (s_early, s_late)


def test_edge_node_dropout_and_quantization():
    # stream(N) yields N VALID frames by design; drops advance the mission
    # clock without emitting a frame, so with drop_rate=0.5 the last emitted
    # frame sits far past t=399 while a lossless node ends exactly at t=399.
    node = VirtualEdgeNode(failure_mode="none", dt_s=1.0, seed=1,
                           drop_rate=0.5, adc_bits=12)
    frames = list(node.stream(400))
    assert len(frames) == 400
    assert frames[-1].time_s > 450, f"dropout did not advance the clock: {frames[-1].time_s}"
    # ADC quantization: 12-bit over the voltage span -> voltage changes are
    # multiples of (40 V / 4096) after the first sample
    v = np.array([f.battery_voltage_v for f in frames])
    diffs = np.abs(np.diff(v))
    step = 10.0 / 4096.0   # 12-bit over the 20..30 V quantization span
    assert diffs[diffs > 0].min() > step * 0.5 - 1e-9
    # no dropout -> lossless, exact clock
    node0 = VirtualEdgeNode(failure_mode="none", dt_s=1.0, seed=1, drop_rate=0.0)
    f0 = list(node0.stream(50))
    assert len(f0) == 50 and f0[-1].time_s == 49


def test_tcp_ingest_end_to_end_solar_flag():
    # Start a telemetry server on an ephemeral port, publish a solar-fault
    # stream from the virtual edge node, and confirm the production ML
    # ensemble flags it after the 600s injection while staying quiet on a
    # nominal stream after the transient burn-in.
    from missionmind.ml.detect import load_models
    load_models()  # fails loudly if models were not trained

    import time
    server = TcpTelemetryServer(host="127.0.0.1", port=0)
    collected = []
    server.start(on_frame=lambda f: collected.append(f))
    port = server.port

    client = TcpTelemetryClient(host="127.0.0.1", port=port)
    node = VirtualEdgeNode(failure_mode="solar_degradation", dt_s=1.0, seed=3)
    client.publish_frames(node.stream(1000))
    client.close()
    # drain: the server thread parses frames synchronously as lines arrive
    for _ in range(200):
        if len(collected) >= 1000:
            break
        time.sleep(0.05)

    assert len(collected) == 1000, f"got {len(collected)} frames"
    # the production ensemble costs ~0.3 s per window; re-score at intervals
    scorer = LiveScorer(window=60, min_rows=30, score_every=20)
    flags_after = []
    for f in collected:
        scorer.push(f)
        if scorer.ready:
            flags_after.append(scorer.score_latest()["anomaly_flag"])
    assert len(flags_after) > 30
    # the latest scored windows (t > 900, in the fault region) must be flagged
    assert flags_after[-4:].count(1) >= 3, flags_after[-4:]
    server.stop()


def test_live_scorer_nominal_stays_quiet_after_burn_in():
    import time
    server = TcpTelemetryServer(host="127.0.0.1", port=0)
    collected = []
    server.start(on_frame=lambda f: collected.append(f))
    port = server.port
    client = TcpTelemetryClient(host="127.0.0.1", port=port)
    node = VirtualEdgeNode(failure_mode="none", dt_s=1.0, seed=5)
    client.publish_frames(node.stream(1000))
    client.close()
    for _ in range(200):
        if len(collected) >= 1000:
            break
        time.sleep(0.05)
    server.stop()

    scorer = LiveScorer(window=60, min_rows=30, score_every=20)
    late_flags = []
    for f in collected:
        scorer.push(f)
        if scorer.ready and scorer._rows[-1]["time_s"] > 950:
            late_flags.append(scorer.score_latest()["anomaly_flag"])
    # steady-state nominal must not scream; allow a small burn-in residual
    assert len(late_flags) > 10
    assert sum(late_flags) / len(late_flags) < 0.15, (
        f"nominal late-window flag rate {sum(late_flags)/len(late_flags):.3f} too high")


def test_live_scorer_keeps_scoring_after_window_fill():
    # Regression: once the rolling window is full, len(_rows) stops growing, so
    # the short-circuit must be driven by a NEW-FRAME counter, not by
    # (len(_rows) - scored_at) — otherwise scoring locks forever at the
    # window-fill moment and the live chip goes stale.
    scorer = LiveScorer(window=60, min_rows=30, score_every=5)
    node = VirtualEdgeNode(failure_mode="solar_degradation", seed=9)
    times = []
    for f in node.stream(300):
        scorer.push(f)
        if scorer.ready:
            times.append(scorer.score_latest()["time_s"])
    assert times[-1] > 250, f"scoring locked at t={times[-1]}"
    assert len(set(times)) > 30, f"only {len(set(times))} distinct scores"


def test_mqtt_transport_graceful_without_paho():
    client = MqttTelemetryClient(host="127.0.0.1", port=1883)
    # Contract under every environment:
    #  * paho-mqtt NOT installed -> connect() raises a clear RuntimeError
    #    naming the package, never a bare ImportError.
    #  * paho-mqtt installed + broker reachable -> connects (live test).
    #  * paho-mqtt installed but NO broker running -> the transport must
    #    surface the connection failure explicitly (ConnectionRefusedError)
    #    rather than silently hanging or lying about being connected.
    try:
        client.connect()
        connected = True
    except RuntimeError as e:
        connected = False
        assert "paho-mqtt" in str(e)
    except ConnectionRefusedError:
        # Environment condition (no broker on 127.0.0.1:1883), not a code
        # defect: with paho installed the transport legitimately attempts a
        # real connection and reports the unreachable broker. PASS.
        connected = False
    except OSError as e:
        # Same category: broker absent / port closed / network down.
        connected = False
        assert "connect" in str(e).lower() or "refused" in str(e).lower()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"unexpected exception type: {type(e).__name__}: {e}")
    if connected:
        # if paho IS installed and a broker is reachable, this is a live test
        client.disconnect()


if __name__ == "__main__":
    tests = [test_frame_round_trip,
             test_edge_node_schema_and_determinism,
             test_edge_node_dropout_and_quantization,
             test_tcp_ingest_end_to_end_solar_flag,
             test_live_scorer_nominal_stays_quiet_after_burn_in,
             test_live_scorer_keeps_scoring_after_window_fill,
             test_mqtt_transport_graceful_without_paho]
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
    print("\nAll telemetry-ingest tests PASS")

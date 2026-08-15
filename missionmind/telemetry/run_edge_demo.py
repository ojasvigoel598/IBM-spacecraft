#!/usr/bin/env python3
"""MissionMind virtual-IoT edge demo.

A simulated ESP32-class edge device streams physics-derived telemetry frames
over TCP (JSON-lines) to the ingest server; the production ML ensemble scores
each window live and prints an operator-style table.

Run:
    .venv/Scripts/python.exe -m missionmind.telemetry.run_edge_demo
    .venv/Scripts/python.exe -m missionmind.telemetry.run_edge_demo --fault radiator_degradation
    .venv/Scripts/python.exe -m missionmind.telemetry.run_edge_demo --realtime   # pace at 1 Hz

If you had real hardware: an ESP32/RPi publishing the same JSON lines to the
same TCP port (or an MQTT topic) replaces the virtual node with zero changes
downstream — that is the drop-in electronics contract.
"""

import argparse
import sys
import time

# Windows console is cp1252 by default; the arrows/deltas below need UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .frame import TELEMETRY_SCHEMA
from .edge_node import VirtualEdgeNode
from .ingest import TcpTelemetryServer, TcpTelemetryClient, LiveScorer

SCENARIOS = {
    "none": "nominal operation",
    "solar_degradation": "solar array degradation (ramp 600-900 s)",
    "radiator_degradation": "radiator degradation (ramp 600-900 s)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fault", choices=sorted(SCENARIOS), default="solar_degradation")
    ap.add_argument("--frames", type=int, default=1200, help="frames to stream (1 s each)")
    ap.add_argument("--realtime", action="store_true", help="pace the stream at 1 Hz")
    ap.add_argument("--drop", type=float, default=0.0, help="packet-drop probability")
    ap.add_argument("--noise-off", action="store_true", help="disable sensor noise")
    args = ap.parse_args()

    print("=" * 78)
    print("MissionMind — virtual IoT edge → ingest → live ML ensemble")
    print("=" * 78)
    print(f"scenario : {args.fault}  ({SCENARIOS[args.fault]})")
    print(f"edge node: virtual ESP32-class, ADC 12-bit, drop {args.drop:.0%}, "
          f"noise {'off' if args.noise_off else 'on'}")
    print(f"wire     : TCP JSON-lines (a real ESP32/RPi publishes the same "
          f"frames to this port)")
    print(f"consumer : production IsolationForest ensemble + physics rules\n")

    server = TcpTelemetryServer(host="127.0.0.1", port=0)
    collected = []
    server.start(on_frame=lambda f: collected.append(f))
    print(f"[ingest] listening on 127.0.0.1:{server.port} — connecting edge node...")

    client = TcpTelemetryClient(host="127.0.0.1", port=server.port)
    node = VirtualEdgeNode(failure_mode=args.fault, dt_s=1.0, seed=42,
                           noise=not args.noise_off, drop_rate=args.drop)
    scorer = LiveScorer(window=120, min_rows=30, score_every=10)

    print(f"[edge]   streaming {args.frames} frames "
          f"({'realtime' if args.realtime else 'as fast as possible'})\n")
    print(f"{'T+':>6s} {'solar(W)':>9s} {'temp(C)':>8s} {'V(V)':>7s} "
          f"{'score':>8s} {'flag':>4s} {'src':>7s}   note")
    print("-" * 66)

    last_report = -1
    n_flagged = 0
    t_start = time.time()
    for f in node.stream(args.frames, realtime=args.realtime):
        client.send(f)
        scorer.push(f)
        if not scorer.ready:
            continue
        s = scorer.score_latest()
        if s["time_s"] - last_report >= 60 or s["time_s"] == 0:
            last_report = s["time_s"]
            note = ""
            if s["anomaly_flag"]:
                n_flagged += 1
                src = {0: "full", 1: "power", 2: "thermal"}.get(s["anomaly_source"], "?")
                note = f"⚠ {src} model fired"
            elif s["time_s"] > 900:
                note = "steady state"
            print(f"{s['time_s']:>6d} {s['solar_power_w']:>9.1f} "
                  f"{s['temperature_c']:>8.2f} {s['battery_voltage_v']:>7.2f} "
                  f"{s['anomaly_score']:>8.3f} {s['anomaly_flag']:>4d} "
                  f"{s['anomaly_source']:>7d}   {note}")
    client.close()
    server.stop()

    elapsed = time.time() - t_start
    print("-" * 66)
    # post-injection flag rate over the scored windows with t > 900
    late_flags, late_scores = 0, 0
    scorer2 = LiveScorer(window=60, min_rows=30, score_every=10)
    for f in collected:
        scorer2.push(f)
        if scorer2.ready and scorer2._rows[-1]["time_s"] > 900:
            late_flags += scorer2.score_latest()["anomaly_flag"]
            late_scores += 1
    print(f"\n[result] {len(collected)} frames ingested in {elapsed:.1f}s "
          f"({len(collected)/max(elapsed,0.01):.0f} fps)")
    print(f"[result] post-injection flag rate (t>900): "
          f"{late_flags/max(1, late_scores):.2f} ({late_flags}/{late_scores}) "
          f"— the ML ensemble reacts to the fault as it arrives over the wire")
    print(f"[note]   a real ESP32/RPi publishing the same JSON lines to the "
          f"same TCP port replaces the virtual node — no downstream change.")


if __name__ == "__main__":
    main()

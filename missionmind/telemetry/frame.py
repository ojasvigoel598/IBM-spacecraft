"""
Telemetry frame protocol.

One JSON-lines object per telemetry sample, carrying the SAME schema as
`simulator/run_scenarios.py` output so every downstream consumer
(`ml/detect.score_dataframe`, physics rules, RAG, Granite) works unchanged on
ingested frames. Header fields (frame_id, source, schema_version) let a real
edge device be identified and the stream audited.

Real-hardware drop-in contract: an ESP32 / Raspberry Pi publishes the same
JSON lines to the same TCP port (or MQTT topic) — nothing else in the
pipeline changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict

# Must match simulator/run_scenarios.py output columns.
TELEMETRY_SCHEMA = [
    "time_s", "solar_power_w", "load_power_w", "battery_soc",
    "battery_voltage_v", "heat_in_w", "heat_out_w", "temperature_c",
    "failure_mode",
]

# 1.1 adds the device header fields a constrained edge device actually
# publishes (device_state, sensor_ok, uptime_s). from_json_line tolerates
# payloads without them (older devices / the simulator), so the wire format
# stays backward compatible.
SCHEMA_VERSION = "1.1"


@dataclass
class TelemetryFrame:
    time_s: float
    solar_power_w: float
    load_power_w: float
    battery_soc: float
    battery_voltage_v: float
    heat_in_w: float
    heat_out_w: float
    temperature_c: float
    failure_mode: str = "none"
    frame_id: int = 0
    source: str = "virtual-edge-01"
    schema_version: str = SCHEMA_VERSION
    # Device header (schema 1.1): what a real ESP32/RPi would publish on the
    # wire so the sink can see device health, not just sensor values.
    device_state: str = "nominal"   # boot | nominal | sensor_fault | recovery | rebooting
    sensor_ok: int = 1              # 0 = >=1 channel stale (sensor dropout)
    uptime_s: float = 0.0           # seconds since the last boot

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json_line(self) -> str:
        """One line of JSON — the wire format."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "TelemetryFrame":
        obj = json.loads(line)
        # tolerate a payload without the header extras (e.g. an old device)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in obj.items() if k in allowed}
        return cls(**payload)

    def to_dataframe_row(self) -> Dict:
        """Dict with exactly the telemetry-schema keys (DataFrame-compatible)."""
        d = asdict(self)
        return {k: d[k] for k in TELEMETRY_SCHEMA}

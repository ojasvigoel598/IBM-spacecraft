"""MissionMind telemetry-ingest layer: virtual IoT edge -> transport -> ML.

This is the "electronics side" of the digital twin: a simulated edge node
(ESP32-class device) publishes physics-derived telemetry frames over a real
transport (TCP JSON-lines by default, MQTT when paho-mqtt is available), and
the existing ML ensemble / physics rules / RAG / Granite consume the frames
exactly as they consume simulator CSVs.

The frame schema is identical to `simulator/run_scenarios.py` output, so any
downstream consumer works unchanged — and a real ESP32/RPi can replace the
virtual node by publishing the same JSON lines to the same endpoint.
"""

from .frame import TelemetryFrame, TELEMETRY_SCHEMA
from .edge_node import VirtualEdgeNode
from .ingest import TcpTelemetryServer, TcpTelemetryClient, LiveScorer
from .ingest import MqttTelemetryClient

__all__ = [
    "TelemetryFrame", "TELEMETRY_SCHEMA", "VirtualEdgeNode",
    "TcpTelemetryServer", "TcpTelemetryClient", "LiveScorer",
    "MqttTelemetryClient",
]

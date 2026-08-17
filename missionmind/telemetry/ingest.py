"""
Telemetry transports + live scoring.

Transport contract (the "real hardware" seam):
  * TcpTelemetryServer  — JSON-lines TCP sink; an edge device connects and
                          streams frames. stdlib-only, works fully offline.
  * TcpTelemetryClient  — JSON-lines TCP publisher (used by the virtual edge
                          node; a real ESP32/RPi uses the same wire format).
  * MqttTelemetryClient — paho-mqtt publisher/subscriber, enabled only when
                          paho-mqtt is installed (else a clear RuntimeError),
                          so the MQTT/Event-Streams-style path exists without
                          forcing a broker dependency on the demo.

LiveScorer feeds ingested frames through the production ML ensemble
(ml/detect.score_dataframe) on a rolling window, so the anomaly score / flag /
attribution update as frames arrive — the digital twin reacting to live data.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Callable, Iterator, List, Optional

import pandas as pd

from .frame import TelemetryFrame

# --------------------------------------------------------------------------- #
# TCP transport                                                                #
# --------------------------------------------------------------------------- #
class TcpTelemetryServer:
    """JSON-lines TCP sink. One line = one TelemetryFrame (see frame.py)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 backlog: int = 4):
        self.host = host
        self.port = port
        self.backlog = backlog
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self.port = self._sock.getsockname()[1]
        self._listener: Optional[threading.Thread] = None
        self._running = False

    def start(self, on_frame: Optional[Callable[[TelemetryFrame], None]] = None):
        """Begin accepting connections in a background thread."""
        self._on_frame = on_frame or (lambda f: None)
        self._running = True
        self._listener = threading.Thread(target=self._accept_loop, daemon=True)
        self._listener.start()
        return self

    def _accept_loop(self):
        self._sock.listen(self.backlog)
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket):
        buf = b""
        try:
            while self._running:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        try:
                            frame = TelemetryFrame.from_json_line(line.decode("utf-8"))
                            self._on_frame(frame)
                        except Exception:  # noqa: BLE001 - a bad frame is skipped, never fatal
                            pass
        finally:
            conn.close()

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        if self._listener is not None:
            self._listener.join(timeout=5)

    def join(self, timeout: Optional[float] = None):
        """Wait for in-flight connections to drain (used by tests)."""
        if self._listener is not None:
            self._listener.join(timeout=timeout)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class TcpTelemetryClient:
    """JSON-lines TCP publisher — the device side of the wire contract."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 timeout: float = 10.0):
        self.host = host
        self.port = port
        self._sock = socket.create_connection((host, port), timeout=timeout)

    def send(self, frame: TelemetryFrame):
        self._sock.sendall((frame.to_json_line() + "\n").encode("utf-8"))

    def publish_frames(self, frames: Iterator[TelemetryFrame]):
        for f in frames:
            self.send(f)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# MQTT transport (optional dependency)                                         #
# --------------------------------------------------------------------------- #
class MqttTelemetryClient:
    """MQTT publisher/subscriber behind the same frame contract.

    Enabled only when paho-mqtt is installed. Without it, connect() raises a
    clear RuntimeError — the caller decides whether to degrade (e.g. to TCP).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1883,
                 topic: str = "missionmind/telemetry", client_id: str = "mm-edge-01"):
        self.host = host
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self._client = None

    def connect(self) -> "MqttTelemetryClient":
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise RuntimeError(
                "paho-mqtt is not installed — the MQTT transport is unavailable. "
                "Install it with 'pip install paho-mqtt' or use the stdlib TCP "
                "transport instead.") from e
        self._client = mqtt.Client(client_id=self.client_id)
        self._client.connect(self.host, self.port)
        self._client.loop_start()
        return self

    def publish(self, frame: TelemetryFrame):
        if self._client is None:
            raise RuntimeError("connect() first")
        self._client.publish(self.topic, frame.to_json_line())

    def subscribe(self, on_frame: Callable[[TelemetryFrame], None]):
        if self._client is None:
            raise RuntimeError("connect() first")

        def _on_message(_c, _u, msg):
            try:
                on_frame(TelemetryFrame.from_json_line(msg.payload.decode("utf-8")))
            except Exception:  # noqa: BLE001
                pass

        self._client.on_message = _on_message
        self._client.subscribe(self.topic)

    def disconnect(self):
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None


# --------------------------------------------------------------------------- #
# Live scoring                                                                 #
# --------------------------------------------------------------------------- #
class LiveScorer:
    """Rolling-window scorer: ingested frames -> production ML ensemble.

    Keeps the last `window` frames, re-scores on each push (cheap: a few
    hundred rows through the ensemble), and exposes the LATEST row's
    anomaly_score / anomaly_flag / anomaly_source — the live "health chip".
    """

    def __init__(self, window: int = 120, min_rows: int = 30,
                 score_every: int = 5):
        assert window >= 10
        self.window = int(window)
        self.min_rows = int(min_rows)
        self.score_every = max(1, int(score_every))
        self._rows: List[dict] = []
        self._last = None
        self._unscored = 0  # new frames since the last real score
        # Load the production ensemble ONCE — score_dataframe() reloads the
        # joblib artifacts per call, which is far too slow for frame-rate
        # scoring. We cache the models and call the internal scorer directly.
        from missionmind.ml import detect
        self._models = detect.load_models()

    def push(self, frame: TelemetryFrame):
        d = frame.to_dataframe_row() if hasattr(frame, "to_dataframe_row") \
            else {k: frame.get(k) for k in ("time_s", "solar_power_w",
                                            "battery_soc", "battery_voltage_v",
                                            "temperature_c")}
        self._rows.append(d)
        if len(self._rows) > self.window:
            self._rows = self._rows[-self.window:]
        self._unscored += 1

    @property
    def ready(self) -> bool:
        return len(self._rows) >= self.min_rows

    @property
    def buffered_frames(self) -> int:
        return len(self._rows)

    def score_latest(self) -> dict:
        """Score the buffered window and return the latest row's decision.

        The production ensemble costs ~0.3 s per window, so re-scoring on every
        pushed frame would starve the loop. The window is re-scored at most
        once every `score_every` new frames; in between, the last result is
        returned (the health chip updates at a sane operator rate).
        """
        if not self.ready:
            raise RuntimeError(f"need at least {self.min_rows} frames "
                               f"(have {len(self._rows)})")
        # Short-circuit on NEW FRAMES since the last real score. Counting
        # arrivals (not len(_rows) differences) is deliberate: once the rolling
        # window is full, len(_rows) stops growing and length arithmetic would
        # lock scoring at the window-fill moment forever.
        if self._last is not None and self._unscored < self.score_every:
            return self._last
        from missionmind.ml.train import add_derivative_features
        from missionmind.ml import detect

        df = pd.DataFrame(self._rows)
        df_feat = add_derivative_features(df)
        scores, flags, attribution = detect._ensemble_score_and_flag(
            df_feat, self._models)
        row = df_feat.iloc[-1]
        self._unscored = 0

        # P9: eclipse-aware live scoring. The raw ensemble flags any solar
        # dip, but an eclipse dip is EXPECTED physics. Only suppress when the
        # measured solar matches the eclipse-adjusted expectation (residual
        # within tolerance); a fault far below that expectation keeps the flag.
        _raw_flag = int(flags[-1])
        _flag = _raw_flag
        _explained = None
        if _raw_flag and all(c in df_feat.columns
                             for c in ("in_eclipse", "sun_exposure")):
            from missionmind.physics_rules.rules import eclipse_residual
            ecl = eclipse_residual(df_feat)
            if ecl is not None and ecl["in_eclipse"]:
                if ecl["status"] == "eclipse":
                    _flag = 0
                    _explained = (f"eclipse {ecl['eclipse_frac']:.0%}: solar "
                                  f"{ecl['measured_solar_w']:.0f}W matches "
                                  f"expected {ecl['expected_solar_w']:.0f}W")
                else:
                    _explained = (f"eclipse but solar {ecl['measured_solar_w']:.0f}W "
                                  f"<< expected {ecl['expected_solar_w']:.0f}W -> fault")

        self._last = {
            "time_s": int(row["time_s"]),
            "solar_power_w": float(row["solar_power_w"]),
            "temperature_c": float(row["temperature_c"]),
            "battery_voltage_v": float(row["battery_voltage_v"]),
            "anomaly_score": float(scores[-1]),
            "anomaly_flag": _flag,
            "anomaly_source": int(attribution[-1]),
            "eclipse_explained": _explained,
        }
        return self._last

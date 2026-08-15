"""MissionMind runtime execution trace.

Records which pipeline code actually executes as telemetry flows, so the
dashboard can show the operator (or a judge) the live code path behind the
numbers: simulator steps, ML scoring per detector, physics-rule checks, RAG
retrieval, and narrative generation.

Design
------
- A small thread-safe ring buffer (default 300 events) - cheap append, no
  external dependency. Trimming on every append keeps it bounded.
- Events carry a monotonic `seq` so clients can poll with `since=` cursors.
- Instrumentation lives at the integration seams of the pipeline (detect.py,
  physics_rules/rules.py, ai/rag.py, ml/causal_narrative.py, viz/api_server.py),
  so BOTH the Streamlit app and the web console see the same trace for free.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

_MAX_EVENTS = 300
_events: List[Dict] = []
_seq = 0
_lock = threading.Lock()


def record(module: str, func: str,
           mission_t: Optional[float] = None,
           note: str = "",
           value: Optional[float] = None) -> None:
    """Append one trace event. Never raises - tracing must not break scoring."""
    global _seq
    try:
        with _lock:
            _seq += 1
            _events.append({
                "seq": _seq,
                "ts": time.time(),
                "module": module,
                "func": func,
                "mission_t": round(mission_t, 1) if mission_t is not None else None,
                "note": note,
                "value": value,
            })
            if len(_events) > _MAX_EVENTS:
                del _events[:-_MAX_EVENTS]
    except Exception:  # noqa: BLE001 - tracing is best-effort
        pass


def events_since(seq: int = 0, limit: int = 300) -> tuple:
    """Return (events with seq > `seq`, current last_seq). Oldest first."""
    with _lock:
        out = [e for e in _events if e["seq"] > seq][-limit:]
        return out, _seq


def last(n: int = 40) -> List[Dict]:
    """Most recent n events (newest first)."""
    with _lock:
        return list(reversed(_events[-n:]))

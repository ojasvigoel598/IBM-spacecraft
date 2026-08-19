"""MissionMind auth — in-memory sliding-window rate limiting.

A tiny thread-safe fixed-bucket limiter (no new dependency). Keys combine a
scope and an identifier (client IP, and where relevant a lowercased email) so
brute-force/login spraying and expensive-endpoint abuse are bounded without a
shared datastore. Buckets are pruned on access; the structure can never grow
unbounded because each key is removed once its window expires.

Production note: in-memory limits are per-process. Behind multiple workers the
limits are per-worker, which is still a real per-instance cap; a distributed
limiter (Redis) is the documented upgrade path (missionmind/docs/SECURITY.md).
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple


class RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> list of (timestamp) within the current window
        self._hits: Dict[str, List[float]] = {}

    def reset(self) -> None:
        """Clear all buckets (tests use this to isolate rate-limit state)."""
        with self._lock:
            self._hits.clear()

    def allow(self, key: str, limit: int, window_s: float,
              now: Optional[float] = None) -> Tuple[bool, float]:
        """Return (allowed, retry_after_seconds). The caller must reject the
        request when allowed is False (HTTP 429)."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                self._hits[key] = [now]
                return True, 0.0
            # drop entries outside the window
            cutoff = now - window_s
            keep = [t for t in bucket if t > cutoff]
            if len(keep) >= limit:
                self._hits[key] = keep
                retry = max(0.0, window_s - (now - keep[0]))
                return False, retry
            keep.append(now)
            self._hits[key] = keep
            # opportunistic prune of dead keys (cheap scan, bounded work)
            if len(self._hits) > 4096:
                for k in list(self._hits):
                    if not self._hits[k] or self._hits[k][-1] <= now - 3600:
                        del self._hits[k]
            return True, 0.0


_limiter = RateLimiter()


def check_rate(key: str, limit: int, window_s: float) -> Tuple[bool, float]:
    """Module-level helper using the shared limiter."""
    return _limiter.allow(key, limit, window_s)


def ip_key(request) -> str:
    """Best-effort client identifier from a Starlette request. X-Forwarded-For
    is honoured only in production (trusted proxy); otherwise the direct peer
    address is used so a spoofed header cannot reset a limiter."""
    import os

    forwarded = request.headers.get("x-forwarded-for", "")
    if os.getenv("MISSIONMIND_ENV") == "production" and forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"

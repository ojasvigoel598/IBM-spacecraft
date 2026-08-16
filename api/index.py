"""Vercel serverless entry point for the MissionMind FastAPI backend.

Vercel auto-detects Python functions in the `api/` directory and routes
`/api/*` requests to them. This module re-exports the real FastAPI app
under the name `app` so the Vercel Python runtime serves it as ASGI
directly (no changes to missionmind/viz/api_server.py were needed).

The project root is put on sys.path because the app imports the
`missionmind` package, which lives outside the `api/` directory.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from missionmind.viz.api_server import app  # noqa: E402

__all__ = ["app"]

"""MissionMind auth — real multi-user authentication for the FastAPI backend.

Zero new dependencies: SQLite (stdlib) + PBKDF2-HMAC-SHA256 (stdlib) + an
in-memory rate limiter. See missionmind/docs/SECURITY.md for the architecture
and production configuration.
"""

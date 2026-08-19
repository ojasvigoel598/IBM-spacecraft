"""MissionMind auth — FastAPI dependency guards.

Every protected endpoint depends on one of these. Authorization is decided
entirely server-side from the session cookie: the client can never claim an
identity, a role, or a verification state.

Usage:
    @app.get("/api/scenario/{mode}")
    def scenario(mode: str, user: dict = Depends(require_verified)): ...
"""

from __future__ import annotations

from typing import Optional

from fastapi import Cookie, Depends, HTTPException

from missionmind.auth import service

SESSION_COOKIE = "missionmind_session"


def get_current_user(missionmind_session: Optional[str] = Cookie(default=None)
                     ) -> dict:
    """Require a valid, unexpired session cookie. Returns the user dict."""
    user = service.get_user_by_session(missionmind_session) if missionmind_session else None
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return user


def require_verified(user: dict = Depends(get_current_user)) -> dict:
    """Require an authenticated AND email-verified account."""
    if not user.get("email_verified"):
        raise HTTPException(
            status_code=403,
            detail="email verification required",
            headers={"X-Requires-Verification": "1"},
        )
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Require an authenticated, verified admin account. Role comes from the
    database row, never from the request."""
    if not user.get("email_verified"):
        raise HTTPException(
            status_code=403,
            detail="email verification required",
            headers={"X-Requires-Verification": "1"},
        )
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin access required")
    return user

"""MissionMind auth — domain service.

All flows are enumeration-safe: the API layer returns the SAME generic
message whether an email exists or not (login, reset, resend, signup for an
already-registered address). Sessions, verification tokens and reset tokens
are single-use, expire, and are stored as SHA-256 digests.

Error policy: every public function either returns a value or raises a
domain-level exception with a GENERIC message. The API layer maps exceptions
to HTTP codes; nothing here mentions whether an account exists.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

from missionmind.auth import db, security

SESSION_TTL_S = int(os.getenv("MISSIONMIND_SESSION_TTL_S", str(7 * 24 * 3600)))  # 7 days
VERIFY_TTL_S = int(os.getenv("MISSIONMIND_VERIFY_TTL_S", str(24 * 3600)))        # 24 h
RESET_TTL_S = int(os.getenv("MISSIONMIND_RESET_TTL_S", str(60 * 60)))            # 1 h

SESSION_TTL_S = max(300, min(SESSION_TTL_S, 30 * 24 * 3600))
VERIFY_TTL_S = max(300, min(VERIFY_TTL_S, 7 * 24 * 3600))
RESET_TTL_S = max(300, min(RESET_TTL_S, 24 * 3600))

EMAIL_MAX = 254
PASSWORD_MIN = 8
PASSWORD_MAX = 128


class AuthError(Exception):
    """Base class. `message` is always safe to show to a client."""


class EmailTakenError(AuthError):
    pass


class VerificationError(AuthError):
    pass


class ResetError(AuthError):
    pass


class CredentialsError(AuthError):
    pass


class SessionError(AuthError):
    pass


def validate_email(email: str) -> str:
    """Basic structural validation (no new dependency). Returns the
    lowercased email; raises AuthError with a generic message otherwise."""
    if not isinstance(email, str):
        raise AuthError("invalid email")
    email = email.strip().lower()
    if not (3 <= len(email) <= EMAIL_MAX):
        raise AuthError("invalid email")
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise AuthError("invalid email")
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain:
        raise AuthError("invalid email")
    if any(c.isspace() for c in email):
        raise AuthError("invalid email")
    return email


def validate_password(password: str) -> None:
    """Server-side password policy. Raises AuthError with a safe message."""
    if not isinstance(password, str):
        raise AuthError("password must be 8-128 characters")
    if not (PASSWORD_MIN <= len(password) <= PASSWORD_MAX):
        raise AuthError("password must be 8-128 characters")
    if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        raise AuthError("password must contain at least one letter and one digit")


# ---- users -----------------------------------------------------------------

def create_user(email: str, password: str, role: str = "user") -> dict:
    """Create a user (email unverified). Returns {user, verification_token}.
    Raises EmailTakenError with a generic message if the address exists."""
    email = validate_email(email)
    validate_password(password)
    pwd_hash, salt = security.hash_password(password)
    now = db.utcnow_iso()
    conn = db.get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, salt, role, email_verified,"
            " created_at, updated_at) VALUES (?,?,?,?,0,?,?)",
            (email, pwd_hash, salt, role, now, now))
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise EmailTakenError("registration failed") from None
    token = security.new_token()
    conn.execute(
        "INSERT INTO email_verifications (user_id, token_hash, created_at,"
        " expires_at, used) VALUES (?,?,?,?,0)",
        (user_id, security.hash_token(token), now, _iso_future(VERIFY_TTL_S)))
    conn.commit()
    return {"user": _user_row(user_id), "verification_token": token}


def get_user_by_email(email: str) -> Optional[dict]:
    email = validate_email(email)
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return _user_row(row["id"]) if row else None


def get_user(user_id: int) -> Optional[dict]:
    return _user_row(user_id)


def _user_row(user_id: int) -> Optional[dict]:
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "role": row["role"],
        "email_verified": bool(row["email_verified"]),
        "created_at": row["created_at"],
    }


# ---- email verification ----------------------------------------------------

def verify_email(token: str) -> dict:
    """Consume a verification token. Single-use, expiry-checked, stored as a
    digest. Raises VerificationError on any failure (generic message)."""
    if not token or len(token) > 256:
        raise VerificationError("invalid or expired verification token")
    digest = security.hash_token(token)
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM email_verifications WHERE token_hash=?", (digest,)
    ).fetchone()
    if row is None:
        raise VerificationError("invalid or expired verification token")
    if row["used"]:
        raise VerificationError("invalid or expired verification token")
    if row["expires_at"] < db.utcnow_iso():
        conn.execute("DELETE FROM email_verifications WHERE id=?", (row["id"],))
        conn.commit()
        raise VerificationError("invalid or expired verification token")
    # single-use: mark consumed atomically, then flip the user to verified
    conn.execute("UPDATE email_verifications SET used=1 WHERE id=?", (row["id"],))
    conn.execute("UPDATE users SET email_verified=1, updated_at=? WHERE id=?",
                 (db.utcnow_iso(), row["user_id"]))
    conn.commit()
    user = _user_row(row["user_id"])
    if user is None:
        raise VerificationError("invalid or expired verification token")
    return user


def resend_verification(email: str) -> Optional[str]:
    """Issue a fresh verification token for an existing UNVERIFIED account.
    Returns the raw token, or None when the email is unknown/verified (the
    caller must respond generically either way). Old pending tokens are
    invalidated so an attacker cannot accumulate valid tokens."""
    email = validate_email(email)
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if row is None or row["email_verified"]:
        return None
    conn.execute("DELETE FROM email_verifications WHERE user_id=?", (row["id"],))
    token = security.new_token()
    now = db.utcnow_iso()
    conn.execute(
        "INSERT INTO email_verifications (user_id, token_hash, created_at,"
        " expires_at, used) VALUES (?,?,?,?,0)",
        (row["id"], security.hash_token(token), now, _iso_future(VERIFY_TTL_S)))
    conn.commit()
    return token


# ---- sessions --------------------------------------------------------------

def create_session(user_id: int) -> str:
    token = security.new_token()
    now = db.utcnow_iso()
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO sessions (user_id, token_hash, created_at, expires_at)"
        " VALUES (?,?,?,?)",
        (user_id, security.hash_token(token), now, _iso_future(SESSION_TTL_S)))
    conn.commit()
    return token


def get_user_by_session(token: str) -> Optional[dict]:
    if not token or len(token) > 256:
        return None
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE token_hash=?", (security.hash_token(token),)
    ).fetchone()
    if row is None:
        return None
    if row["expires_at"] < db.utcnow_iso():
        conn.execute("DELETE FROM sessions WHERE id=?", (row["id"],))
        conn.commit()
        return None
    user = _user_row(row["user_id"])
    if user is None:
        conn.execute("DELETE FROM sessions WHERE id=?", (row["id"],))
        conn.commit()
        return None
    return user


def destroy_session(token: str) -> None:
    if not token or len(token) > 256:
        return
    conn = db.get_conn()
    conn.execute("DELETE FROM sessions WHERE token_hash=?",
                 (security.hash_token(token),))
    conn.commit()


# ---- password reset --------------------------------------------------------

def request_password_reset(email: str) -> Optional[str]:
    """Create a single-use, short-lived reset token for an existing account.
    Returns the raw token or None when the email is unknown — the caller
    replies identically in both cases (no user enumeration)."""
    email = validate_email(email)
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if row is None:
        return None
    conn.execute("DELETE FROM password_resets WHERE user_id=?", (row["id"],))
    token = security.new_token()
    now = db.utcnow_iso()
    conn.execute(
        "INSERT INTO password_resets (user_id, token_hash, created_at,"
        " expires_at, used) VALUES (?,?,?,?,0)",
        (row["id"], security.hash_token(token), now, _iso_future(RESET_TTL_S)))
    conn.commit()
    return token


def reset_password(token: str, new_password: str) -> dict:
    """Consume a reset token and set a new password (single-use, expired
    tokens rejected). Also revokes every session so a stolen token cannot
    keep an attacker logged in after the password changes."""
    validate_password(new_password)
    if not token or len(token) > 256:
        raise ResetError("invalid or expired reset token")
    conn = db.get_conn()
    row = conn.execute(
        "SELECT * FROM password_resets WHERE token_hash=?",
        (security.hash_token(token),)).fetchone()
    if row is None:
        raise ResetError("invalid or expired reset token")
    if row["used"]:
        raise ResetError("invalid or expired reset token")
    if row["expires_at"] < db.utcnow_iso():
        conn.execute("DELETE FROM password_resets WHERE id=?", (row["id"],))
        conn.commit()
        raise ResetError("invalid or expired reset token")
    pwd_hash, salt = security.hash_password(new_password)
    now = db.utcnow_iso()
    conn.execute("UPDATE password_resets SET used=1 WHERE id=?", (row["id"],))
    conn.execute("UPDATE users SET password_hash=?, salt=?, updated_at=? WHERE id=?",
                 (pwd_hash, salt, now, row["user_id"]))
    conn.execute("DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
    conn.commit()
    user = _user_row(row["user_id"])
    if user is None:
        raise ResetError("invalid or expired reset token")
    return user


# ---- admin bootstrap -------------------------------------------------------

def bootstrap_admin() -> None:
    """Create the bootstrap admin from MISSIONMIND_ADMIN_EMAIL /
    MISSIONMIND_ADMIN_PASSWORD when both are set and the account does not
    exist. No-op otherwise. The account is created email-verified."""
    email = os.getenv("MISSIONMIND_ADMIN_EMAIL", "").strip().lower()
    password = os.getenv("MISSIONMIND_ADMIN_PASSWORD", "")
    if not email or not password:
        return
    if get_user_by_email(email) is not None:
        return
    try:
        created = create_user(email, password, role="admin")
    except AuthError:
        return
    # verify immediately (bootstrap admin needs no email round-trip)
    try:
        verify_email(created["verification_token"])
    except VerificationError:
        pass


# ---- helpers ---------------------------------------------------------------

def _iso_future(seconds: int) -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(
        timespec="seconds")

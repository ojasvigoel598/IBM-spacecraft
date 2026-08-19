"""MissionMind auth — cryptographic primitives.

Zero new dependencies: password hashing uses PBKDF2-HMAC-SHA256 from the
stdlib (OWASP-recommended iteration count, per-user random salt, constant-time
comparison), and session/verification/reset tokens are high-entropy random
strings whose SHA-256 digests are what actually get stored — a database leak
never exposes a usable token.

Iterations are configurable via MISSIONMIND_PBKDF2_ITERATIONS (default 310000,
the OWASP 2023 recommendation for PBKDF2-HMAC-SHA256). Never lower it in
production; tests may lower it only to keep the suite fast.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

_ITERATIONS = int(os.getenv("MISSIONMIND_PBKDF2_ITERATIONS", "310000"))
_ITERATIONS = max(1000, min(_ITERATIONS, 10_000_000))  # sanity clamp

_SALT_BYTES = 16
_TOKEN_BYTES = 32


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (password_hash_hex, salt_hex) using PBKDF2-HMAC-SHA256.

    A fresh random salt is generated when none is supplied. The password is
    UTF-8 encoded; a length cap is enforced here as a final defence even
    though the API layer validates earlier.
    """
    if password is None or len(password) > 1024:
        raise ValueError("password must be a string of at most 1024 characters")
    salt_bytes = bytes.fromhex(salt) if salt else secrets.token_bytes(_SALT_BYTES)
    if len(salt_bytes) != _SALT_BYTES:
        raise ValueError("invalid salt length")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 salt_bytes, _ITERATIONS)
    return digest.hex(), salt_bytes.hex()


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Constant-time password verification. Returns False on any mismatch or
    malformed input — never raises for bad inputs."""
    try:
        candidate, _ = hash_password(password, salt)
    except Exception:  # noqa: BLE001 - verification must degrade to False
        return False
    return hmac.compare_digest(candidate, password_hash)


def new_token() -> str:
    """URL-safe random token (32 bytes of entropy). This is what the client
    receives; only its SHA-256 digest is stored."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def hash_token(token: str) -> str:
    """SHA-256 digest of a token for at-rest storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)

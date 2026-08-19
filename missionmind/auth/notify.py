"""MissionMind auth — token delivery (email).

In production, verification and password-reset tokens MUST be delivered by
email — the API never returns them to the client. This module sends them via
SMTP using only the stdlib (`smtplib`, `email.message`), configured with:

    MISSIONMIND_SMTP_HOST        (required in production)
    MISSIONMIND_SMTP_PORT        (default 587)
    MISSIONMIND_SMTP_USERNAME    (optional; relay auth)
    MISSIONMIND_SMTP_PASSWORD    (optional; relay auth)
    MISSIONMIND_SMTP_FROM        (default: the username, or noreply@<host>)
    MISSIONMIND_SMTP_USE_TLS     (default 1)
    MISSIONMIND_PUBLIC_URL       (base URL used to build clickable links,
                                  e.g. https://missionmind.vercel.app)

`send_secret` never raises to the caller and never logs the token — delivery
failures are logged with the recipient address only.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

log = logging.getLogger("missionmind.auth.notify")

# Links point at the SPA root with a token query param; the console's auth
# screen reads ?vt= (verify) / ?rt= (reset) and prefills the flow.
VERIFY_QUERY = "vt"
RESET_QUERY = "rt"


def smtp_configured() -> bool:
    return bool(os.getenv("MISSIONMIND_SMTP_HOST", "").strip())


def _from_addr() -> str:
    username = os.getenv("MISSIONMIND_SMTP_USERNAME", "").strip()
    explicit = os.getenv("MISSIONMIND_SMTP_FROM", "").strip()
    if explicit:
        return explicit
    if username and "@" in username:
        return username
    host = os.getenv("MISSIONMIND_SMTP_HOST", "localhost")
    return f"noreply@{host}"


def _subject_and_body(kind: str, token: str, public_url: str) -> tuple[str, str]:
    if kind == "verify":
        subject = "MissionMind — verify your email"
        query = f"{VERIFY_QUERY}={token}"
        action = "verify your email address"
    else:
        subject = "MissionMind — reset your password"
        query = f"{RESET_QUERY}={token}"
        action = "reset your password"
    link = f"{public_url.rstrip('/')}/?{query}"
    body = (
        f"MissionMind account security\n\n"
        f"Use the link below to {action}:\n{link}\n\n"
        f"Or paste this one-time token into the console:\n{token}\n\n"
        f"This token expires and can only be used once. If you did not "
        f"request this, ignore this email."
    )
    return subject, body


def send_secret(kind: str, email: str, token: str) -> bool:
    """Send a verification ('verify') or reset ('reset') token by email.
    Returns True on success. Never raises to the caller; failures are logged
    (recipient only — the token is never logged)."""
    if not smtp_configured():
        log.error(
            "cannot deliver %s token for %s: MISSIONMIND_SMTP_HOST not "
            "configured (production mode must not return tokens to clients)",
            kind, email)
        return False
    public_url = os.getenv("MISSIONMIND_PUBLIC_URL", "").strip()
    if not public_url:
        log.error(
            "cannot deliver %s token for %s: MISSIONMIND_PUBLIC_URL not "
            "configured (absolute link required in production)", kind, email)
        return False
    try:
        subject, body = _subject_and_body(kind, token, public_url)
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = _from_addr()
        msg["To"] = email
        msg.set_content(body)
        port = int(os.getenv("MISSIONMIND_SMTP_PORT", "587"))
        use_tls = os.getenv("MISSIONMIND_SMTP_USE_TLS", "1").strip() != "0"
        with smtplib.SMTP(os.getenv("MISSIONMIND_SMTP_HOST"), port, timeout=20) as smtp:
            smtp.ehlo()
            if use_tls:
                smtp.starttls()
                smtp.ehlo()
            username = os.getenv("MISSIONMIND_SMTP_USERNAME", "").strip()
            password = os.getenv("MISSIONMIND_SMTP_PASSWORD", "").strip()
            if username:
                smtp.login(username, password)
            smtp.send_message(msg)
        log.info("delivered %s token for %s via SMTP", kind, email)
        return True
    except Exception as e:  # noqa: BLE001 - delivery must never crash auth
        log.error("failed to deliver %s token for %s via SMTP: %s",
                  kind, email, type(e).__name__)
        return False

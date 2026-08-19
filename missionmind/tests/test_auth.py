"""
MissionMind security regression suite.

Covers the 22 scenarios a hostile client could throw at the FastAPI backend:
authn/authz boundaries, session handling, token lifecycle, rate limiting,
payload/input validation, error hygiene, secret handling, CORS, admin gates
and frontend credential exposure. Every test drives the REAL HTTP surface
(no mocks) so a bypass in any middleware/dependency layer fails loudly.

Design notes:
  - conftest points the auth DB at a throwaway file and raises only the
    cross-cutting rate caps; the per-email login cap and per-user API caps
    stay at real values and are exercised here with deterministic counts.
  - the rate limiter is reset after every test (autouse fixture), so burst
    tests cannot poison later tests.
"""

import os
import uuid

from fastapi.testclient import TestClient

from missionmind.viz.api_server import app

PASSWORD = "testpassword1"


def _signup(client, email):
    r = client.post("/api/auth/signup",
                    json={"email": email, "password": PASSWORD})
    assert r.status_code == 201, r.text
    return r.json()["verification_token"]


def _verify(client, token):
    r = client.post("/api/auth/verify", json={"token": token})
    assert r.status_code == 200, r.text


def _new_client():
    """A fresh TestClient (isolated cookie jar) against the real app."""
    return TestClient(app)


def _unique_email(tag):
    return f"{tag}-{uuid.uuid4().hex[:10]}@missionmind.test"


# ---- 1. unauthenticated API -> 401 -----------------------------------------

def test_unauthenticated_api_is_rejected():
    c = _new_client()
    for path in ("/api/scenario/none?t0=0&t1=10",
                 "/api/summary/none?t=0",
                 "/api/alert/none?t=0",
                 "/api/live/next?mode=none&n=1",
                 "/api/trace",
                 "/api/models",
                 "/api/admin/status"):
        r = c.get(path)
        assert r.status_code == 401, f"{path}: expected 401, got {r.status_code}"
        assert "authentication required" in r.text


# ---- 2. authenticated but unverified -> 403 --------------------------------

def test_unverified_user_cannot_access_protected_endpoints():
    c = _new_client()
    email = _unique_email("uv")
    _signup(c, email)
    r = c.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    r = c.get("/api/summary/none?t=0")
    assert r.status_code == 403, f"expected 403, got {r.status_code}"
    assert "verification" in r.text
    # public endpoints stay reachable for an unverified account
    assert c.get("/api/health").status_code == 200


# ---- 3. verified user -> allowed -------------------------------------------

def test_verified_user_can_access_protected_endpoints(authed_client):
    for path in ("/api/scenario/none?t0=0&t1=10",
                 "/api/summary/none?t=0",
                 "/api/alert/none?t=0",
                 "/api/live/next?mode=none&n=1",
                 "/api/trace",
                 "/api/models"):
        r = authed_client.get(path)
        assert r.status_code == 200, f"{path}: expected 200, got {r.status_code}"


# ---- 4. normal user -> admin endpoint denied -------------------------------

def test_normal_user_is_denied_admin_endpoint(authed_client):
    r = authed_client.get("/api/admin/status")
    assert r.status_code == 403, f"expected 403, got {r.status_code}"


# ---- 5. user A cannot access user B's state --------------------------------

def test_live_stream_state_is_isolated_per_user():
    """The live edge-node stream is per-user: advancing A's node must not
    advance (or be visible from) B's node."""
    a, b = _new_client(), _new_client()
    for c, tag in ((a, "isoA"), (b, "isoB")):
        email = _unique_email(tag)
        _verify(c, _signup(c, email))
        assert c.post("/api/auth/login",
                      json={"email": email, "password": PASSWORD}).status_code == 200

    r = a.get("/api/live/next?mode=solar_degradation&n=10")
    assert r.status_code == 200 and r.json()["total"] == 10
    # B starts its own stream from zero
    r = b.get("/api/live/next?mode=solar_degradation&n=10")
    assert r.status_code == 200 and r.json()["total"] == 10, \
        "user B must not inherit user A's live stream"
    # A's stream is unchanged by B's activity
    r = a.get("/api/live/next?mode=solar_degradation&n=5")
    assert r.json()["total"] == 15, "user A's stream must advance only on A's calls"


# ---- 6. expired session -> denied ------------------------------------------

def test_expired_session_is_rejected():
    from missionmind.auth import db
    c = _new_client()
    email = _unique_email("exp")
    _verify(c, _signup(c, email))
    assert c.post("/api/auth/login",
                  json={"email": email, "password": PASSWORD}).status_code == 200
    conn = db.get_conn()
    conn.execute("UPDATE sessions SET expires_at='2000-01-01T00:00:00+00:00'")
    conn.commit()
    r = c.get("/api/auth/me")
    assert r.status_code == 401, f"expired session must be rejected, got {r.status_code}"


# ---- 7. invalid / forged session -> denied ---------------------------------

def test_forged_session_cookie_is_rejected():
    c = _new_client()
    c.cookies.set("missionmind_session", "forged-token-value")
    r = c.get("/api/auth/me")
    assert r.status_code == 401, "a client-chosen cookie value must never authenticate"


# ---- 8. logout invalidates the session -------------------------------------

def test_logout_invalidates_session(authed_client):
    assert authed_client.get("/api/auth/me").status_code == 200
    r = authed_client.post("/api/auth/logout")
    assert r.status_code == 200
    assert authed_client.get("/api/auth/me").status_code == 401, \
        "session must be dead after logout"


# ---- 9. verification token replay -> denied --------------------------------

def test_verification_token_is_single_use():
    c = _new_client()
    token = _signup(c, _unique_email("vrep"))
    _verify(c, token)
    r = c.post("/api/auth/verify", json={"token": token})
    assert r.status_code == 400, "token replay must be rejected"


# ---- 10. expired verification token -> denied ------------------------------

def test_expired_verification_token_is_rejected():
    from missionmind.auth import db
    from missionmind.auth.security import hash_token
    c = _new_client()
    email = _unique_email("vexp")
    token = _signup(c, email)
    conn = db.get_conn()
    conn.execute(
        "UPDATE email_verifications SET expires_at='2000-01-01T00:00:00+00:00'"
        " WHERE token_hash=?", (hash_token(token),))
    conn.commit()
    r = c.post("/api/auth/verify", json={"token": token})
    assert r.status_code == 400, "expired token must be rejected"
    # the account must still be unverified -> protected endpoints still 403
    assert c.post("/api/auth/login",
                  json={"email": email, "password": PASSWORD}).status_code == 200
    assert c.get("/api/summary/none?t=0").status_code == 403


# ---- 11. password reset token replay -> denied -----------------------------

def test_reset_token_is_single_use_and_revokes_sessions():
    c = _new_client()
    email = _unique_email("rrep")
    _verify(c, _signup(c, email))
    assert c.post("/api/auth/login",
                  json={"email": email, "password": PASSWORD}).status_code == 200
    r = c.post("/api/auth/reset", json={"email": email})
    assert r.status_code == 200
    reset_token = r.json().get("reset_token")
    assert reset_token, "dev mode must return the reset token"
    r = c.post("/api/auth/reset/confirm",
               json={"token": reset_token, "password": "newpassword9"})
    assert r.status_code == 200, r.text
    # replay must fail
    r = c.post("/api/auth/reset/confirm",
               json={"token": reset_token, "password": "anotherpass9"})
    assert r.status_code == 400, "reset token replay must be rejected"
    # all old sessions are revoked
    assert c.get("/api/auth/me").status_code == 401, \
        "sessions must be revoked after a password reset"


# ---- 12. brute-force login -> rate limited ---------------------------------

def test_brute_force_login_is_rate_limited():
    c = _new_client()
    email = _unique_email("bf")
    _verify(c, _signup(c, email))
    for _ in range(10):
        c.post("/api/auth/login",
               json={"email": email, "password": "wrongpassword9"})
    last = c.post("/api/auth/login",
                  json={"email": email, "password": "wrongpassword9"})
    assert last.status_code == 429, f"expected 429, got {last.status_code}"
    assert "Retry-After" in last.headers
    # a different email (spray) is also capped per IP
    for _ in range(31):
        c.post("/api/auth/login",
               json={"email": _unique_email("spray"), "password": "wrongpassword9"})
    r = c.post("/api/auth/login",
               json={"email": _unique_email("spray"), "password": "wrongpassword9"})
    assert r.status_code == 429, f"per-IP spray cap failed: {r.status_code}"


# ---- 13. excessive API requests -> rate limited ----------------------------

def test_excessive_api_requests_are_rate_limited(authed_client, monkeypatch):
    monkeypatch.setenv("MISSIONMIND_API_LIMIT", "5")
    for _ in range(5):
        assert authed_client.get("/api/scenario/none?t0=0&t1=0").status_code == 200
    r = authed_client.get("/api/scenario/none?t0=0&t1=0")
    assert r.status_code == 429, f"expected 429, got {r.status_code}"
    assert "Retry-After" in r.headers


# ---- 14. oversized payload -> 413 ------------------------------------------

def test_oversized_payload_is_rejected(authed_client):
    big = "x" * (16 * 1024 + 1)
    r = authed_client.post("/api/auth/login",
                           json={"email": "a@b.com", "password": big})
    assert r.status_code == 413, f"expected 413, got {r.status_code}"


# ---- 15. invalid parameters -> rejected ------------------------------------

def test_invalid_parameters_are_rejected(authed_client):
    # unknown scenario mode -> 404 (never a 500)
    assert authed_client.get("/api/alert/nope?t=0").status_code == 404
    assert authed_client.get("/api/summary/nope?t=0").status_code == 404
    # extra (unknown) body fields -> 422
    r = authed_client.post("/api/auth/login", json={
        "email": "a@b.com", "password": PASSWORD, "admin": True})
    assert r.status_code == 422, r.text
    # wrong types -> 422
    r = authed_client.post("/api/auth/login", json={"email": 42, "password": 42})
    assert r.status_code == 422, r.text
    # weak password -> 422
    r = authed_client.post("/api/auth/signup",
                           json={"email": _unique_email("weak"),
                                 "password": "short"})
    assert r.status_code == 422, r.text
    # absurd live-stream size is clamped, not crashed
    r = authed_client.get("/api/live/next?mode=none&n=999999999")
    assert r.status_code == 200, r.text[:200]
    # out-of-range mission time is clamped, not crashed
    r = authed_client.get("/api/alert/none?t=-5")
    assert r.status_code == 200, r.text[:200]


# ---- 16. malicious input -> safely rejected ---------------------------------

def test_sql_and_command_injection_are_rejected(authed_client):
    for evil in ("a' OR '1'='1", "a@b.com; rm -rf /", "' UNION SELECT * FROM users --",
                 "<script>alert(1)</script>", "a@b.com\x00"):
        r = authed_client.post("/api/auth/login",
                               json={"email": evil, "password": PASSWORD})
        # rejected cleanly (401/422) - never 500, never a stack trace
        assert r.status_code in (401, 422), f"{evil!r}: got {r.status_code}: {r.text[:120]}"
        assert "Traceback" not in r.text and "sqlite3" not in r.text.lower()
    r = authed_client.post("/api/auth/login", json={
        "email": _unique_email("inj"), "password": "x' OR 1=1 --"})
    assert r.status_code in (401, 422), r.text[:120]


# ---- 17. CORS from unauthorized origin -> rejected -------------------------

def test_cors_rejects_untrusted_origin(api_client):
    r = api_client.get("/api/health", headers={"Origin": "https://evil.attacker.io"})
    assert "access-control-allow-origin" not in r.headers
    r = api_client.options("/api/auth/login", headers={
        "Origin": "https://evil.attacker.io",
        "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in r.headers


# ---- 18. direct API access without the frontend is still protected ---------

def test_direct_curl_style_api_calls_are_protected():
    """A curl/Postman/script client (no browser, no CORS) gets the same
    authn/authz treatment as the browser — CORS is never a security boundary."""
    import http.client

    # raw HTTP/1.1 against the ASGI app via a tiny in-process server is overkill;
    # a bare TestClient WITHOUT any cookie is exactly the curl-equivalent path
    c = TestClient(app)
    r = c.get("/api/summary/none?t=0")
    assert r.status_code == 401
    # even a POST to an auth endpoint with no body is handled cleanly
    r = c.post("/api/auth/login", content=b"", headers={"content-type": "application/json"})
    assert r.status_code in (413, 422, 401), r.status_code
    # header-only probing never leaks internals
    assert "Traceback" not in r.text
    del http.client  # (import used to document the threat model; not needed at runtime)


# ---- 19. exceptions -> no stack traces / internals to the client -----------

def test_errors_never_leak_stack_traces(api_client):
    from missionmind.viz import api_server as api_mod

    @api_mod.app.get("/api/_test_boom", include_in_schema=False)
    def _boom():
        raise RuntimeError("TOP-SECRET internal path C:\\secret\\file.py")

    # raise_server_exceptions=False mirrors what a real HTTP client sees:
    # the server answers 500 with the safe body instead of re-raising.
    raw = TestClient(api_mod.app, raise_server_exceptions=False)
    try:
        r = raw.get("/api/_test_boom")
        assert r.status_code == 500
        body = r.text
        assert "Something went wrong" in body
        assert "TOP-SECRET" not in body
        assert "Traceback" not in body
        assert "\\secret\\" not in body and "/secret/" not in body
        assert "api_server.py" not in body
    finally:
        api_mod.app.router.routes = [
            rt for rt in api_mod.app.router.routes
            if getattr(rt, "path", None) != "/api/_test_boom"]


# ---- 20. secrets never appear in responses ---------------------------------

def test_secrets_never_appear_in_any_response(api_client, monkeypatch):
    monkeypatch.setenv("WATSONX_APIKEY", "REAL-LOOKING-SECRET-KEY-12345")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "secret-project-id")
    monkeypatch.setenv("MISSIONMIND_ADMIN_PASSWORD", "admin-secret-password")
    for path in ("/api/health", "/api/scenarios"):
        r = api_client.get(path)
        assert "REAL-LOOKING-SECRET-KEY-12345" not in r.text
        assert "secret-project-id" not in r.text
        assert "admin-secret-password" not in r.text
    # fresh client (no session cookie) - unauthenticated response too
    r = _new_client().get("/api/auth/me")
    assert r.status_code == 401
    assert "REAL-LOOKING-SECRET-KEY-12345" not in r.text


# ---- 21. frontend bundle contains no backend credentials -------------------

def test_frontend_has_no_backend_credentials():
    """No watsonx/IBM secret reference may exist in the web source (the API key
    is server-side only). Also scan the built bundle when present."""
    web_src = os.path.join(os.path.dirname(__file__), "..", "..", "web", "src")
    for root, _dirs, files in os.walk(web_src):
        for f in files:
            if not f.endswith((".ts", ".tsx", ".js", ".jsx", ".html")):
                continue
            with open(os.path.join(root, f), encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            assert "WATSONX_APIKEY" not in text, f"secret ref in {f}"
            assert "WATSONX_API_KEY" not in text, f"secret ref in {f}"
    dist = os.path.join(os.path.dirname(__file__), "..", "..", "web", "dist")
    if os.path.isdir(dist):
        for root, _dirs, files in os.walk(dist):
            for f in files:
                if not f.endswith((".js", ".html")):
                    continue
                with open(os.path.join(root, f), encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                assert "WATSONX_APIKEY" not in text, f"secret in built bundle {f}"


# ---- 22. admin / debug endpoints are locked down ---------------------------

def test_admin_endpoint_locked_down(authed_client):
    assert _new_client().get("/api/admin/status").status_code == 401
    assert authed_client.get("/api/admin/status").status_code == 403


def test_admin_bootstrap_via_env_grants_admin_access(monkeypatch):
    """An admin bootstrapped from environment variables can reach the admin
    endpoint; the account is email-verified automatically."""
    from missionmind.auth import service as auth_service
    email = _unique_email("admin")
    monkeypatch.setenv("MISSIONMIND_ADMIN_EMAIL", email)
    monkeypatch.setenv("MISSIONMIND_ADMIN_PASSWORD", "adminpassword9")
    auth_service.bootstrap_admin()
    c = _new_client()
    r = c.post("/api/auth/login", json={"email": email, "password": "adminpassword9"})
    assert r.status_code == 200, r.text
    r = c.get("/api/admin/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "users" in body and "active_sessions" in body
    assert "secret" not in r.text.lower() or "password" not in r.text.lower()


# ---- extra: enumeration safety + session/cookie hygiene --------------------

def test_no_account_enumeration(api_client):
    """Signup/login/reset must answer identically whether the account exists."""
    email = _unique_email("enum")
    # signup twice with the same email -> identical message
    r1 = api_client.post("/api/auth/signup",
                         json={"email": email, "password": PASSWORD})
    r2 = api_client.post("/api/auth/signup",
                         json={"email": email, "password": PASSWORD})
    assert r1.status_code == 201 and r2.status_code == 201
    assert "account created" in r1.json()["message"]
    assert "account created" in r2.json()["message"]
    # login: unknown email and wrong password -> identical detail
    r_unknown = api_client.post("/api/auth/login",
                                json={"email": _unique_email("ghost"),
                                      "password": "wrongpassword9"})
    r_wrong = api_client.post("/api/auth/login",
                              json={"email": email, "password": "wrongpassword9"})
    assert r_unknown.status_code == r_wrong.status_code == 401
    assert r_unknown.json()["detail"] == r_wrong.json()["detail"]
    # reset: registered and unregistered emails -> identical message
    r_reg = api_client.post("/api/auth/reset", json={"email": email})
    r_gone = api_client.post("/api/auth/reset",
                             json={"email": _unique_email("ghost2")})
    assert "if that email is registered" in r_reg.json()["message"]
    assert "if that email is registered" in r_gone.json()["message"]


def test_session_cookie_attributes_are_secure():
    c = _new_client()
    email = _unique_email("cook")
    _verify(c, _signup(c, email))
    r = c.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    set_cookie = r.headers.get("set-cookie", "")
    assert "missionmind_session=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Secure" not in set_cookie  # dev mode: not Secure
    assert "path=/" in set_cookie.lower()


def test_security_headers_present(api_client):
    r = api_client.get("/api/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("referrer-policy") == "no-referrer"
    assert "camera=()" in r.headers.get("permissions-policy", "")
    assert r.headers.get("cache-control") == "no-store"
    assert "default-src 'none'" in r.headers.get("content-security-policy", "")


def test_hsts_only_in_production(api_client, monkeypatch):
    monkeypatch.setenv("MISSIONMIND_ENV", "production")
    r = api_client.get("/api/health")
    assert "strict-transport-security" in r.headers
    monkeypatch.delenv("MISSIONMIND_ENV", raising=False)
    r = api_client.get("/api/health")
    assert "strict-transport-security" not in r.headers

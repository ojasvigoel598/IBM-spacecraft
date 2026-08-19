# MissionMind — Security Architecture & Production Operations

This document describes the security model of the MissionMind FastAPI backend
(`missionmind/viz/api_server.py`), its authentication system
(`missionmind/auth/`), and how to operate it in production. It is a
**decision-support console**, not an autonomous flight-control authority.

---

## 1. Threat model

Every browser/client is treated as hostile:

- The frontend is never trusted: authentication and authorization are decided
  **server-side from the session cookie**, never from request JSON, query
  parameters, headers, or the React UI.
- CORS is not authentication. Direct API calls (curl, Postman, scripts) get
  exactly the same authn/authz treatment as the browser.
- No backend secret (watsonx API key, project ID, admin password, database
  path) ever reaches the client — not in responses, not in the JS bundle,
  not in errors.
- Attacker goals defended against: account takeover (brute force, spraying,
  token replay), enumeration, session theft/fixation, input-triggered 500s,
  SQL/command/path injection, resource exhaustion (rate limits, body caps),
  cross-origin reads, admin escalation, and leaked stack traces.

## 2. Authentication architecture

Zero new dependencies: **SQLite** (stdlib) for storage, **PBKDF2-HMAC-SHA256**
(stdlib) for password hashing, **secrets.token_urlsafe** for tokens.

| Concern | Implementation |
|---|---|
| Password storage | PBKDF2-HMAC-SHA256, 310 000 iterations (OWASP), per-user 16-byte salt, constant-time compare (`hmac.compare_digest`). Configurable via `MISSIONMIND_PBKDF2_ITERATIONS` — never lower in production. |
| Sessions | Opaque 32-byte random token in an **HttpOnly, SameSite=Lax** cookie (`missionmind_session`), `Secure` when `MISSIONMIND_ENV=production`, 7-day expiry. Only the SHA-256 digest is stored. Logout invalidates; password reset revokes **all** sessions. |
| Email verification | One-time token, 24 h expiry, stored as digest, invalidated after use and on resend (cannot be replayed or brute-forced — rate-limited 10/10 min/IP). |
| Password reset | One-time token, 1 h expiry, stored as digest, single-use; on confirm the password changes and every session is revoked. |
| Enumeration | Signup/login/reset/resend answer identically whether an account exists ("invalid email or password"; "if that email is registered…"). |
| Rate limiting | In-memory sliding buckets: signup 5/15 min/IP, login 10/5 min per (IP,email) + 30/5 min per IP, verify 10/10 min/IP, resend 3/15 min per (IP,email), reset 5/15 min, reset-confirm 10/10 min/IP, mission API 240/min per user (60/min for `/api/models` and `/api/live/next`), public endpoints 120/min per IP. All return `429 + Retry-After`. Tunable via `MISSIONMIND_AUTH_*` / `MISSIONMIND_API_*` env vars. |

### Flows

```
Sign up  -> email + password (server-side policy: 8-128 chars, letter + digit)
         -> account created unverified, one-time token issued
         -> dev mode (MISSIONMIND_ENV != production): token returned in the
            response so the demo works without SMTP
         -> production: token is NEVER returned; wire a mailer (see §7)

Verify   -> POST /api/auth/verify {token}  -> account verified (single use)

Log in   -> POST /api/auth/login  -> Set-Cookie (HttpOnly, SameSite=Lax)
Log out  -> POST /api/auth/logout -> session row deleted, cookie cleared
Reset    -> POST /api/auth/reset {email} (generic reply)
         -> POST /api/auth/reset/confirm {token, new password}
```

Admin accounts are bootstrapped from environment variables only
(`MISSIONMIND_ADMIN_EMAIL` / `MISSIONMIND_ADMIN_PASSWORD`), created
email-verified, never from any client input.

## 3. Authorization

| Endpoint | Access |
|---|---|
| `GET /api/health`, `GET /api/scenarios` | Public (rate-limited) |
| `POST /api/auth/*` | Public (rate-limited) |
| `GET /api/scenario/{mode}`, `summary/{mode}`, `alert/{mode}`, `live/next`, `models`, `trace` | Verified user (401 anon, 403 unverified) |
| `GET /api/admin/status` | Admin only |

Roles and verification state come from the database row keyed by the session
cookie. A client cannot claim `role=admin`, `email_verified=true`, or another
user's identity — even a forged cookie value never matches a stored digest.

## 4. Multi-user isolation

- **Live edge-node streams are per-user** (keyed by user id + scenario): user
  A advancing the live stream cannot disturb or observe user B's stream.
- Scenario/alert/summary data is a **deterministic shared cache** (identical
  physics solve for every user) — read-only, same for everyone by design.
- The runtime trace is a process-global execution log (no user data).
- The auth database is the only user-data store; all tokens in it are digests.

**Caveat:** in-memory rate limits and the per-user node state are per-process.
Behind multiple workers they multiply; a Redis-backed limiter is the documented
upgrade path for large deployments (not needed for a hackathon demo).

## 5. Request defence

- **Body cap:** 16 KB on POST/PUT/PATCH → 413.
- **Schemas:** every auth request body is a strict Pydantic model with
  `extra="forbid"` (unknown fields → 422) and type validation.
- **Query bounds:** scenario mode is validated against an allowlist (404
  otherwise); `t`/`t0`/`t1` clamp to the mission range; `n` clamps to
  [1,250]; `limit` clamps to [1,500]; `since` ≥ 0.
- **Error hygiene:** `HTTPException` → structured `{detail}`; validation
  errors → generic 422; any other exception → logged server-side with a full
  traceback, client receives `500 {"detail":"Something went wrong. Please try
  again."}` — no stack traces, filesystem paths, or env values.
- **Security headers (all responses):** `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
  `Permissions-Policy: camera=(), microphone=(), geolocation=()`,
  `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`,
  `Cache-Control: no-store`, and HSTS (1 year, includeSubDomains) when
  `MISSIONMIND_ENV=production`.
- **CORS:** strict allowlist (never `*`), default local-dev origins, deploy
  origin via `MISSIONMIND_ALLOWED_ORIGINS`. Methods limited to GET/POST/
  OPTIONS (telemetry reads + auth).

## 6. RAG / Granite security

- IBM credentials live **only** in the environment; the frontend never
  receives them. `/api/health` reports booleans (`api_key_present`) and a
  state machine (`MOCK` / `REAL_READY` / `REAL_FAILED`) — never the key.
- Real-call mode is strict: a configured-but-failing IBM call is surfaced as
  `REAL_FAILED` and never silently replaced by the mock pretending to be IBM;
  the deterministic mock is always tagged `source="mock"`.
- Retrieved documents are framed as DATA in the system prompt; prompt
  injection through telemetry/RAG content is covered by the adversarial test
  suite (`missionmind/tests/test_rag_adversarial.py`).
- RAG citations are repo-relative markdown paths validated to exist; the UI
  never receives hallucinated source paths.

## 7. Production checklist (deployment)

1. `MISSIONMIND_ENV=production` — makes cookies `Secure`, enables HSTS, and
   stops returning verification/reset tokens. The app **refuses to start** in
   production unless items 2 and 3 are configured (`check_production_config`
   in `missionmind/auth/api.py` raises with a clear message).
2. **Email delivery** — `MISSIONMIND_SMTP_HOST` (+ optional PORT/USERNAME/
   PASSWORD/FROM/TLS) and `MISSIONMIND_PUBLIC_URL`. Tokens are emailed via
   the stdlib SMTP relay in `missionmind/auth/notify.py`; links point at the
   console root with `?vt=` / `?rt=`, which the auth screen prefills. In
   production the token is **never** returned to the client, even if
   delivery fails (the failure is logged server-side).
3. `MISSIONMIND_DB_PATH` → a **persistent** path. On Vercel serverless, the
   filesystem is ephemeral/read-only per invocation — the auth database must
   live on a persistent service (e.g. Neon/PlanetScale/Supabase or a small
   VM) for real multi-user auth; for the hackathon demo, self-host the
   FastAPI backend (documented in the README) or accept ephemeral local auth
   per warm instance. This is the single biggest deployment caveat.
4. `MISSIONMIND_ALLOWED_ORIGINS` → the exact deployed frontend origin(s).
5. `MISSIONMIND_ADMIN_EMAIL` / `MISSIONMIND_ADMIN_PASSWORD` → create the
   admin; keep the password out of git and rotate it.
6. Reverse proxy terminates TLS; keep `Secure` cookies on.
7. Run the security suite before each release:
   `python -m pytest missionmind/tests/test_auth.py -q` and the dependency
   audits (`pip-audit -r requirements.txt`, `npm audit --omit=dev`).
8. `/api/health` reports auth readiness (`auth.mode`, `auth.db` basename,
   `auth.smtp_configured`, `auth.delivery`) — no secrets, no full paths.

## 8. Testing

`missionmind/tests/test_auth.py` (31 tests) drives the **real HTTP surface**
with no mocks:

- authn/authz boundaries (401 anon / 403 unverified / admin gate)
- token lifecycle (single-use verify + reset, expiry, replay)
- session security (forged/expired cookie, logout invalidation, cookie flags)
- per-user isolation of live streams
- abuse (login brute force + per-IP spray 429, API flood 429, oversized
  body 413, invalid/extra params, SQLi/shell-injection inputs)
- traversal/encoded-path rejection, malformed queries, 405s, 60-thread
  concurrent flood (429-only, no 500s)
- hygiene (no stack traces, no secrets in responses, no credential refs in
  the frontend bundle, security headers, HSTS gating)

Run: `python -m pytest missionmind/tests/test_auth.py -q`

## 9. Honest limitations

- In-memory rate limiting is per-process (see §4).
- Email delivery requires an SMTP relay (stdlib client, `missionmind/auth/
  notify.py`); dev mode returns tokens instead — never in production.
- SQLite is single-writer; fine for a demo/team scale, not a
  high-concurrency SaaS (move to a hosted DB per §7.3).
- The Streamlit dashboard (`missionmind/viz/app.py`) runs its own in-process
  pipeline and does not use the FastAPI auth — it is a local-operator tool,
  not the public multi-user surface.

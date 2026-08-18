"""
API-server tests: the FastAPI HTTP surface of missionmind/viz/api_server.py.

Covered (vertical slice 1 - the documented /api/alert endpoint):
1. /api/alert/{mode}?t=  at a fault time returns 200 with a structured alert:
   mode, label, t, active flag, detected_at, severity, telemetry snapshot,
   physics-rule evidence strings, and RAG source citations (path + score).
2. /api/alert with an unknown mode returns 404.
3. /api/alert in a nominal window reports active=0 and no severity escalation.

The endpoint is declared in the module docstring ("physics+ML+RAG alert evidence
at time t") but was not implemented; these tests are the spec for it.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi.testclient import TestClient
from missionmind.viz.api_server import app

client = TestClient(app)


def test_alert_solar_fault_window_returns_evidence():
    """At t=900 in the solar scenario the alert must be ACTIVE with
    physics + RAG evidence attached (this is the operator-facing contract)."""
    r = client.get("/api/alert/solar_degradation?t=900")
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
    body = r.json()

    assert body["mode"] == "solar_degradation"
    assert body["label"] == "Solar Array Degradation"
    assert body["t"] == 900
    assert body["active"] == 1, "solar fault must be active at t=900"
    assert body["severity"] in ("elevated", "critical")
    assert body["detected_at"] <= 900, "detected_at must be <= current t"

    # telemetry snapshot must include the ensemble result
    telemetry = body["telemetry"]
    for key in ("solar_power_w", "battery_soc", "temperature_c", "anomaly_score",
                "anomaly_flag", "anomaly_source"):
        assert key in telemetry, f"telemetry missing {key}"
    assert telemetry["anomaly_flag"] == 1

    # physics evidence: at least one human-readable rule string
    physics = body["physics"]
    assert isinstance(physics, list) and len(physics) >= 1, "physics evidence empty"
    assert isinstance(physics[0], str) and len(physics[0]) > 10

    # RAG evidence: at least one source citation with a path and score
    rag = body["rag"]
    assert isinstance(rag, list) and len(rag) >= 1, "RAG sources empty"
    for src in rag:
        assert "path" in src, "RAG source missing path"
        assert "score" in src, "RAG source missing score"
        # citations must be readable repo-relative paths, not absolute
        # filesystem paths with '..' chains (operator-facing contract)
        assert ".." not in src["path"], f"rag path not repo-relative: {src['path']}"
        assert not os.path.isabs(src["path"]), f"rag path absolute: {src['path']}"
        assert src["path"].endswith(".md"), f"rag path not a markdown doc: {src['path']}"


def test_alert_unknown_mode_returns_404():
    """Unknown scenario ids must not silently produce an alert."""
    r = client.get("/api/alert/not_a_scenario?t=900")
    assert r.status_code == 404, f"expected 404, got {r.status_code}"


def test_alert_nominal_window_is_inactive():
    """Early mission (burn-in) / no-fault window must report active=0."""
    r = client.get("/api/alert/solar_degradation?t=100")
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["active"] == 0, "no fault expected at t=100 in solar scenario"
    assert body["severity"] == "nominal"


def test_alert_active_has_operator_narrative():
    """An active alert must carry the 4-line operator narrative
    (WARN / SUBSYSTEM / EVIDENCE / ACTION) from causal_narrative."""
    r = client.get("/api/alert/solar_degradation?t=900")
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["active"] == 1
    narrative = body.get("narrative", "")
    assert isinstance(narrative, str) and len(narrative) > 30, "narrative missing"
    assert "WARN" in narrative, "narrative must open with the WARN line"
    assert "SUBSYSTEM" in narrative and "EVIDENCE" in narrative \
        and "ACTION" in narrative, "narrative must be the 4-line causal block"


def test_alert_carries_adaptive_decision_block():
    """The alert must include the situation-aware decision-layer verdict:
    a strategy chosen per situation (rule-first for confirmed subsystem
    faults), a fused adaptive score, and human-readable reasoning lines."""
    for mode, expected in (("solar_degradation", "RULE_POWER"),
                           ("radiator_degradation", "RULE_THERMAL")):
        r = client.get(f"/api/alert/{mode}?t=1500")
        assert r.status_code == 200, r.text[:200]
        dec = (r.json().get("decision") or {})
        assert dec.get("strategy"), f"{mode}: decision missing strategy"
        assert dec["strategy"] == expected, \
            f"{mode}: expected {expected}, got {dec['strategy']}"
        assert isinstance(dec.get("reasoning"), list) and dec["reasoning"], \
            f"{mode}: reasoning empty"
        assert all(isinstance(x, str) and x for x in dec["reasoning"])
        assert "adaptive_score" in dec and "adaptive_flag" in dec
        # past the ramp, a physics rule must have fired -> flag
        assert dec["adaptive_flag"] == 1, f"{mode}: fault window should flag"

    # nominal early window: decision present but inactive
    r = client.get("/api/alert/solar_degradation?t=100")
    dec = (r.json().get("decision") or {})
    assert dec.get("strategy") in ("BURN_IN_SUPPRESS", "NOMINAL"), dec.get("strategy")
    assert dec.get("adaptive_flag") == 0


def test_trace_records_live_execution_and_scoring():
    """The runtime trace must capture which code actually executes as
    telemetry flows: edge-node stepping + ML scoring on live/next, and
    physics-rule checks on summary/alert. `since` cursors must advance."""
    # 1) a live batch must leave scoring + edge events in the trace
    r = client.get("/api/live/next?mode=none&n=30")
    assert r.status_code == 200, r.text[:200]
    tr = client.get("/api/trace").json()
    assert "events" in tr and "last_seq" in tr, "trace response missing fields"
    events = tr["events"]
    assert len(events) >= 1, "trace empty after a live batch"
    names = {e.get("module", "") for e in events}
    assert any("detect" in n for n in names), \
        f"expected ML scoring events in trace, got modules={names}"

    # 2) physics-rule checks must appear after a summary call
    client.get("/api/summary/solar_degradation?t=900")
    tr2 = client.get("/api/trace").json()
    names2 = {e.get("module", "") for e in tr2["events"]}
    assert any("physics" in n for n in names2), \
        f"expected physics-rule events in trace, got {names2}"

    # 3) the cursor must advance: events after `since=last_seq` are new
    seq = tr["last_seq"]
    client.get("/api/live/next?mode=none&n=5")
    tr3 = client.get(f"/api/trace?since={seq}").json()
    assert tr3["last_seq"] > seq, "trace cursor did not advance"
    assert all(e["seq"] > seq for e in tr3["events"]), \
        "since cursor must only return newer events"


def test_live_stream_total_advances_but_retained_is_bounded():
    """The live stream must never retain every frame forever: `total`
    (frames ever streamed) advances without bound, while `retained`
    (frames kept for scoring) stays capped at the scoring window.

    Regression for an unbounded _buffers[mode] growth in the live path.
    """
    total_before = None
    retained_before = None
    # stream well past the retention cap in one mode
    for _ in range(3):
        r = client.get("/api/live/next?mode=radiator_degradation&n=250")
        assert r.status_code == 200, r.text[:200]
        body = r.json()
        assert "total" in body, "live response missing total"
        assert "retained" in body, "live response missing retained"
        total_before = body["total"]
        retained_before = body["retained"]

    assert total_before is not None and total_before > 650, \
        f"expected many streamed frames, got total={total_before}"
    assert retained_before is not None and retained_before <= 600, \
        f"retained must be capped at the scoring window, got {retained_before}"
    assert retained_before < total_before, "retained must not equal total once past the cap"


def test_cors_blocks_unknown_origins():
    """A browser from an origin outside the allowlist must NOT receive CORS
    headers (so it cannot read the response). Regression for the old
    allow_origins=["*"] configuration."""
    r = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers, \
        "unknown origin must not be CORS-allowed"


def test_cors_allows_local_dashboard_origin():
    """The Streamlit dashboard origin (localhost:8501) is in the default
    allowlist and must receive an echoed Access-Control-Allow-Origin."""
    r = client.get("/api/health", headers={"Origin": "http://localhost:8501"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_cors_allowlist_is_env_configurable():
    """Deployed origins are configured via MISSIONMIND_ALLOWED_ORIGINS; the
    default must never be the wildcard, and a configured origin must work."""
    import missionmind.viz.api_server as api_mod
    assert api_mod._allowed_origins(), "allowlist must not be empty"
    assert "*" not in api_mod._allowed_origins(), "wildcard CORS is forbidden"

    old = os.environ.get("MISSIONMIND_ALLOWED_ORIGINS")
    try:
        os.environ["MISSIONMIND_ALLOWED_ORIGINS"] = \
            "https://missionmind.vercel.app, http://localhost:3000"
        parsed = api_mod._allowed_origins()
        assert parsed == ["https://missionmind.vercel.app", "http://localhost:3000"], parsed
    finally:
        if old is None:
            os.environ.pop("MISSIONMIND_ALLOWED_ORIGINS", None)
        else:
            os.environ["MISSIONMIND_ALLOWED_ORIGINS"] = old


def test_health_reports_granite_state_without_secrets():
    """/api/health must expose the Granite state machine (MOCK / REAL_READY /
    REAL_FAILED) with booleans only — never the key value itself — and keep
    the legacy watsonx_key field the web frontend reads."""
    r = client.get("/api/health")
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["watsonx_key"] is True or body["watsonx_key"] is False
    g = body["granite"]
    for key in ("mode", "sdk_installed", "api_key_present", "project_id_present",
                "model_id", "url", "ready_for_real_call", "last_real_request"):
        assert key in g, f"granite status missing {key}"
    assert g["mode"] in ("MOCK", "REAL_READY", "REAL_FAILED")
    assert isinstance(g["api_key_present"], bool)
    # the response must never contain an actual key value
    for env in ("WATSONX_APIKEY", "WATSONX_API_KEY"):
        val = os.environ.get(env)
        if val:
            assert val not in r.text, "API key leaked into health response"


def test_cors_methods_are_read_only():
    """The API is GET-only; the CORS middleware must not advertise write
    methods to browsers."""
    r = client.options("/api/health", headers={
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "POST",
    })
    allowed = r.headers.get("access-control-allow-methods", "")
    assert "POST" not in allowed, f"POST must not be CORS-advertised: {allowed}"
    assert "GET" in allowed, f"GET must be CORS-advertised: {allowed}"


if __name__ == "__main__":
    tests = [test_alert_solar_fault_window_returns_evidence,
             test_alert_unknown_mode_returns_404,
             test_alert_nominal_window_is_inactive,
             test_alert_active_has_operator_narrative,
             test_alert_carries_adaptive_decision_block,
             test_trace_records_live_execution_and_scoring,
             test_live_stream_total_advances_but_retained_is_bounded,
             test_cors_blocks_unknown_origins,
             test_cors_allows_local_dashboard_origin,
             test_cors_allowlist_is_env_configurable,
             test_cors_methods_are_read_only,
             test_health_reports_granite_state_without_secrets]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print("\nAll API-server tests PASS")

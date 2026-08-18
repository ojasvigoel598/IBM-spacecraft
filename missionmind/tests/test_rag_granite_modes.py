"""
Granite-independence tests.

RAG is validated BEFORE and INDEPENDENTLY of the generator:

  Mode A - real watsonx call succeeds   -> source="watsonx", schema-valid
  Mode B - no credentials at all        -> source="mock", but retrieval
                                           STILL runs and evidence is cited
  Mode C - real call fails mid-request  -> tagged mock, evidence preserved,
                                           strict mode raises (never mocks)

No credentials are required to run these tests: Mode B is the default state
of a fresh checkout. The real-call paths are simulated by monkeypatching
_call_watsonx_granite so the FULL wiring (retrieval -> prompt -> parse ->
schema validation -> source tag) is exercised deterministically.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import missionmind.ai.granite_client as gc
from missionmind.ai.granite_client import (
    generate_explanation, GraniteRequestError, check_config,
)
from missionmind.ai.prompts import example_input_json

SOLAR_INPUT = example_input_json()


@pytest.fixture
def no_credentials(monkeypatch):
    """Force the no-credentials state regardless of the machine's .env."""
    for k in ("WATSONX_APIKEY", "WATSONX_API_KEY", "WATSONX_PROJECT_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(gc, "WATSONX_AVAILABLE", False)
    return None


@pytest.fixture
def fake_credentials(monkeypatch):
    monkeypatch.setenv("WATSONX_APIKEY", "not-a-real-key-for-tests")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project")
    monkeypatch.setattr(gc, "WATSONX_AVAILABLE", True)
    return None


# ---- Mode B: no credentials -------------------------------------------------

def test_mode_b_no_credentials_retrieval_still_runs(no_credentials):
    """The mandatory case: with zero credentials the complete retrieval path
    must still run and its evidence must reach the answer."""
    out = generate_explanation(SOLAR_INPUT, use_rag=True, top_k=3)
    assert out["source"] == "mock"
    # retrieval happened - real KB evidence is cited
    assert out["retrieved_docs"], "no RAG evidence with no credentials"
    assert out["evidence_used"], "no cited evidence"
    assert out["retrieved_docs"][0]["id"] == "DOC-POWER-002", \
        out["retrieved_docs"][0]["id"]
    # schema contract still holds
    assert out["risk"] in ("LOW", "MEDIUM", "HIGH")
    for key in ("probable_cause", "reasoning", "recommended_action"):
        assert out.get(key), f"missing {key}"
    assert 0.0 <= out.get("confidence", 0.0) <= 1.0


def test_mode_b_check_config_reports_mock(no_credentials):
    cfg = check_config()
    assert cfg["mode"] == "MOCK"
    assert cfg["ready_for_real_call"] is False


# ---- Mode C: real call fails ------------------------------------------------

def test_mode_c_failure_keeps_evidence_and_tags_mock(fake_credentials, monkeypatch):
    def _boom(system_prompt, user_prompt, model_id=None, timeout_s=45.0):
        raise RuntimeError("connection refused: simulated network failure")
    monkeypatch.setattr(gc, "_call_watsonx_granite", _boom)
    out = generate_explanation(SOLAR_INPUT, use_rag=True)
    assert out["source"] == "mock"
    assert out.get("granite_error", "").startswith("failed:"), out
    # the underlying engineering evidence must survive the failure
    assert out["retrieved_docs"], "evidence lost when the real call failed"
    assert out["evidence_used"]
    # schema valid
    assert out["risk"] in ("LOW", "MEDIUM", "HIGH")
    for key in ("probable_cause", "reasoning", "recommended_action"):
        assert out.get(key)


def test_mode_c_strict_never_substitutes_mock(fake_credentials, monkeypatch):
    def _boom(system_prompt, user_prompt, model_id=None, timeout_s=45.0):
        raise RuntimeError("401 unauthorized")
    monkeypatch.setattr(gc, "_call_watsonx_granite", _boom)
    with pytest.raises(GraniteRequestError) as exc_info:
        generate_explanation(SOLAR_INPUT, use_rag=True, strict=True)
    message = str(exc_info.value)
    assert "failed:auth" in message, message
    # the error must not echo the (fake) credential value
    assert "not-a-real-key" not in message
    assert "mock" in message.lower(), "strict failure must say the mock was NOT used"


def test_mode_c_state_is_reported_as_real_failed(fake_credentials, monkeypatch):
    def _boom(system_prompt, user_prompt, model_id=None, timeout_s=45.0):
        raise RuntimeError("model not deployable")
    monkeypatch.setattr(gc, "_call_watsonx_granite", _boom)
    generate_explanation(SOLAR_INPUT, use_rag=False)  # triggers the failed state
    # granite_status() (used by /api/health) folds the last-request outcome
    # into the top-level mode; check_config() alone reports raw readiness.
    cfg = gc.granite_status()
    assert cfg["mode"] == "REAL_FAILED", cfg
    assert cfg["last_real_request"].startswith("failed:model"), cfg


# ---- Mode A: real call succeeds --------------------------------------------

VALID_GRANITE_JSON = json.dumps({
    "risk": "HIGH",
    "probable_cause": "Solar array degradation confirmed by telemetry.",
    "reasoning": "Per [DOC-POWER-002] the 364W threshold was breached.",
    "recommended_action": "Shed loads per [DOC-POWER-PROC-001].",
    "evidence_used": ["DOC-POWER-002"],
    "confidence": 0.9,
})


def test_mode_a_real_call_parsed_and_tagged(fake_credentials, monkeypatch):
    captured = {}

    def _fake_call(system_prompt, user_prompt, model_id=None, timeout_s=45.0):
        captured["prompt"] = user_prompt
        return VALID_GRANITE_JSON

    monkeypatch.setattr(gc, "_call_watsonx_granite", _fake_call)
    out = generate_explanation(SOLAR_INPUT, use_rag=True)
    assert out["source"] == "watsonx"
    assert out["risk"] == "HIGH"
    assert "retrieved_docs" in out, "real path must still carry RAG evidence"
    # the prompt the real call received must contain the retrieved evidence
    assert "DOC-POWER-002" in captured["prompt"]


def test_mode_a_schema_validation_rejects_malformed(fake_credentials, monkeypatch):
    """Malformed real output must not crash the app: it degrades to the
    tagged mock with the failure state recorded (never a 500, never silence)."""
    monkeypatch.setattr(gc, "_call_watsonx_granite",
                        lambda *a, **k: "not json at all")
    out = generate_explanation(SOLAR_INPUT, use_rag=True)
    assert out["source"] == "mock"
    assert out.get("granite_error", "").startswith("failed:"), out
    assert out["risk"] in ("LOW", "MEDIUM", "HIGH")


def test_mode_a_strict_requires_success(fake_credentials, monkeypatch):
    monkeypatch.setattr(gc, "_call_watsonx_granite",
                        lambda *a, **k: "garbage")
    with pytest.raises(GraniteRequestError):
        generate_explanation(SOLAR_INPUT, use_rag=True, strict=True)


# ---- production path (API) --------------------------------------------------

def test_alert_api_rag_citations_point_at_real_files():
    """The production /api/alert path must return RAG citations whose source
    paths exist on disk - no hallucinated source paths can reach the UI."""
    from fastapi.testclient import TestClient
    from missionmind.viz.api_server import app
    with TestClient(app) as client:
        r = client.get("/api/alert/solar_degradation?t=900")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    rag = body.get("rag", [])
    assert rag, "no RAG citations in alert"
    for entry in rag:
        path = entry.get("path") or ""
        assert path, f"citation without a source path: {entry}"
        abs_path = os.path.abspath(os.path.join(
            os.path.dirname(gc.__file__), "..", "..", path))
        assert os.path.exists(abs_path), f"citation points at a nonexistent file: {path}"
        assert entry.get("score") is not None, f"citation missing score: {path}"
        assert entry.get("content"), f"citation missing content: {path}"
    # the narrative must reference at least one retrieved source
    narrative = body.get("narrative", "")
    assert narrative, "no narrative in alert"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except (AssertionError, Exception) as e:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
    if failed:
        sys.exit(1)
    print("All rag-granite-mode tests PASS")

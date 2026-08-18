"""
MissionMind - Granite Client via watsonx.ai
Spec Section 8 - Must fetch current SDK docs, but also provide robust fallback.

Design:
- Tries to use ibm-watsonx-ai SDK if credentials present (WATSONX_APIKEY, WATSONX_PROJECT_ID)
- Otherwise uses deterministic rule-based mock that produces valid JSON matching schema
- RAG-enhanced path uses retrieved evidence

This ensures demo works fully offline for judges without IBM cloud account,
while still showing real integration pattern (code ready for production).
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

# Load the project-root .env so a key dropped into .env is picked up without
# exporting it in the shell. Existing environment variables always win.
# python-dotenv is optional: the code reads os.environ directly, so a missing
# dotenv just means the user must export the vars themselves.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

from .prompts import SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_RAG, build_user_prompt, build_rag_user_prompt
from .rag import get_retriever

# Try import watsonx SDK
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    WATSONX_AVAILABLE = True
except Exception as e:
    WATSONX_AVAILABLE = False
    # print(f"watsonx SDK not available: {e}")

# Current IBM watsonx.ai deploy-on-demand Granite model. The Granite 3.0/3.2-era
# instruct models (e.g. ibm/granite-3-2b-instruct) have been withdrawn from the
# multitenant catalogue (IBM deprecation notices, 2026); granite-4-h-small is
# IBM's current small hybrid instruct model and the one IBM's own docs sample
# with ModelInference. Override with WATSONX_MODEL_ID when a different model or
# region is required — the app always reports which model it is using.
GRANITE_DEFAULT_MODEL = "ibm/granite-4-h-small"

# Outcome of the most recent real watsonx attempt, surfaced by /api/health and
# check_config so a judge can distinguish MOCK / READY / REQUEST-FAILED without
# any credential being exposed. Values: "not_attempted" | "succeeded" |
# "failed:auth" | "failed:model" | "failed:timeout" | "failed:network" |
# "failed:unknown".
_last_real_state: str = "not_attempted"


class GraniteRequestError(RuntimeError):
    """A real watsonx Granite request was attempted and failed. Raised only by
    the strict path (credentialed smoke test); the demo path returns a tagged
    mock instead so the dashboard keeps working without IBM."""


def _classify_granite_error(exc: Exception) -> str:
    """Coarse error category (auth/model/timeout/network/unknown) so callers
    can distinguish failure modes without surfacing the raw SDK message."""
    low = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, TimeoutError) or "timeout" in low:
        return "timeout"
    if any(k in low for k in ("unauthorized", "401", "apikey", "api_key",
                              "authentication", "invalid api")):
        return "auth"
    if any(k in low for k in ("not found", "404", "deploy", "not available",
                              "not supported", "model id")):
        return "model"
    if any(k in low for k in ("connection", "network", "dns", "ssl",
                              "refused", "unreachable")):
        return "network"
    return "unknown"


def _parse_granite_json(raw_output: str) -> Optional[Dict]:
    """Extract a JSON object from Granite text output.

    Granite is instructed to return JSON only, but may wrap it in markdown
    fences or add prose. Extract the outermost {...} span and parse it;
    return None (never raise) when no valid object is present.
    """
    if not raw_output:
        return None
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw_output[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None

def _mock_granite_response(anomaly_input: dict, retrieved_docs: Optional[List[Dict]] = None) -> Dict:
    """
    Deterministic fallback that produces valid JSON matching spec schema,
    but grounded in actual numbers and evidence if RAG provided.
    This ensures end-to-end demo works offline, but explains real physics change.
    """
    subsystem = anomaly_input.get("subsystem", "power")
    flag = anomaly_input.get("physics_flag")
    conf = anomaly_input.get("physics_confidence", 0.8)
    curr = anomaly_input.get("current_values", {})
    nom = anomaly_input.get("nominal_values", {})
    t = anomaly_input.get("time_s", 0)

    evidence_ids = [d["id"] for d in (retrieved_docs or [])]

    if flag == "solar_degradation":
        risk = "HIGH" if curr.get("soc", 1) < 0.4 or curr.get("battery_voltage_v", 28) < 25 else "MEDIUM"
        solar = curr.get("solar_power_w", 250)
        factor = solar/520 if solar else 0.48
        net = solar - 400
        dSOC = net/3600/100
        cause = f"Solar array degradation - power dropped to {solar}W from nominal {nom.get('solar_power_w',520)}W (factor {factor:.2f}), net {net:.1f}W"
        reasoning = (
            f"At t={t}s, ML score {anomaly_input.get('anomaly_score')} (flag) + physics {flag} conf {conf} agree. "
            f"Real change: solar mean {solar}W < 0.7*Pmax 364W threshold per [DOC-POWER-002], "
            f"net power {net:.1f}W (520→{solar}W vs 400W load) → dSOC/dt={dSOC:.6f}/s negative, "
            f"SOC {curr.get('soc')} vs nominal {nom.get('soc')} (Δ {(curr.get('soc',0)-nom.get('soc',1))*100:+.1f}%), "
            f"voltage {curr.get('battery_voltage_v')}V vs {nom.get('battery_voltage_v')}V. "
            f"Pattern matches ramp injection 600-900s where degradation_factor 1.0→0.48. "
            f"Equilibrium: with +120W normal SOC rises to 1.0 in ~300s; with -150W drains to 0% in ~40min. "
        )
        if evidence_ids:
            reasoning += f"Evidence {', '.join(evidence_ids[:2])} confirms solar degradation signature; mission rule [DOC-MISSION-POWER-001] says if SOC<0.3 HIGH risk."
        action = "Shed non-critical loads to P_load ≤ P_solar (~250W) per [DOC-POWER-PROC-001], check Sun sensor alignment and panel string voltages, monitor depth-of-discharge, consider safe mode if SOC<0.2."
        evidence_used = evidence_ids[:3] if evidence_ids else ["DOC-POWER-002","DOC-POWER-PROC-001","DOC-MISSION-POWER-001"]
    elif flag == "radiator_degradation":
        risk = "HIGH" if curr.get("temperature_c",0) > 50 else "MEDIUM"
        temp = curr.get("temperature_c", 0)
        epsA = curr.get("epsilon_A", 0.0425)
        q_in = curr.get("heat_in_w",60)
        q_out = curr.get("heat_out_w", 30)
        net_heat = q_in - q_out
        dTdt = net_heat/2000
        cause = f"Radiator degradation - epsilon*A dropped to {epsA} from nominal {nom.get('epsilon_A',0.425)} (10% final), temp {temp}C vs nominal {nom.get('temperature_c',-42)}C, net heating +{net_heat:.1f}W"
        reasoning = (
            f"At t={t}s, ML score {anomaly_input.get('anomaly_score')} + physics {flag} conf {conf}. "
            f"Real change: Q_in={q_in}W constant (P_load*(1-η)), Q_out dropped {nom.get('heat_out_w')}W→{q_out}W (Δ {q_out-nom.get('heat_out_w',60):+.1f}W) because epsilon*A impaired, "
            f"net heating {net_heat:.1f}W → dT/dt={dTdt:.5f}K/s, "
            f"temperature {temp}C vs nominal {nom.get('temperature_c')}C (Δ {temp-nom.get('temperature_c',-42):+.1f}C) rising slope >0.003 threshold per [DOC-THERM-002] while heat_in stable (<1 W/s). "
            f"New equilibrium predicted 124C (60W/(σ*0.0425)), currently {temp}C heading there. "
        )
        if evidence_ids:
            reasoning += f"Retrieved {', '.join(evidence_ids[:2])} confirm radiator louver stuck/degraded coating; mission rule [DOC-MISSION-001] HIGH if T>60C."
        action = "Reduce load to lower Q_in per [DOC-THERM-PROC-001], improve radiator view factor by attitude slew, check louver status, activate backup radiator, monitor if T>60C enter safe mode per [DOC-MISSION-001]."
        evidence_used = evidence_ids[:3] if evidence_ids else ["DOC-THERM-002","DOC-THERM-PROC-001","DOC-MISSION-001"]
    else:
        ml_flag = anomaly_input.get("ml_flag", 0)
        if ml_flag:
            # ML detector tripped but no physics rule has confirmed the root
            # cause yet (e.g. mid-ramp t=605s). Do NOT call this nominal.
            risk = "MEDIUM"
            cause = (f"ML detector flagged an anomaly at t={t}s - deviation from learned normal telemetry; "
                     f"physics attribution pending, monitoring {curr.get('solar_power_w')}W solar, "
                     f"SOC {curr.get('soc')}, temp {curr.get('temperature_c')}C")
            reasoning = (f"At t={t}s, ML score {anomaly_input.get('anomaly_score')} crossed the detector threshold "
                         f"({anomaly_input.get('anomaly_score_threshold','n/a')}) while physics rules have not yet "
                         f"confirmed a subsystem signature. Score trend and telemetry deltas are being tracked "
                         f"for attribution; this is the leading indicator window before physics confirmation.")
            action = "Monitor telemetry trend; keep subsystems nominal until a physics rule or sustained ML flag confirms the root cause."
            evidence_used = evidence_ids[:2] if evidence_ids else ["DOC-MISSION-002"]
        else:
            risk = "LOW" if anomaly_input.get("anomaly_score",0) < 0.3 else "MEDIUM"
            cause = f"Nominal operation at t={t}s - no physics violation, ML score low"
            reasoning = f"ML score {anomaly_input.get('anomaly_score')} low, physics None. Current solar {curr.get('solar_power_w')}W ~ nominal {nom.get('solar_power_w')}W, SOC {curr.get('soc')} vs {nom.get('soc')}, temp {curr.get('temperature_c')}C vs {nom.get('temperature_c')}C stable. No injection. Q_in=Q_out balanced, equilibrium -42C with tuned mc_p 2000."
            action = "Continue nominal monitoring per [DOC-MISSION-002], no action required."
            evidence_used = evidence_ids[:2] if evidence_ids else ["DOC-MISSION-002"]

    result = {
        "risk": risk,
        "probable_cause": cause,
        "reasoning": reasoning,
        "recommended_action": action,
    }
    if retrieved_docs is not None:
        result["evidence_used"] = evidence_used
        result["confidence"] = round(float(conf),2) if flag else 0.55
        result["retrieved_docs"] = [{"id": d["id"], "title": d["title"], "score": round(d.get("score",0),3)} for d in retrieved_docs[:3]]

    return result

def _call_watsonx_granite(system_prompt: str, user_prompt: str,
                          model_id: Optional[str] = None,
                          timeout_s: float = 45.0) -> str:
    """
    Real watsonx call - adaptable to current SDK.
    Looks up env vars: WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL (optional),
    WATSONX_MODEL_ID (optional, defaults to GRANITE_DEFAULT_MODEL).
    Returns raw text output. Raises on any failure (never mocks).
    """
    if model_id is None:
        model_id = os.getenv("WATSONX_MODEL_ID", GRANITE_DEFAULT_MODEL)
    api_key = os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    if not api_key or not project_id:
        raise RuntimeError("WATSONX credentials missing")
    if not WATSONX_AVAILABLE:
        raise RuntimeError("ibm-watsonx-ai SDK not installed")

    creds = Credentials(api_key=api_key, url=url)
    model = ModelInference(
        model_id=model_id,
        credentials=creds,
        project_id=project_id,
        params={"decoding_method": "greedy", "max_new_tokens": 500, "temperature": 0.2}
    )
    # Combine system + user as prompt - Granite instruct expects chat format
    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    # Application-level timeout: the SDK exposes no per-call timeout, and a
    # hung IBM request must not stall the dashboard or API forever. The pool
    # is shut down without waiting so a genuinely stuck SDK call cannot hold
    # the caller hostage after we have already given up on it.
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(model.generate_text, prompt=full_prompt)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"watsonx Granite call exceeded {timeout_s:.0f}s timeout")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

def generate_explanation(anomaly_input: dict, use_rag: bool = True, top_k: int = 3,
                         strict: bool = False) -> Dict:
    """
    Main entry point used by Streamlit app and pipeline.

    Steps:
    1. Retrieve evidence via RAG if enabled
    2. Build prompt (RAG-enhanced or base)
    3. Try the real watsonx call when credentials are present
    4. Validate JSON schema

    strict=False (demo/dashboard default): a real-call failure falls back to
    the deterministic mock, but the result is tagged source="mock" with a
    sanitized granite_error so callers can never mistake it for real IBM
    inference.

    strict=True (credentialed smoke test): requires credentials AND a
    successful real call. On any failure it raises GraniteRequestError — it
    never silently substitutes the mock, so a smoke test can only pass when
    IBM actually answered.
    """
    global _last_real_state
    retrieved_docs = []
    if use_rag:
        try:
            retriever = get_retriever()
            retrieved_docs = retriever.query_from_anomaly(anomaly_input, top_k=top_k)
        except Exception as e:
            print(f"[RAG] retrieval failed: {e}")
            retrieved_docs = []

    # Choose prompt
    if use_rag and retrieved_docs:
        system_prompt = SYSTEM_PROMPT_RAG
        user_prompt = build_rag_user_prompt(anomaly_input, retrieved_docs)
    else:
        system_prompt = SYSTEM_PROMPT_BASE
        user_prompt = build_user_prompt(anomaly_input)

    has_key = bool(os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY"))
    use_real = WATSONX_AVAILABLE and has_key and os.getenv("WATSONX_PROJECT_ID")

    # Strict mode: credentials are REQUIRED — fail clearly, never mock.
    if strict and not use_real:
        missing = []
        if not WATSONX_AVAILABLE:
            missing.append("ibm-watsonx-ai SDK")
        if not has_key:
            missing.append("WATSONX_APIKEY")
        if not os.getenv("WATSONX_PROJECT_ID"):
            missing.append("WATSONX_PROJECT_ID")
        raise GraniteRequestError(
            "Real Granite smoke test requires: " + ", ".join(missing)
            + " (set them in .env — never committed)")

    if use_real:
        try:
            raw_output = _call_watsonx_granite(system_prompt, user_prompt)
            parsed = _parse_granite_json(raw_output)
            required = {"risk", "probable_cause", "reasoning", "recommended_action"}
            if parsed is None or not required.issubset(parsed.keys()):
                raise ValueError(
                    "Granite response missing required fields "
                    f"(got {sorted(parsed.keys()) if parsed else 'no JSON'})")
            # Enforce the same value contract the mock guarantees, so the
            # dashboard never renders an unexpected risk level.
            parsed["risk"] = parsed.get("risk", "MEDIUM").upper()
            if parsed["risk"] not in ("LOW", "MEDIUM", "HIGH"):
                parsed["risk"] = "MEDIUM"
            for key in ("probable_cause", "reasoning", "recommended_action"):
                if not isinstance(parsed.get(key), str) or not parsed[key].strip():
                    raise ValueError(f"Granite field '{key}' is empty or not text")
            if retrieved_docs:
                parsed.setdefault("evidence_used", [d["id"] for d in retrieved_docs])
                parsed.setdefault("confidence", anomaly_input.get("physics_confidence", 0.8))
                parsed.setdefault("retrieved_docs",
                                  [{"id": d["id"], "title": d["title"]} for d in retrieved_docs])
            parsed["source"] = "watsonx"
            _last_real_state = "succeeded"
            return parsed
        except Exception as e:
            _last_real_state = "failed:" + _classify_granite_error(e)
            if strict:
                raise GraniteRequestError(
                    f"Real watsonx Granite request FAILED ({_last_real_state}). "
                    f"The mock was NOT substituted. {type(e).__name__}") from e
            print(f"[Granite] watsonx call failed ({_last_real_state}): "
                  f"{type(e).__name__}, using tagged mock")
    elif strict:  # unreachable: strict+not use_real raises above
        raise GraniteRequestError("Granite credentials/SDK unavailable")

    # Fallback mock - always valid, always explicitly tagged as mock.
    mock_result = _mock_granite_response(anomaly_input, retrieved_docs if use_rag else None)
    mock_result["source"] = "mock"
    if use_real:
        # credentials were present but the real call failed -> say why
        mock_result["granite_error"] = _last_real_state
    return mock_result

def check_config() -> Dict:
    """Report whether a real watsonx Granite call is possible right now.

    Returns plain booleans/strings so the dashboard, tests, and the --check
    CLI can all use the same truth. No network is touched, no credentials
    are returned.
    """
    api_key = os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    ready = bool(WATSONX_AVAILABLE and api_key and project_id)
    return {
        "sdk_installed": bool(WATSONX_AVAILABLE),
        "api_key_present": bool(api_key),
        "project_id_present": bool(project_id),
        "url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "model_id": os.getenv("WATSONX_MODEL_ID", GRANITE_DEFAULT_MODEL),
        "ready_for_real_call": ready,
        # Clear top-level state for judges: MOCK (will never touch IBM),
        # REAL_READY (credentials+SDK present, call not yet proven),
        # REAL_FAILED (credentials present but the last real call failed).
        "mode": "REAL_READY" if ready else "MOCK",
        "last_real_request": _last_real_state,
    }


def granite_status() -> Dict:
    """Public status dict for /api/health — never includes credentials."""
    cfg = check_config()
    if cfg["mode"] == "REAL_READY" and cfg["last_real_request"].startswith("failed"):
        cfg["mode"] = "REAL_FAILED"
    return cfg

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Granite client self-test")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify watsonx config; makes a real call if a key is set",
    )
    args = parser.parse_args()

    if args.check:
        cfg = check_config()
        print("=== Granite / watsonx config check ===")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
        if not cfg["ready_for_real_call"]:
            missing = []
            if not cfg["sdk_installed"]:
                missing.append("ibm-watsonx-ai SDK (pip install -r requirements.txt)")
            if not cfg["api_key_present"]:
                missing.append("WATSONX_APIKEY")
            if not cfg["project_id_present"]:
                missing.append("WATSONX_PROJECT_ID")
            print("\nNOT READY for a real watsonx call. Missing:")
            for item in missing:
                print(f"  - {item}")
            print("Paste the key and project id into .env, then re-run this check.")
            raise SystemExit(1)
        print("\nConfig looks ready. Making a real Granite call to verify the key...")
        print(f"Model: {cfg['model_id']}  URL: {cfg['url']}")
        from .prompts import example_input_json
        try:
            # STRICT: the smoke test must never substitute the mock — it can
            # only pass when IBM actually answered with schema-valid JSON.
            out = generate_explanation(example_input_json(), use_rag=True, strict=True)
            assert out.get("source") == "watsonx", f"expected a real call, got {out.get('source')}"
            print(json.dumps(out, indent=2))
            print("\nCHECK PASS: real watsonx Granite call succeeded.")
        except Exception as e:
            print(f"\nCHECK FAIL: {e}")
            print("The key or project id may be wrong, the model may not be deployable"
                  " in your watsonx project, or the model is not available in the"
                  " configured region. Fix .env and re-run.")
            raise SystemExit(1)
        raise SystemExit(0)

    from .prompts import example_input_json, example_thermal_input
    print("=== Test Granite Client Mock (Power) ===")
    inp = example_input_json()
    out = generate_explanation(inp, use_rag=True)
    print(json.dumps(out, indent=2))
    print("\n=== Test Thermal ===")
    inp2 = example_thermal_input()
    out2 = generate_explanation(inp2, use_rag=True)
    print(json.dumps(out2, indent=2))
    # Verify schema
    assert out["risk"] in ("LOW","MEDIUM","HIGH")
    assert "probable_cause" in out
    assert "reasoning" in out
    assert "recommended_action" in out
    print("PASS schema valid")

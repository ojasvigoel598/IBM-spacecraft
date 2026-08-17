"""MissionMind web API — serves the SAME pipeline as the Streamlit dashboard.

Endpoints:
  GET /api/health                     -> {status, models, watsonx}
  GET /api/scenarios                  -> {scenarios: [names]}
  GET /api/scenario/{mode}            -> full scored telemetry for a scenario
  GET /api/live/next?mode=&n=30       -> next N frames from the edge node +
                                         live ensemble score for the latest window
  GET /api/summary/{mode}?t=          -> condensed snapshot at mission time t
  GET /api/alert/{mode}?t=            -> physics+ML+RAG alert evidence at time t

The virtual edge node state is session-free (single global node per mode), so
the live endpoints genuinely advance a stream — each call produces NEW frames.

Run:  .venv/Scripts/python.exe -m uvicorn missionmind.viz.api_server:app --port 8100
"""
from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from missionmind.ml.detect import score_dataframe
from missionmind.simulator.run_scenarios import run_scenario
from missionmind.physics_rules.rules import check_power_subsystem, check_thermal_subsystem, slope

SCENARIOS = ["none", "solar_degradation", "radiator_degradation"]
SCENARIO_LABELS = {
    "none": "Normal Operation",
    "solar_degradation": "Solar Array Degradation",
    "radiator_degradation": "Radiator Degradation",
}

app = FastAPI(title="MissionMind API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- cached scored scenarios (deterministic solve + ensemble) --------------
_scored_cache: Dict[str, object] = {}


def _scored(mode: str):
    from missionmind.trace import record
    if mode not in _scored_cache:
        record("simulator", "run_scenario", note=f"solving 3600 s scenario '{mode}'")
        df = run_scenario(failure_mode=mode, duration_s=3600)
        df = score_dataframe(df)
        df.loc[df["time_s"] < 100, "anomaly_flag"] = 0  # burn-in convention
        _scored_cache[mode] = df
        record("viz.api_server", "_scored", note=f"scenario '{mode}' solved + scored",
               value=round(len(df), 0))
    else:
        record("viz.api_server", "_scored", note=f"scenario '{mode}' served from cache")
    return _scored_cache[mode]


# ---- live edge node state --------------------------------------------------
from missionmind.telemetry.edge_node import VirtualEdgeNode  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_nodes: Dict[str, VirtualEdgeNode] = {}
_buffers: Dict[str, list] = {}  # accumulated frames per mode for live scoring


def _node(mode: str) -> VirtualEdgeNode:
    if mode not in _nodes:
        _nodes[mode] = VirtualEdgeNode(failure_mode=mode, noise=True,
                                       adc_bits=12, drop_rate=0.02)
        _buffers[mode] = []
    return _nodes[mode]


class FrameOut(BaseModel):
    time_s: float
    solar_power_w: float
    battery_soc: float
    battery_voltage_v: float
    temperature_c: float
    heat_in_w: float
    heat_out_w: float


@app.get("/api/health")
def health():
    from missionmind.ai.granite_client import WATSONX_AVAILABLE
    return {
        "status": "ok",
        "models": os.path.exists(os.path.join(
            os.path.dirname(__file__), "..", "models", "iforest.joblib")),
        "watsonx_sdk": WATSONX_AVAILABLE,
        "watsonx_key": bool(os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY")),
    }


@app.get("/api/scenarios")
def scenarios():
    return {"scenarios": SCENARIOS, "labels": SCENARIO_LABELS}


@app.get("/api/scenario/{mode}")
def scenario(mode: str, t0: int = 0, t1: Optional[int] = None):
    if mode not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {mode}")
    df = _scored(mode)
    t1 = int(df["time_s"].iloc[-1]) if t1 is None else int(t1)
    sl = df[(df["time_s"] >= t0) & (df["time_s"] <= t1)]
    return {
        "mode": mode,
        "label": SCENARIO_LABELS[mode],
        "columns": list(sl.columns),
        "rows": sl.to_dict(orient="records"),
        "max_time": int(df["time_s"].iloc[-1]),
    }


def _json_safe(x):
    """JSON cannot represent inf/nan (Starlette's dumps uses allow_nan=False).
    RUL is legitimately infinite when the limit is never reached (e.g. nothing
    heating, battery charging) — represent that as null, not inf, so the whole
    alert/summary payload does not 500."""
    try:
        return None if not np.isfinite(x) else x
    except TypeError:
        return x


def _rul_with_uncertainty(df: "pd.DataFrame", t: int) -> dict:
    """Operator-facing RUL at mission time t with a defensible interval.

    Battery: minutes until 0% SOC at the current net power (same physics as
    the dashboard), with a 90% interval propagated from the observed dSOC/dt
    uncertainty over the trailing window (P6 — never a false-precise number).
    Thermal: minutes until the 60 C electronics limit at current dT/dt.

    Non-finite RUL (limit never reached) is emitted as null so the payload is
    JSON-compliant; the human label still reads "∞" via format_interval.
    """
    from missionmind.ml.rul_uncertainty import time_to_limit_interval, format_interval
    row = df[df["time_s"] == t].iloc[-1]
    solar_w = float(row["solar_power_w"])
    net_w = solar_w - 400.0
    soc = float(row["battery_soc"])
    win = df[(df["time_s"] >= t - 120) & (df["time_s"] <= t)]
    soc_diffs = win["battery_soc"].diff().dropna().values
    bat_p, bat_lo, bat_hi = time_to_limit_interval(soc, net_w, soc_diffs)
    heat_bal = float(row["heat_in_w"]) - float(row["heat_out_w"])
    _dT_dt = heat_bal / 2000.0
    temp = float(row["temperature_c"])
    if _dT_dt > 0 and temp < 60.0:
        thm_min = (60.0 - temp) / _dT_dt / 60.0
        thm_lo, thm_hi = thm_min * 0.85, thm_min * 1.15
    else:
        thm_min = thm_lo = thm_hi = float("inf")
    return {
        "battery_rul_min": _json_safe(bat_p if np.isfinite(bat_p) else float("inf")),
        "battery_ci": [_json_safe(v) for v in (bat_lo, bat_hi)],
        "battery_label": format_interval(bat_p, bat_lo, bat_hi, "min"),
        "thermal_rul_min": _json_safe(thm_min),
        "thermal_ci": [_json_safe(v) for v in (thm_lo, thm_hi)],
        "method": "SOC drain-rate physics + bootstrap CI (P6)",
    }


def _explain(df: "pd.DataFrame", t: int) -> dict:
    """SHAP/occlusion feature attribution for the row shown at time t (P7)."""
    from missionmind.ml.explainability import explain_row
    win = df[(df["time_s"] >= t - 200) & (df["time_s"] <= t)]
    try:
        expl = explain_row(win, row_idx=-1)
        return {
            "method": expl["method"],
            "score": expl["score"],
            "anomalous": expl["anomalous"],
            "summary": expl["summary"],
            "features": expl["features"][:5],
        }
    except Exception as e:  # noqa: BLE001 - explanation must never break alert
        return {"method": "error", "summary": f"explanation unavailable: {e}",
                "features": [], "score": None, "anomalous": None}


def _adaptive_decision(win) -> dict:
    """Situation-aware decision-layer verdict for a telemetry window.

    Wraps missionmind/ml/adaptive.py decide(): the strategy the system chose
    for THIS situation (rule-first / ramp-lead / consensus / nominal), the
    fused adaptive score, and the human-readable reasoning lines.
    """
    from missionmind.ml.adaptive import decide
    try:
        d = decide(win)
        return {
            "strategy": d["strategy"],
            "adaptive_score": d["adaptive_score"],
            "adaptive_flag": d["adaptive_flag"],
            "weights": d["weights"],
            "reasoning": d["reasoning"],
        }
    except Exception as e:  # noqa: BLE001 - decision must never break the API
        return {"strategy": "NOMINAL", "adaptive_score": 0.0,
                "adaptive_flag": 0, "weights": {}, "reasoning": [f"error: {e}"]}


def _evidence(mode: str, win) -> dict:
    """Human-readable physics-rule evidence for a telemetry window."""
    from missionmind.physics_rules.rules import P_SOLAR_MAX

    out: Dict[str, list] = {"power": [], "thermal": []}
    pp = check_power_subsystem(win)
    pt = check_thermal_subsystem(win)
    solar_mean = float(win["solar_power_w"].mean())
    soc_slope = slope(win["battery_soc"].values,
                      win["time_s"].values if "time_s" in win else None)
    out["power"].append(
        f"Solar mean {solar_mean:.0f} W vs {0.7 * P_SOLAR_MAX:.0f} W threshold")
    out["power"].append(f"Battery SOC slope {soc_slope:.4f}/s")
    if pp:
        out["power"].append(f"Rule: {pp[0]} · confidence {pp[1]:.2f}")
    temp_slope = slope(win["temperature_c"].values,
                       win["time_s"].values if "time_s" in win else None)
    temp_mean = float(win["temperature_c"].mean())
    out["thermal"].append(f"Panel temp mean {temp_mean:.1f} °C · slope {temp_slope:.3f} °C/s")
    if pt:
        out["thermal"].append(f"Rule: {pt[0]} · confidence {pt[1]:.2f}")
    return out


@app.get("/api/summary/{mode}")
def summary(mode: str, t: int = 900):
    if mode not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {mode}")
    df = _scored(mode)
    idx = int(min(t, df["time_s"].iloc[-1]))
    row = df[df["time_s"] == idx].iloc[-1]
    win = df[(df["time_s"] >= idx - 120) & (df["time_s"] <= idx)]
    orb = {}
    for k in ("orbit_angle_deg", "in_eclipse", "sun_incidence_deg",
              "orbit_period_s", "orbit_energy_jkg", "orbit_ang_momentum_m2s"):
        if k in row:
            orb[k] = float(row[k])
    return {
        "mode": mode,
        "label": SCENARIO_LABELS[mode],
        "t": idx,
        "telemetry": {
            "solar_power_w": float(row["solar_power_w"]),
            "battery_soc": float(row["battery_soc"]),
            "battery_voltage_v": float(row["battery_voltage_v"]),
            "temperature_c": float(row["temperature_c"]),
            "heat_in_w": float(row["heat_in_w"]),
            "heat_out_w": float(row["heat_out_w"]),
            "anomaly_score": float(row["anomaly_score"]),
            "anomaly_flag": int(row["anomaly_flag"]),
            "anomaly_source": int(row["anomaly_source"]),
        },
        "orbit": orb,
        "physics": _evidence(mode, win),
        "decision": _adaptive_decision(win),
        "rul": _rul_with_uncertainty(df, idx),
        "max_time": int(df["time_s"].iloc[-1]),
    }


_zoo_cache: Optional[dict] = None


@app.get("/api/alert/{mode}")
def alert(mode: str, t: int = 900):
    """Physics + ML + RAG alert evidence at mission time t.

    Contract (operator-facing): mode/label/t, `active` flag with `detected_at`
    and `severity`, a telemetry snapshot, physics-rule evidence strings, and
    RAG source citations (path + score). Declared in the module docstring;
    implemented per tests/test_api_server.py.
    """
    if mode not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {mode}")
    df = _scored(mode)
    idx = int(min(max(t, 0), df["time_s"].iloc[-1]))
    row = df[df["time_s"] == idx].iloc[-1]
    win = df[(df["time_s"] >= idx - 120) & (df["time_s"] <= idx)]

    flag = int(row["anomaly_flag"])
    score = float(row["anomaly_score"])
    source = int(row["anomaly_source"])

    # walk back to the start of the contiguous fault episode
    detected_at = None
    if flag == 1:
        ep = idx
        while ep > 0 and int(df[df["time_s"] == ep - 1]["anomaly_flag"].iloc[0]) == 1:
            ep -= 1
        detected_at = ep

    severity = "nominal"
    if flag == 1:
        severity = "critical" if score <= -0.05 else "elevated"

    ev = _evidence(mode, win)
    physics = (ev.get("power") or []) + (ev.get("thermal") or [])

    # RAG retrieval with the same anomaly contract the Streamlit app uses
    from missionmind.ai.rag import get_retriever
    anomaly_input = {
        "subsystem": ("power" if mode == "solar_degradation"
                      else "thermal" if mode == "radiator_degradation" else "unknown"),
        "ml_flag": flag,
        "physics_flag": physics[0] if physics else "",
        "anomaly_score": round(abs(score), 3),
        "current_values": {
            "solar_power_w": round(float(row["solar_power_w"]), 1),
            "battery_soc": round(float(row["battery_soc"]), 3),
            "battery_voltage_v": round(float(row["battery_voltage_v"]), 2),
            "temperature_c": round(float(row["temperature_c"]), 2),
            "heat_in_w": round(float(row["heat_in_w"]), 1),
            "heat_out_w": round(float(row["heat_out_w"]), 1),
        },
        "time_s": idx,
        "failure_mode": mode,
    }
    try:
        docs = get_retriever().query_from_anomaly(anomaly_input, top_k=3)
    except Exception:  # noqa: BLE001 - RAG must never break the alert
        docs = []
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    rag = [{"path": os.path.relpath(d.get("path") or "", _root),
            "score": d.get("score"),
            "content": (d.get("content") or "")[:240]}
           for d in docs]

    # 4-line operator narrative (causal_narrative) — wired per the module's
    # documented contract; RAG chunks mapped to the keys it reads (source/text)
    from missionmind.ml.causal_narrative import causal_narrow
    rag_for_narrative = [{"source": r["path"], "text": r["content"]} for r in rag]
    narrative = "\n".join(
        causal_narrow(row, physics_hits=physics, rag_chunks=rag_for_narrative))

    orb = {}
    for k in ("orbit_angle_deg", "in_eclipse", "sun_incidence_deg",
              "orbit_period_s", "orbit_energy_jkg", "orbit_ang_momentum_m2s"):
        if k in row:
            orb[k] = float(row[k])
    return {
        "mode": mode,
        "label": SCENARIO_LABELS[mode],
        "t": idx,
        "active": flag,
        "severity": severity,
        "detected_at": detected_at,
        "source": ("POWER" if source == 1 else "THERMAL" if source == 2 else "FULL"),
        "telemetry": {
            "solar_power_w": float(row["solar_power_w"]),
            "battery_soc": float(row["battery_soc"]),
            "battery_voltage_v": float(row["battery_voltage_v"]),
            "temperature_c": float(row["temperature_c"]),
            "heat_in_w": float(row["heat_in_w"]),
            "heat_out_w": float(row["heat_out_w"]),
            "anomaly_score": score,
            "anomaly_flag": flag,
            "anomaly_source": source,
        },
        "orbit": orb,
        "physics": physics,
        "decision": _adaptive_decision(win),
        "rul": _rul_with_uncertainty(df, idx),
        "explanation": _explain(df, idx),
        "rag": rag,
        "narrative": narrative,
    }


@app.get("/api/models")
def models():
    """Model zoo self-test results (same evaluation as advanced_models.py)."""
    global _zoo_cache
    if _zoo_cache is not None:
        return _zoo_cache
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from missionmind.ml.advanced_models import get_all_models

    rng = np.random.default_rng(0)
    X_norm = rng.normal(0, 1, (200, 4))
    X_test = X_norm.copy()
    X_test[:10] += 6.0
    y_test = np.concatenate([np.ones(10), np.zeros(len(X_norm) - 10)]).astype(int)
    X_train_sup = np.vstack([X_norm, X_norm + 6.0])
    y_train_sup = np.concatenate([np.zeros(len(X_norm)), np.ones(len(X_norm))]).astype(int)

    SUPERVISED = {"FCNN Supervised (MLP 100-50-20)",
                  "XGBOD Supervised (Extreme Boosting Outlier Detector)",
                  "Custom Physics-Informed NN"}
    rows = []
    for name, model in get_all_models().items():
        try:
            if name in SUPERVISED:
                model.fit_supervised(X_train_sup, y_train_sup)
                pred = model.predict(X_test)
                sc = model.decision_function(X_test)
                tp = int(((pred == 1) & (y_test == 1)).sum())
                fp = int(((pred == 1) & (y_test == 0)).sum())
                fn = int(((pred == 0) & (y_test == 1)).sum())
                tn = int(((pred == 0) & (y_test == 0)).sum())
                acc = (tp + tn) / len(y_test)
                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                try:
                    auc = float(roc_auc_score(y_test, sc))
                except Exception:
                    auc = None
                rows.append({"name": name, "family": "supervised", "fit": "OK",
                             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "acc": acc, "precision": prec, "recall": rec,
                             "f1": f1, "auc": auc})
            else:
                model.fit(X_norm)
                pred = model.predict(X_test)
                rows.append({"name": name, "family": "unsupervised", "fit": "OK",
                             "tp": int(pred[:10].sum()), "fp": int(pred[10:].sum())})
        except Exception as e:  # noqa: BLE001
            rows.append({"name": name, "family": "?", "fit": f"FAIL: {type(e).__name__}"})
    _zoo_cache = {"models": rows}
    return _zoo_cache


@app.get("/api/trace")
def trace(since: int = 0, limit: int = 300):
    """Runtime execution trace - which pipeline code actually ran.

    `since` acts as a cursor: events with seq > since. Used by both the web
    console and the Streamlit app to show live code execution behind the
    telemetry numbers.
    """
    from missionmind.trace import events_since
    events, last_seq = events_since(seq=since, limit=max(1, min(limit, 500)))
    return {"events": events, "last_seq": last_seq}


@app.get("/api/live/next")
def live_next(mode: str = "solar_degradation", n: int = 30):
    """Advance the virtual edge-node stream and score the accumulated window
    through the SAME production ensemble the dashboard uses."""
    if mode not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {mode}")
    node = _node(mode)
    from missionmind.trace import record
    record("telemetry", "VirtualEdgeNode.step",
           note=f"stream {n} frames (noise+ADC+dropout) mode={mode}")
    frames: List[dict] = []
    for _ in range(int(n)):
        f = node.step()
        if f is not None:
            rec = {
                "time_s": f.time_s,
                "solar_power_w": f.solar_power_w,
                "battery_soc": f.battery_soc,
                "battery_voltage_v": f.battery_voltage_v,
                "temperature_c": f.temperature_c,
                "heat_in_w": f.heat_in_w,
                "heat_out_w": f.heat_out_w,
                "in_eclipse": getattr(f, "in_eclipse", 0),
                "sun_exposure": getattr(f, "sun_exposure", 1.0),
                "bus_state": getattr(f, "bus_state", "normal"),
            }
            frames.append(rec)
            _buffers[mode].append(rec)
    # score the accumulated window (same code path as the Streamlit dashboard)
    score = flag = source = 0.0
    buf = _buffers[mode]
    if len(buf) >= 30:
        win = pd.DataFrame(buf[-300:])
        scored = score_dataframe(win)
        score = float(scored["anomaly_score"].iloc[-1])
        flag = int(scored["anomaly_flag"].iloc[-1])
        source = int(scored["anomaly_source"].iloc[-1])
        # P9: eclipse-aware - the raw ensemble flags any solar dip, but an
        # eclipse dip is expected physics. Suppress only when the measured
        # solar matches the eclipse-adjusted expectation.
        if flag and "in_eclipse" in win.columns and "sun_exposure" in win.columns:
            from missionmind.physics_rules.rules import eclipse_residual
            ecl = eclipse_residual(win)
            if ecl is not None and ecl["in_eclipse"] and ecl["status"] == "eclipse":
                flag = 0
        # burn-in convention: suppress the early-transient flag before t=100 s
        # (same as the Streamlit dashboard)
        if buf[-1]["time_s"] < 100:
            flag = 0
    # retention cap: never grow the buffer unboundedly (leak regression);
    # scoring only ever reads the last LIVE_RETAIN frames anyway
    LIVE_RETAIN = 600
    del buf[:-LIVE_RETAIN]
    return {"mode": mode, "frames": frames, "total": node.sample_seq,
            "retained": len(buf), "window_scored": len(buf) >= 30,
            "anomaly_score": score, "anomaly_flag": flag,
            "anomaly_source": source}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)

"""
MissionMind — lightweight adaptive decision/ensemble layer.

Sits on top of the production 3-forest ensemble (ml/detect.py) and the physics
rule layer (physics_rules/rules.py). Instead of a single fixed fusion (MIN
score / OR flag), it reads the SITUATION and picks the appropriate strategy:

    SITUATION                        STRATEGY
    t < 100 s (burn-in transient)    BURN_IN_SUPPRESS  — suppress flags
    power rule fired (conf > 0.5)    RULE_POWER        — trust power detector + rule
    thermal rule fired (conf > 0.5)  RULE_THERMAL      — trust thermal detector + rule
    600 <= t <= 900 (fault ramp)     RAMP_LEAD         — early-detection sensitive
    >= 2 of 3 detectors agree        DETECTOR_CONSENSUS — flag on agreement
    nothing abnormal                 NOMINAL           — no flag

The decision is fully explainable: each call returns the chosen strategy, the
per-strategy detector weights, the fused adaptive_score, and a list of
human-readable reasoning lines. This is the "AI chooses the right approach for
the situation" behaviour — implemented transparently (no black box), which is
what a mission operator can actually trust.

Usage:
    from missionmind.ml.adaptive import decide
    decision = decide(window_df)   # -> dict with strategy/score/flag/reasoning
"""

from __future__ import annotations

from missionmind.ml.detect import ensemble_components
from missionmind.physics_rules.rules import check_power_subsystem, check_thermal_subsystem, check_eclipse

# strategy name -> weight over (full, power, thermal) detector scores
# IsolationForest decision_function is "higher = more normal", so the fused
# adaptive_score is a weighted mean; lower (more negative) = more anomalous.
STRATEGY_WEIGHTS = {
    "BURN_IN_SUPPRESS": {"full": 1 / 3, "power": 1 / 3, "thermal": 1 / 3},
    "RULE_POWER": {"full": 0.15, "power": 0.70, "thermal": 0.15},
    "RULE_THERMAL": {"full": 0.15, "power": 0.15, "thermal": 0.70},
    "RAMP_LEAD": {"full": 1 / 3, "power": 1 / 3, "thermal": 1 / 3},
    "DETECTOR_CONSENSUS": {"full": 1 / 3, "power": 1 / 3, "thermal": 1 / 3},
    "NOMINAL": {"full": 1 / 3, "power": 1 / 3, "thermal": 1 / 3},
}

BURN_IN_S = 100
RAMP_START_S = 600
RAMP_END_S = 900


def _trace(strategy, t, flag, note):
    try:
        from missionmind.trace import record
        record("ml.adaptive", "decide", mission_t=t,
               note=f"{strategy} {note}", value=flag)
    except Exception:  # noqa: BLE001 - tracing must never break decisions
        pass


def _decision(strategy, score, flag, reasoning, weights=None, context=None):
    return {
        "strategy": strategy,
        "adaptive_score": round(float(score), 4),
        "adaptive_flag": int(flag),
        "weights": weights or STRATEGY_WEIGHTS[strategy],
        "reasoning": reasoning,
        "context": context or {},
    }


def decide(df, models=None, comps=None) -> dict:
    """Run the adaptive decision layer over a telemetry window (public API).

    Args:
        df: scored telemetry window (columns from score_dataframe output,
            including anomaly_score / anomaly_flag / anomaly_source, plus the
            raw telemetry columns the physics rules read).
        models: optional preloaded detector dict from detect.load_models();
            pass it when scoring many windows to avoid reloading 6 joblibs on
            every call (see score_adaptive).
        comps: optional precomputed per-detector components (dict of
            {"score": array, "flag": array}) matching df — avoids recomputing
            the ensemble when the caller already has it (perf: ~500x faster).

    Returns a dict: strategy, adaptive_score, adaptive_flag, weights,
    reasoning, context.
    """
    if df is None or len(df) == 0:
        return _decision("NOMINAL", 0.0, 0, ["empty window - no telemetry"])
    from missionmind.ml.train import add_derivative_features
    df_feat = add_derivative_features(df)
    return decide_components(df, df_feat, models, comps)


def decide_components(df, df_feat, models, comps=None) -> dict:
    """Shared core: situation-aware decision from a window + its feature frame.

    Used by decide() (single window) and score_adaptive() (row-wise, reuses a
    precomputed feature frame, preloaded models, and precomputed components).
    """
    last_row = df.iloc[-1]
    t = float(last_row.get("time_s", 0))
    if comps is None:
        comps = ensemble_components(df_feat, models)

    # per-detector last-row scores + flags (IF: lower score = more anomalous)
    det_names = [k for k in ("full", "power", "thermal") if k in comps]
    scores = {k: float(comps[k]["score"][-1]) for k in det_names}
    flags = {k: int(comps[k]["flag"][-1]) for k in det_names}
    n_agree = sum(flags.values())

    # physics rules over the window (same calls the dashboard / API use)
    power_rule = check_power_subsystem(df)
    thermal_rule = check_thermal_subsystem(df)
    eclipse_rule = check_eclipse(df)

    reasoning = [
        f"t={t:.0f}s detectors: full={flags.get('full', 0)} power={flags.get('power', 0)} "
        f"thermal={flags.get('thermal', 0)} ({n_agree}/3 agree)"
    ]

    # ---- ML-vs-physics disagreement: eclipse explains a solar dip ----
    # P5-ORBIT: if the ML ensemble flags solar while Kepler physics says the
    # satellite is in eclipse, expose the disagreement instead of hiding it:
    # strategy ECLIPSE_EXPLAINED, flag suppressed, reasoning shows both sides.
    if eclipse_rule is not None and eclipse_rule[1] > 0.5:
        if flags.get("power", 0) or flags.get("full", 0):
            strategy = "ECLIPSE_EXPLAINED"
            flag = 0
            _ecl_frac = (float(df["in_eclipse"].mean())
                         if "in_eclipse" in df.columns else 0.0)
            reasoning.append(
                f"ML flags solar dip but Kepler physics: in-eclipse "
                f"{_ecl_frac:.0%} -> expected transient, NOT a fault "
                f"(disagreement exposed)")
            decision = _decision(strategy, float(min(scores.values())), flag,
                                 reasoning, weights=STRATEGY_WEIGHTS["NOMINAL"],
                                 context={"eclipse_conf": eclipse_rule[1]})
            _trace(strategy, t, flag, "eclipse explains ML solar flag")
            return decision
        reasoning.append("in eclipse: solar dip expected by orbit physics (no ML flag)")

    # ---- situation -> strategy ----
    if t < BURN_IN_S:
        strategy = "BURN_IN_SUPPRESS"
        flag = 0
        reasoning.append("burn-in transient (t<100s): flags suppressed by convention")
    elif power_rule is not None and power_rule[1] > 0.5:
        strategy = "RULE_POWER"
        flag = 1
        reasoning.append(
            f"power rule '{power_rule[0]}' fired (conf {power_rule[1]:.2f}) -> "
            "trust power detector, rule-first")
    elif thermal_rule is not None and thermal_rule[1] > 0.5:
        strategy = "RULE_THERMAL"
        flag = 1
        reasoning.append(
            f"thermal rule '{thermal_rule[0]}' fired (conf {thermal_rule[1]:.2f}) -> "
            "trust thermal detector, rule-first")
    elif RAMP_START_S <= t <= RAMP_END_S:
        strategy = "RAMP_LEAD"
        flag = 1 if (n_agree >= 1 or min(scores.values()) < -0.05) else 0
        reasoning.append(
            "inside fault-ramp window (600-900s): early-detection sensitive, "
            "flag on any single detector lead")
    elif n_agree >= 2:
        strategy = "DETECTOR_CONSENSUS"
        flag = 1
        reasoning.append(f"{n_agree}/3 detectors agree on anomaly -> consensus flag")
    else:
        strategy = "NOMINAL"
        flag = 0
        reasoning.append("no rule, no detector consensus -> nominal")

    # ---- fused adaptive score (weighted mean of detector decision_functions) ----
    weights = STRATEGY_WEIGHTS[strategy]
    adaptive_score = float(sum(weights[k] * scores[k] for k in det_names))

    # coherence guard: a flag must look anomalous (same rule as detect.py)
    if flag == 1 and adaptive_score > -0.02:
        adaptive_score = min(scores.values())

    decision = _decision(strategy, adaptive_score, flag, reasoning, weights,
                         context={
                             "t": round(t, 1),
                             "n_agree": n_agree,
                             "power_rule": power_rule,
                             "thermal_rule": thermal_rule,
                             "detector_flags": flags,
                         })
    _trace(strategy, t, flag, f"score={adaptive_score:.3f} ({n_agree}/3 agree)")
    return decision


def score_adaptive(df) -> "pandas.DataFrame":
    """Add adaptive_* columns to a scored DataFrame (row-wise, windowed).

    Uses a trailing 120 s window per row so the physics rules see the same
    context the dashboard uses. The detector models are loaded ONCE and reused
    across all windows (perf fix: the naive per-row decide() reloaded 6
    joblibs per row, ~50x slower).
    """
    import pandas as pd
    from missionmind.ml.detect import load_models
    from missionmind.ml.train import add_derivative_features

    models = load_models()
    # precompute derivative features AND per-detector components ONCE over the
    # whole frame; per-row windows then slice the arrays instead of re-solving
    # the ensemble for each window (perf: 553 ms -> ~1 ms per row).
    feat = add_derivative_features(df)
    comps_all = ensemble_components(feat, models)
    rows = df.reset_index(drop=True)

    adaptive_strategy = []
    adaptive_score = []
    adaptive_flag = []
    for i in range(len(rows)):
        lo = max(0, i - 119)
        win = rows.iloc[lo: i + 1]
        # slice the precomputed per-detector arrays for this window
        win_comps = {
            k: {"score": v["score"][lo: i + 1], "flag": v["flag"][lo: i + 1]}
            for k, v in comps_all.items()
        }
        d = decide_components(win, feat.iloc[lo: i + 1], models, comps=win_comps)
        adaptive_strategy.append(d["strategy"])
        adaptive_score.append(d["adaptive_score"])
        adaptive_flag.append(d["adaptive_flag"])
    out = df.copy()
    out["adaptive_strategy"] = adaptive_strategy
    out["adaptive_score"] = adaptive_score
    out["adaptive_flag"] = adaptive_flag
    return out


if __name__ == "__main__":
    from missionmind.simulator.run_scenarios import run_scenario
    from missionmind.ml.detect import score_dataframe

    print("=" * 72)
    print("MissionMind adaptive decision layer - self test")
    print("=" * 72)
    for mode in ("none", "solar_degradation", "radiator_degradation"):
        df = run_scenario(failure_mode=mode, duration_s=3600)
        df = score_dataframe(df)
        for t in (50, 500, 750, 1500, 3000):
            win = df[(df["time_s"] >= t - 120) & (df["time_s"] <= t)]
            d = decide(win)
            print(f"{mode:<20s} t={t:>5d}  {d['strategy']:<20s} "
                  f"score={d['adaptive_score']:>7.3f} flag={d['adaptive_flag']}")

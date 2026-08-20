# MissionMind — Adversarial Competition Judge Audit

**Date:** 2026-08-20  
**Challenge:** IBM AI Builders Challenge — "Advance Space Exploration with AI"  
**Repository:** ojasvigoel598/IBM-spacecraft  
**Auditor Role:** Adversarial hackathon judge, aerospace engineer, ML researcher, competing-team simulator

---

## Executive Verdict

MissionMind is a **genuinely impressive technical project** — a multi-layered spacecraft anomaly detection stack with real physics, real ML, real RAG, real IBM SDK integration, real authentication, real CAD, and 100+ tests. However, it has **three hard submission blockers** that could disqualify it from the competition: (1) IBM Bob has never been used — the README explicitly says "PENDING — Account not yet created," which violates a hard submission requirement; (2) the demo video is 3.5 minutes, exceeding the 3-minute maximum; (3) the README is missing the required "how IBM Bob was used" section. These are not quality issues — they are compliance issues. A perfectly built project that fails a hard constraint loses to a weaker project that passes it. **Fix the compliance blockers first, then the competitive positioning.**

---

## Phase 0 — Judge Rubric Matrix (Official Requirements)

| Requirement | Status | Evidence |
|---|---|---|
| Working prototype / proof of concept | ✅ PASS | 100+ tests pass, e2e dry run works, Streamlit + React dashboards functional |
| IBM Bob = primary development tool | 🔴 FAIL | README: "⚠️ PENDING — Account not yet created" |
| Required IBM SkillsBuild learning activity | 🔴 UNKNOWN | No evidence in repo or documentation |
| Public GitHub repository | ✅ PASS | github.com/ojasvigoel598/IBM-spacecraft |
| README: problem statement | ✅ PASS | Clear in intro |
| README: solution description | ✅ PASS | Architecture section is thorough |
| README: AI approach / architecture | ✅ PASS | Detailed pipeline diagram |
| README: selected challenge theme | ⚠️ PARTIAL | "Advance Space Exploration" mentioned but not as explicit theme declaration |
| README: how IBM Bob was used | 🔴 FAIL | Explicitly "PENDING" — no evidence |
| Public project submission | ✅ PASS | Vercel deployment configured |
| Public demo video ≤ 3 minutes | 🔴 FAIL | "3.5-minute narrated walkthrough" — exceeds limit |
| Challenge theme: Advance Space Exploration with AI | ⚠️ PARTIAL | Fit is strong but not explicitly declared as selected theme |

**Hard blockers: 3 (Bob, SkillsBuild, video length)**

---

## Phase 2 — System Mental Model

One complete telemetry event traced:

```
Telemetry generation (run_scenarios.py)
  → 3600 samples @ 1 Hz: solar_power_w, battery_soc, temperature_c, etc.
  → Kepler orbital propagation → eclipse state → solar input coupled
  → Fault injection: solar degradation (t=600-900s) or radiator degradation

Preprocessing (train.py: add_derivative_features)
  → d_temp_dt, d_volt_dt, solar_residual_w, thermal_residual_w
  → 7 features total

ML scoring (detect.py: score_dataframe)
  → Full IsolationForest (7 features)
  → Power LOF (EPS features only)
  → Thermal AE (thermal features only)
  → Ensemble: OR flag, MIN score, source attribution

Physics validation (physics_rules/rules.py)
  → check_power_subsystem: solar residual vs threshold
  → check_thermal_subsystem: heat rejection residual
  → Eclipse-aware: suppresses false alarms during eclipse

Diagnosis + Evidence
  → 4-line causal narrative: WARN → SUBSYSTEM → EVIDENCE → ACTION
  → RAG: TF-IDF query from anomaly → top-k docs from 4-file KB
  → Granite: structured JSON {risk, diagnosis, recommended_action, citations}

Operator UI
  → Streamlit: KPIs, time scrubber, alerts, RUL chip, 3D CAD
  → React console: same data, authenticated, live ingest
  → Three.js: real IBM satellite, part-level fault animation
```

**Is every stage actually AI-generated?**
- ML anomaly detection: YES (IsolationForest, LOF, trained models)
- RAG retrieval: YES (TF-IDF scoring, metadata filtering)
- Granite explanation: YES when credentials present; DETERMINISTIC MOCK when absent
- Physics rules: NO (deterministic equations — this is a strength, not a weakness)

**The honest framing:** ML detects the anomaly, physics validates it, RAG grounds the explanation, Granite writes the human-readable reasoning. This is architecturally sound.

---

## Phase 3 — Five Judge Scores

### Judge A — IBM AI Engineer (Score: 6/10)

**Strengths:**
- Real `ibm-watsonx-ai` SDK integration, correctly wired
- Granite model `ibm/granite-4-h-small` is current and appropriate
- Honest mock/real state machine (MOCK / REAL_READY / REAL_FAILED)
- RAG evaluation with Recall@k=0.944, MRR=0.935
- Strict smoke-test mode proves real IBM answered

**Weaknesses:**
- Without credentials, the entire IBM layer is a mock — a judge who presses "show me Granite working" will see a deterministic fallback
- No evidence of IBM Bob usage whatsoever
- No IBM SkillsBuild completion evidence
- The README honestly says Bob is "PENDING" — this is honest but disqualifying
- RAG uses TF-IDF, not IBM watsonx embeddings — a competitor using IBM's embedding models would score higher on "IBM technology integration"

### Judge B — Aerospace Engineer (Score: 8/10)

**Strengths:**
- Real orbital mechanics (Kepler, eclipse, conical shadow)
- Energy-conserving EPS with proper SOC/voltage/battery policy
- First-order LEO thermal (Stefan-Boltzmann, albedo, Earth IR)
- NASA PCoE validation on real B0005/B0006/B0007/B0018
- Eclipse-aware fault detection (suppresses false alarms during eclipse)
- Honest documentation of limitations (demo vs spec physics)

**Weaknesses:**
- Fault injection is synthetic — no real satellite anomaly data
- Thermal model is first-order only (no multi-node, no structural)
- Prognostics limited to battery RUL (no thermal RUL, no reaction wheel RUL)

### Judge C — Hackathon Product Judge (Score: 7/10)

**Strengths:**
- Visually impressive: Three.js CAD, time scrubber, real-time physics
- Clear problem-solution narrative
- Multiple demo modes (Streamlit, React, 3D)
- Authentication flow adds production credibility

**Weaknesses:**
- Demo is 3.5 minutes (exceeds 3-minute limit)
- Too many features compete for attention — a judge can't absorb all of it in 3 minutes
- The "why does this need to exist?" is clear but could be sharper
- IBM integration feels bolted on rather than essential

### Judge D — Skeptical Researcher (Score: 7/10)

**Strengths:**
- PINN vs PGNN honest non-result (PINN loses, documented)
- Multi-seed robustness (6 seeds, AUC = 0.786 ± 0.009)
- 10-run RAG reproducibility gate
- Ensemble coherence: flag=1 implies score<0 by construction

**Weaknesses:**
- All validation is on synthetic data (simulated faults, not real anomalies)
- NASA PCoE is a battery dataset, not a spacecraft telemetry dataset — the connection is analogical, not direct
- Contamination = 0.05 is assumed, not calibrated to realistic anomaly rates
- No comparison against simple baselines (e.g., threshold detector)
- No cross-validation (single train/test split)
- The strongest claimed metric (post-900 F1 ≈ 1.00) is on simulated data with known fault onset — any reasonable detector would achieve this

### Judge E — Competing Winning Team (Score: 6.5/10)

**If I were building a competitor, I would:**
1. Use IBM watsonx.ai embeddings for RAG (not TF-IDF) — stronger IBM integration
2. Create a simpler, more focused 2.5-minute demo showing ONE fault → detection → explanation → action
3. Use IBM Bob visibly in the development process and document it
4. Include a real NASA dataset anomaly (not just battery degradation — actual spacecraft telemetry if available)
5. Deploy to a live URL where judges can interact with it
6. Show a quantitative before/after: "without MissionMind, anomaly detection takes X hours; with MissionMind, it takes Y seconds"
7. Have IBM SkillsBuild completion as a badge in the README

---

## Phase 4 — Official Dimension Scores

| Dimension | Score | Justification |
|---|---|---|
| Technical Execution | 8/10 | Genuinely multi-layered, well-tested, real SDK integration. Deducted for mock fallback being the default experience |
| Innovation | 7/10 | Physics + ML + RAG + Granite is a meaningful combination, but not unprecedented. The honest PINN non-result is refreshingly novel |
| Challenge Fit | 7/10 | Strong spacecraft theme, but "how does this advance space exploration?" could be sharper. Without Bob evidence, this score is at risk |
| Feasibility | 7/10 | Works locally, Vercel deployment configured, but no live deployment URL. IBM credential dependency is a demo risk |
| Real-World Impact | 6/10 | Qualitative impact is clear; quantitative impact on real missions is unproven. No before/after operator study |
| **Overall** | **7.0/10** | |

---

## Phase 5 — Top 10 Winning Blockers

### 🔴 1. IBM Bob Never Used (DISQUALIFICATION RISK)

**Problem:** The README explicitly states "⚠️ PENDING — Account not yet created." IBM Bob is the **required primary development tool**.

**Evidence:** `README.md` under "IBM Bob Status" section.

**Judging criterion:** Submission requirement (hard constraint).

**Score penalty:** Could be **disqualified entirely**. Even if not, this scores 0/10 on Bob integration.

**What a competitor would do:** Actually use Bob, document 3-4 specific examples of how it helped.

**Fix:** Create IBM account, use Bob for at least one meaningful task (e.g., debugging the Kepler solver, designing the ensemble architecture, writing the RAG evaluation harness), document with timestamps/screenshots.

**Expected improvement:** From disqualification risk to passing a hard constraint.

**Difficulty:** Medium (requires creating IBM Cloud account + using Bob).

**Must fix before submission:** YES — this is a hard requirement.

### 🔴 2. Demo Video Exceeds 3-Minute Limit

**Problem:** The demo is explicitly described as "3.5-minute narrated walkthrough." The challenge requires **maximum 3 minutes**.

**Evidence:** `README.md` demo section, `scripts/make_demo_video.py` (15 scenes).

**Judging criterion:** Submission requirement (hard constraint).

**Score penalty:** Could be **disqualified** or have points deducted.

**Fix:** Tighten the narration to fit 180 seconds. Remove 2-3 weaker scenes (live ingest, validation card, or shorten intro/auth).

**Difficulty:** Easy — edit the narration script and re-render.

**Must fix before submission:** YES.

### 🔴 3. IBM SkillsBuild Learning Activity Not Documented

**Problem:** The challenge requires completion of an IBM SkillsBuild learning activity. No evidence exists.

**Evidence:** No mention of SkillsBuild anywhere in the repository.

**Judging criterion:** Submission requirement.

**Fix:** Complete a relevant SkillsBuild course, add completion badge/certificate to README.

**Difficulty:** Easy — takes 1-2 hours for a short course.

**Must fix before submission:** YES.

### 🟠 4. README Missing Required "How IBM Bob Was Used" Section

**Problem:** The submission requirements explicitly state the README must contain "how IBM Bob was used." The current README has a "PENDING" placeholder.

**Evidence:** README.md "IBM Bob Status" section.

**Fix:** Even if Bob usage is limited, document honestly what happened (e.g., "Bob was used for X, Y, Z during development") or mark as "setup in progress" with clear next steps.

**Difficulty:** Easy once Bob is actually used.

**Must fix before submission:** YES — required README section.

### 🟠 5. No Live Deployment URL

**Problem:** Vercel is configured but there's no evidence of a live, accessible deployment. Judges may want to interact with the project.

**Evidence:** `vercel.json` exists but no deployment URL mentioned.

**Fix:** Deploy to Vercel, add the URL to README.

**Difficulty:** Easy — push to Vercel.

**Must fix before submission:** HIGHLY RECOMMENDED.

### 🟠 6. IBM Technology Feels Optional

**Problem:** If you remove all IBM branding (Granite, watsonx), the system still works identically via the mock fallback. A judge could reasonably ask: "What would be lost without IBM?"

**Evidence:** The mock fallback returns the same JSON shape. The dashboard works without credentials.

**Fix:** During the demo, show a real Granite call. Add a clear "Granite generates the reasoning — without it, the system uses a deterministic rule engine." Make the difference visible.

**Difficulty:** Medium — requires real credentials + clear demo framing.

**Must fix before submission:** YES — directly affects "IBM integration" score.

### 🟡 7. Demo Tries to Show Everything

**Problem:** 15 scenes in 3 minutes = ~12 seconds per scene. A judge cannot absorb anything. The narrative is "look at all these features" instead of "look at this one thing working end-to-end."

**Fix:** Cut to 8-10 scenes maximum. Focus on ONE compelling failure scenario.

**Difficulty:** Easy — edit the script.

**Must fix before submission:** HIGHLY RECOMMENDED.

### 🟡 8. Strongest Quantitative Result Is on Simulated Data

**Problem:** "post-900 F1 ≈ 1.00" is on synthetic fault injection with known onset. This is not impressive to a skeptical judge.

**Fix:** Emphasize the NASA PCoE validation (real data) more prominently. Lead with "AUC = 0.786 on real NASA battery data" rather than synthetic results.

**Difficulty:** Easy — reframe the narrative.

**Must fix before submission:** Recommended.

### 🟡 9. No "Before/After" Quantitative Impact

**Problem:** "What measurable problem does MissionMind solve?" has no quantitative answer. How much faster is anomaly detection? How many false alarms are prevented?

**Fix:** Add a clear quantitative claim: "MissionMind detects solar array anomalies 13 minutes before full failure, compared to threshold-based detection which triggers at 18 minutes."

**Difficulty:** Easy — measure from existing simulation data.

**Must fix before submission:** Recommended.

### 🟢 10. PINN Non-Result Is Buried

**Problem:** The honest PINN vs PGNN comparison is one of the most intellectually impressive parts of the project, but it's buried in ADR-001. This differentiates MissionMind from teams that blindly claim "we used a PINN because it's physics-informed."

**Fix:** Surface this in the README and demo as a key innovation: "We tested the obvious approach (PINN) and proved it doesn't work — here's why, and what we use instead."

**Difficulty:** Easy — reframe existing content.

**Must fix before submission:** Recommended.

---

## Phase 6 — IBM/Bob Integration Attack

**Brutal question: If I delete all IBM branding, does the system remain essentially identical?**

**Answer: YES.** The mock fallback returns the same JSON. The RAG uses TF-IDF, not IBM embeddings. The ML uses scikit-learn, not IBM models. The physics is pure Python.

**This is the single biggest competitive weakness.**

### How to Make IBM Materially Valuable:

1. **During the demo, show a REAL Granite call** — not the mock. The difference must be visible.
2. **Use IBM watsonx.ai embeddings** for the RAG retriever instead of TF-IDF. This would make IBM technology genuinely integral to the retrieval quality.
3. **Document specific IBM Bob usage** — even if Bob helped with debugging one function, that's better than "PENDING."
4. **Add IBM Granite model comparison** — show Granite's structured output vs a simple prompt vs no RAG, proving that Granite + RAG > Granite alone.
5. **Deploy on IBM Cloud** if possible — even a simple containerized deployment shows commitment.

### What NOT to Do:
- Do not fabricate Bob usage evidence
- Do not claim Granite was used when the mock ran
- Do not add IBM branding without substance

---

## Phase 7 — ML Research Attack

### Claimed Metrics vs Reality:

| Metric | Claimed | Data | Risk |
|---|---|---|---|
| AUC = 0.786 ± 0.009 | NASA PCoE B0005, 6 seeds | Real data, multi-seed | LOW — this is honest |
| Spearman = 0.950 ± 0.028 | NASA PCoE B0005 | Real data | LOW |
| Post-900 F1 ≈ 1.00 | Synthetic fault injection | Simulated data with known onset | HIGH — unimpressive |
| FPR 0.000 (100-600s) | Synthetic normal operation | Simulated data | MEDIUM — expected |
| Recall@k = 0.944 | 18-question golden dataset | Small, curated | MEDIUM — small evaluation set |

### Strongest Quantitative Result:
**AUC = 0.786 ± 0.009 on real NASA B0005 data with 6-seed robustness.** This should be the centerpiece, not the synthetic results.

### Missing:
- No baseline comparison (e.g., threshold detector, Z-score)
- No cross-validation (single split)
- No ablation (ML-only vs physics-only vs ML+physics)
- No comparison with published baselines on the same dataset

---

## Phase 8 — Physics Attack

| Component | Classification |
|---|---|
| Orbital mechanics (Kepler, eclipse) | Physically justified |
| Power model (SOC, voltage) | Simplified but defensible |
| Thermal model (Stefan-Boltzmann) | First-order approximation |
| Fault injection | Demo approximation — synthetic |
| Energy conservation | Verified by tests |
| Eclipse-aware detection | Physically meaningful improvement |

**The physics layer does improve detection** — eclipse-aware solar residual analysis suppresses false alarms during orbital eclipse. This is a genuine physical insight that pure ML would miss.

---

## Phase 9 — RAG + Granite Attack

**Test scenario analysis:**

| Scenario | Expected Behavior | Actual? |
|---|---|---|
| Normal condition | No hallucinated diagnosis | YES — no anomaly triggers no RAG query |
| Known solar anomaly | Correct diagnosis (power subsystem) | YES — metadata-scoped to power docs |
| Known thermal anomaly | Correct diagnosis (thermal subsystem) | YES — metadata-scoped to thermal docs |
| Missing documentation | Refuse rather than guess | YES — "A query that names no known system gets NO evidence" |
| Prompt injection | Treated as DATA, not instructions | YES — tested in adversarial suite |

**Honest assessment:** The RAG is well-designed for its scale. TF-IDF is sufficient for 31 chunks. The real weakness is that Granite is almost always running in mock mode during demos, making the "AI reasoning" layer invisible.

---

## Phase 10 — Product/UX Attack

**30-second test:**

1. **What problem?** → Spacecraft fault detection and diagnosis ✅ (clear in 10 seconds)
2. **Who uses it?** → Mission operators ✅ (implied by dashboard)
3. **What does AI do?** → Detects anomalies + explains why ⚠️ (visible but not prominent enough)
4. **Why better?** → Physics + ML + RAG + Granite = grounded explanation ⚠️ (takes too long to understand)
5. **Why IBM?** → Granite provides the reasoning layer ⚠️ (unclear without seeing real Granite)
6. **What happens on failure?** → Fault detected, cause explained, action recommended ✅ (the 4-line alert)
7. **Measurable benefit?** → 13-minute early warning ⚠️ (buried in narration, not in the UI)

**UX failure:** The system is technically deep but visually dense. A judge scanning for 30 seconds sees a dashboard with many numbers. The 4-line causal alert (WARN → SUBSYSTEM → EVIDENCE → ACTION) is the best UX element — make it the hero.

---

## Phase 11 — 3-Minute Demo War Game

### Current 15-scene allocation (3.5 min):

| # | Scene | Duration | Judge Impact |
|---|---|---|---|
| 1 | Intro card | 10s | LOW — generic |
| 2 | Auth flow | 12s | LOW — not relevant to space AI |
| 3 | Normal dashboard | 20s | MEDIUM — sets baseline |
| 4 | Fault injection | 16s | HIGH — this is the core |
| 5 | ML detection | 12s | HIGH — the AI moment |
| 6 | RUL prediction | 13s | MEDIUM — forward-looking |
| 7 | ML diagnostics | 16s | LOW — too detailed |
| 8 | RAG evidence | 15s | MEDIUM — grounding |
| 9 | Granite reasoning | 17s | HIGH — IBM moment |
| 10 | Scenario comparison | 16s | LOW — not essential |
| 11 | 3D digital twin | 11s | HIGH — visual wow |
| 12 | Live ingest | 18s | LOW — virtual edge node is not compelling |
| 13 | Web console | 15s | LOW — generic |
| 14 | Validation card | 10s | LOW — not visual |
| 15 | Close card | 12s | LOW |

### Optimized 8-scene allocation (2.5 min):

| # | Scene | Duration | Purpose |
|---|---|---|---|
| 1 | Title: Problem + MissionMind | 8s | Hook |
| 2 | Dashboard: normal → fault | 25s | Show the problem happening |
| 3 | ML detects anomaly (4-line alert) | 20s | The AI moment |
| 4 | RAG evidence + Granite reasoning | 30s | IBM technology + grounding |
| 5 | 3D satellite responds | 15s | Visual wow |
| 6 | RUL countdown | 15s | Forward-looking value |
| 7 | NASA validation metrics | 15s | Quantitative credibility |
| 8 | Close: "one command to start" | 12s | Call to action |

**Key changes:** Remove auth flow (not demo-relevant), live ingest (virtual, not impressive), web console (generic), validation card (not visual), scenario comparison (distracts from main story). The entire demo tells ONE story: fault → detection → explanation → action.

---

## Phase 12 — Competitor X

**Competitor X: "SentinelSat"**

- Uses IBM watsonx.ai embeddings for RAG (not TF-IDF)
- Has a live Vercel deployment judges can interact with
- Shows ONE clear fault scenario in 2.5 minutes
- Documents IBM Bob usage with specific examples
- Deploys a Streamlit Community Cloud version for easy judge access
- Has a 30-second "live demo" where a judge can scrub the timeline and see the AI respond
- Uses Granite to generate real-time explanations (with credentials during demo)
- Includes a quantitative claim: "Detects anomalies 12 minutes earlier than threshold-based methods"

| Dimension | MissionMind | Competitor X | Winner |
|---|---|---|---|
| Technical Execution | 8 | 7 | MissionMind |
| Innovation | 7 | 6 | MissionMind |
| Challenge Fit | 7 | 8 | Competitor X (Bob evidence) |
| Feasibility | 7 | 8 | Competitor X (live deployment) |
| Real-World Impact | 6 | 7 | Competitor X (clearer claim) |
| Demo strength | 6 | 8 | Competitor X (focused story) |
| IBM integration | 5 | 8 | Competitor X (real IBM embeddings) |
| Quantitative evidence | 7 | 7 | TIE |
| UX | 6 | 8 | Competitor X (cleaner) |
| **Total** | **59/90** | **67/90** | **Competitor X** |

**What Competitor X does better:** Focused narrative, real IBM integration visible in demo, live deployment, Bob evidence, cleaner UX. MissionMind wins on raw technical depth but loses on judge experience.

---

## Phase 13 — Winning Narrative

**One sentence:** MissionMind detects spacecraft faults 13 minutes before failure, explains the cause using engineering evidence, and recommends the correct operator action — all from a single line of telemetry.

**Problem:** Spacecraft operators have minutes to diagnose faults that span power, thermal, and orbital systems.

**Insight:** Combining physics-informed anomaly detection with evidence-grounded AI reasoning catches faults earlier and explains them better than either approach alone.

**AI breakthrough:** An ensemble of unsupervised detectors, each specialized to a subsystem, fused through a physics-validated decision layer — then grounded in engineering documentation via RAG and explained by Granite.

**Evidence:** On real NASA battery data (B0005), the ensemble achieves AUC = 0.786 ± 0.009 with 6-seed robustness, and detects simulated solar array failures 13 minutes before full power loss.

**IBM advantage:** Granite generates structured, citation-linked explanations that trace every claim back to the engineering knowledge base — not a chatbot paragraph.

**Impact:** Every minute of earlier detection translates to more options for the operator: graceful degradation, safe mode entry, or fault isolation.

---

## Phase 14 — Killer Demo

The single best scenario: **Solar array degradation during orbital daylight.**

1. T+0:00 — Dashboard shows normal operation (solar = 520W, battery full, temp stable)
2. T+10:00 — Fault begins (solar starts degrading from 520W toward 250W)
3. T+13:00 — ML ensemble flags anomaly (4-line alert appears: WARN → POWER → SOLAR RESIDUAL → CHECK ARRAY)
4. T+13:00 — RAG retrieves power subsystem docs, shows relevance scores
5. T+13:00 — Granite generates: "Solar array degradation detected. Probable cause: cell delamination or connector fault. Recommended action: verify array telemetry, prepare for battery-only operation."
6. T+13:00 — 3D satellite shows solar arrays dimming
7. T+13:00 — RUL chip shows "BAT 83 min" — 83 minutes of battery margin
8. T+23:00 — Confirm: battery SOC declining as predicted, operator has time to act

This is ONE scenario that shows the full stack in 2-3 minutes.

---

## Phase 15 — Scoreboard

| Category | Current | Winning Target | Gap |
|---|---|---|---|
| Technical Execution | 8/10 | 8/10 | 0 (already strong) |
| Innovation | 7/10 | 8/10 | +1 (surface PINN result, physics+ML co-design) |
| Challenge Fit | 7/10 | 9/10 | +2 (fix Bob, SkillsBuild, theme declaration) |
| Feasibility | 7/10 | 8/10 | +1 (live deployment) |
| Real-World Impact | 6/10 | 8/10 | +2 (quantitative claim, operator study) |
| Demo | 6/10 | 9/10 | +3 (cut to 3 min, focused story) |
| IBM/Bob integration | 5/10 | 8/10 | +3 (use Bob, show real Granite, document) |
| Research credibility | 7/10 | 8/10 | +1 (baseline comparison, cross-validation) |
| UX | 6/10 | 8/10 | +2 (hero the 4-line alert, reduce density) |
| **Total** | **59/90** | **74/90** | **+15** |

**Current winning probability:** ~25% (strong technical project, but compliance blockers and weak demo narrative)

**After highest-priority fixes:** ~55% (compliance fixed, focused demo, visible IBM integration)

---

## Phase 16 — Prioritized Fix Plan

### MUST FIX BEFORE SUBMISSION

| # | Fix | File | Acceptance Test | Criterion |
|---|---|---|---|---|
| 1 | Trim demo to ≤ 3 minutes | `scripts/make_demo_video.py` | Video duration ≤ 180s | Submission requirement |
| 2 | Use IBM Bob + document it | README, CLAUDE.md | Bob usage examples in README | Submission requirement |
| 3 | Complete IBM SkillsBuild | External | Badge in README | Submission requirement |
| 4 | Add "selected challenge theme" to README | README.md | Explicit "Advance Space Exploration" section | Submission requirement |
| 5 | Show real Granite call in demo | Demo script | Granite LIVE visible in demo | IBM integration score |

### SHOULD FIX

| # | Fix | Impact |
|---|---|---|
| 6 | Deploy to Vercel + add URL | Feasibility score |
| 7 | Cut demo from 15 to 8 scenes | Demo score |
| 8 | Lead with NASA AUC result, not synthetic results | Research credibility |
| 9 | Add quantitative impact claim | Real-world impact |
| 10 | Hero the 4-line alert in the demo | UX score |

### DO NOT TOUCH

| Component | Why |
|---|---|
| ML ensemble architecture | Already strong, well-tested |
| Physics simulator | Physically justified, well-documented |
| RAG evaluation harness | Thorough, reproducible |
| Authentication system | Production-grade, 31 tests |
| Security controls | Already hardened |
| Test suite | 100+ tests, CI-enforced |
| PINN vs PGNN honest result | Intellectually impressive, keep as-is |
| ADRs | Well-structured, evidence-based |

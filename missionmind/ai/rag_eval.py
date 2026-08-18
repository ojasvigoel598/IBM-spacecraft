"""
MissionMind - RAG evaluation harness.

The retrieval layer is evaluated INDEPENDENTLY of Granite: each golden
question carries the engineering evidence that SHOULD be retrieved from the
knowledge base, and the metrics below measure whether the retriever actually
finds it. A green final answer can never hide a retrieval failure - the
tests classify a miss as RETRIEVAL_FAILURE, not LLM_FAILURE.

Question types mirror the engineering needs of the system:

  factual   - direct questions answerable from one KB section
  telemetry - questions about telemetry variables / units / ranges
  anomaly   - fault-signature questions (solar / radiator degradation)
  physics   - questions about the modelled physical relationships
  rules     - mission-rule / threshold / action questions
  multi_hop - questions whose answer spans more than one section
  negative  - questions the KB genuinely cannot answer (no evidence)

Metrics (standard information retrieval):

  Recall@K     |retrieved_K ∩ expected| / |expected|
  Precision@K  |retrieved_K ∩ expected| / K
  MRR          mean reciprocal rank of the first expected hit
  nDCG@K       discounted cumulative gain, binary relevance, ideal ordering

Deterministic: TF-IDF + cosine similarity over a fixed corpus, no RNG, no
network, no credentials. Run from a fresh checkout with no .env at all.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

# ---- Golden dataset ---------------------------------------------------------
# Each entry: id, question, qtype, expected (ordered: most relevant first),
# and a short note on the engineering rationale. Expected IDs must exist in
# the knowledge base (asserted by the test suite).
GOLDEN_QUESTIONS: List[Dict] = [
    # --- factual ---
    {
        "id": "q-factual-bus-voltage",
        "question": "What is the nominal bus voltage and the battery voltage range?",
        "qtype": "factual",
        "expected": ["DOC-TELEMETRY-VOLT-001", "DOC-POWER-SUBSYSTEM-2"],
        "note": "Telemetry dictionary defines the units/range; Overview repeats it.",
    },
    {
        "id": "q-factual-solar-max",
        "question": "What is the maximum solar array output at full illumination?",
        "qtype": "factual",
        "expected": ["DOC-POWER-SUBSYSTEM-3", "DOC-TELEMETRY-SOLAR-001"],
        "note": "Normal Operation: 520W max at AM0; telemetry dict states nominal 520W.",
    },
    {
        "id": "q-factual-battery-capacity",
        "question": "What is the usable battery capacity of the spacecraft?",
        "qtype": "factual",
        "expected": ["DOC-POWER-SUBSYSTEM-3", "DOC-TELEMETRY-SOC-001"],
        "note": "Normal Operation: 100Wh usable; telemetry dict repeats it.",
    },
    # --- telemetry ---
    {
        "id": "q-telemetry-voltage-units",
        "question": "What are the units and expected range of battery_voltage_v?",
        "qtype": "telemetry",
        "expected": ["DOC-TELEMETRY-VOLT-001"],
        "note": "Telemetry dictionary: V, 28V at SOC 1, 24V at SOC 0.",
    },
    {
        "id": "q-telemetry-solar-units",
        "question": "What does solar_power_w measure and what is its nominal value?",
        "qtype": "telemetry",
        "expected": ["DOC-TELEMETRY-SOLAR-001"],
        "note": "Telemetry dictionary: solar array output power, W, nominal 520W.",
    },
    {
        "id": "q-telemetry-temperature-units",
        "question": "What is the radiator emissivity and area used in the thermal model?",
        "qtype": "telemetry",
        "expected": ["DOC-THERMAL-SUBSYSTEM-2"],
        "note": "Thermal Overview: epsilon 0.85, area 0.5 m2.",
    },
    # --- anomaly ---
    {
        "id": "q-anomaly-solar-signature",
        "question": "Solar power dropped below 364W while SOC is declining. What does this indicate?",
        "qtype": "anomaly",
        "expected": ["DOC-POWER-002"],
        "note": "Solar Array Degradation signature.",
    },
    {
        "id": "q-anomaly-radiator-signature",
        "question": "Temperature is rising while heat_in stays flat. What is the probable cause?",
        "qtype": "anomaly",
        "expected": ["DOC-THERM-002"],
        "note": "Radiator Degradation signature: temp slope with stable heat_in.",
    },
    # --- physics ---
    {
        "id": "q-physics-thermal-equilibrium",
        "question": "How is the radiator equilibrium temperature determined?",
        "qtype": "physics",
        "expected": ["DOC-THERMAL-SUBSYSTEM-2", "DOC-THERM-NOM-001"],
        "note": "Overview states the equilibrium condition Q_in = Q_out; Normal Operation gives the numeric result.",
    },
    {
        "id": "q-physics-net-power",
        "question": "What is the relationship between net power and battery SOC?",
        "qtype": "physics",
        "expected": ["DOC-POWER-SUBSYSTEM-3", "DOC-TELEMETRY-SOC-001"],
        "note": "Net +120W charges SOC to 1.0; telemetry dict: SOC declines with negative net power.",
    },
    # --- rules ---
    {
        "id": "q-rules-solar-threshold",
        "question": "At what solar power threshold is degradation suspected?",
        "qtype": "rules",
        "expected": ["DOC-MISSION-POWER-001"],
        "note": "Power Rules: solar < 364W (0.7*Pmax).",
    },
    {
        "id": "q-rules-safe-mode",
        "question": "When should the spacecraft consider safe mode for the battery?",
        "qtype": "rules",
        "expected": ["DOC-POWER-PROC-001"],
        "note": "Troubleshooting: if SOC < 0.2 enter safe mode.",
    },
    {
        "id": "q-rules-evidence-contract",
        "question": "What must every Granite output cite according to the evidence rules?",
        "qtype": "rules",
        "expected": ["DOC-EVIDENCE-001"],
        "note": "Evidence Requirements: current vs nominal numbers, doc ID, reasoning chain.",
    },
    # --- multi-hop ---
    {
        "id": "q-multihop-solar-recovery",
        "question": "Solar degradation is confirmed. What operational steps follow and what mission rule applies?",
        "qtype": "multi_hop",
        "expected": ["DOC-POWER-PROC-001", "DOC-MISSION-POWER-001"],
        "note": "Troubleshooting procedure + Power Rules action.",
    },
    {
        "id": "q-multihop-radiator-action",
        "question": "Radiator degradation is suspected. What mitigations are available and what is the risk limit?",
        "qtype": "multi_hop",
        "expected": ["DOC-THERM-PROC-001", "DOC-THERM-002"],
        "note": "Troubleshooting mitigations + degradation signature/limits.",
    },
    # --- negative (KB must NOT fabricate) ---
    {
        "id": "q-negative-launch-mass",
        "question": "What is the spacecraft launch mass?",
        "qtype": "negative",
        "expected": [],
        "note": "No KB section mentions launch mass - retrieval must return nothing confident.",
    },
    {
        "id": "q-negative-communications",
        "question": "Which document describes the UHF communications subsystem?",
        "qtype": "negative",
        "expected": [],
        "note": "No communications documentation exists in the KB.",
    },
    {
        "id": "q-negative-propulsion",
        "question": "What is the specific impulse of the propulsion system?",
        "qtype": "negative",
        "expected": [],
        "note": "The KB contains no propulsion information.",
    },
]


# ---- metrics ----------------------------------------------------------------

def recall_at_k(retrieved_ids: Sequence[str], expected: Sequence[str], k: int) -> float:
    if not expected:
        return 1.0  # no-answer question: nothing to recall
    hit = set(retrieved_ids[:k]) & set(expected)
    return len(hit) / len(set(expected))


def precision_at_k(retrieved_ids: Sequence[str], expected: Sequence[str], k: int) -> float:
    if k == 0:
        return 0.0
    hit = set(retrieved_ids[:k]) & set(expected)
    return len(hit) / k


def reciprocal_rank(retrieved_ids: Sequence[str], expected: Sequence[str]) -> float:
    """1/rank of the first expected hit (0 if absent). Expected may be ordered."""
    if not expected:
        return 1.0 if not retrieved_ids else 0.0
    expected_set = set(expected)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], expected: Sequence[str], k: int) -> float:
    """nDCG@k with binary relevance against the (ordered) expected list.

    Relevance of a retrieved id is 1 iff it is among the expected ids;
    the ideal ranking puts every expected id first in its expected order.
    """
    if not expected:
        return 1.0 if not retrieved_ids else 0.0
    expected_set = set(expected)
    rel = [1.0 if doc_id in expected_set else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = [1.0] * min(len(expected), k)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _default_k(question_type: str) -> int:
    # negative/no-answer questions are judged at k=5 (the retriever must not
    # return confident hits for them); everything else at k=3 (the app uses
    # top_k=3 for Granite evidence).
    return 5 if question_type == "negative" else 3


def evaluate_question(retriever, question: Dict, top_k: Optional[int] = None) -> Dict:
    """Retrieve for one golden question and return per-question metrics."""
    k = top_k or _default_k(question["qtype"])
    hits = retriever.retrieve(question["question"], top_k=max(k, 5))
    retrieved_ids = [d["id"] for d in hits]
    expected = question["expected"]
    result = {
        "id": question["id"],
        "qtype": question["qtype"],
        "question": question["question"],
        "expected": list(expected),
        "retrieved": retrieved_ids,
        "scores": [round(d.get("score", 0.0), 4) for d in hits],
        "k": k,
        "recall@k": round(recall_at_k(retrieved_ids, expected, k), 4),
        "precision@k": round(precision_at_k(retrieved_ids, expected, k), 4),
        "mrr": round(reciprocal_rank(retrieved_ids, expected), 4),
        "ndcg@k": round(ndcg_at_k(retrieved_ids, expected, k), 4),
    }
    # Failure taxonomy: a miss on a question that HAS expected evidence is a
    # retrieval failure - never blame the generator for it.
    if expected and not (set(retrieved_ids[:k]) & set(expected)):
        result["failure"] = "RETRIEVAL_FAILURE"
    elif not expected and retrieved_ids:
        # No-answer question returned confident hits: the corpus was not
        # able to refuse, which is a retrieval-level precision failure.
        result["failure"] = "NO_ANSWER_VIOLATION"
    else:
        result["failure"] = None
    return result


def evaluate_retrieval(retriever, questions: Optional[List[Dict]] = None,
                       top_k: Optional[int] = None) -> Dict:
    """Run the full golden evaluation; returns per-question rows + aggregates.

    Aggregates are reported overall and per question type so easy factual
    questions are never used to hide weak anomaly/multi-hop retrieval.
    """
    questions = questions if questions is not None else GOLDEN_QUESTIONS
    rows = [evaluate_question(retriever, q, top_k=top_k) for q in questions]

    def _agg(rows_subset: List[Dict]) -> Dict:
        if not rows_subset:
            return {}
        n = len(rows_subset)
        return {
            "n": n,
            "recall@k": round(sum(r["recall@k"] for r in rows_subset) / n, 4),
            "precision@k": round(sum(r["precision@k"] for r in rows_subset) / n, 4),
            "mrr": round(sum(r["mrr"] for r in rows_subset) / n, 4),
            "ndcg@k": round(sum(r["ndcg@k"] for r in rows_subset) / n, 4),
            "retrieval_failures": sum(1 for r in rows_subset if r["failure"] == "RETRIEVAL_FAILURE"),
            "no_answer_violations": sum(1 for r in rows_subset if r["failure"] == "NO_ANSWER_VIOLATION"),
        }

    by_type = {}
    for qtype in sorted({r["qtype"] for r in rows}):
        by_type[qtype] = _agg([r for r in rows if r["qtype"] == qtype])
    return {"rows": rows, "overall": _agg(rows), "by_type": by_type}


def format_report(result: Dict) -> str:
    """Human-readable report of an evaluate_retrieval() result."""
    lines = ["=== RAG retrieval evaluation ==="]
    lines.append(
        f"overall: recall@k={result['overall']['recall@k']} "
        f"precision@k={result['overall']['precision@k']} "
        f"mrr={result['overall']['mrr']} ndcg@k={result['overall']['ndcg@k']} "
        f"({result['overall']['n']} questions)")
    for qtype, agg in result["by_type"].items():
        lines.append(
            f"  {qtype:>9}: recall={agg['recall@k']} precision={agg['precision@k']} "
            f"mrr={agg['mrr']} ndcg={agg['ndcg@k']} "
            f"fail={agg['retrieval_failures']} noviol={agg['no_answer_violations']}")
    for row in result["rows"]:
        if row["failure"]:
            lines.append(
                f"  !! {row['id']} [{row['qtype']}] {row['failure']}: "
                f"expected={row['expected']} retrieved={row['retrieved'][:5]}")
    return "\n".join(lines)

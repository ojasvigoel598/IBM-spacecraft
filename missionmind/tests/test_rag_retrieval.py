"""
RAG retrieval evaluation + provenance regression tests.

These tests run with NO Granite/watsonx credentials and NO network: the
retrieval/evidence layer is validated independently of the generator. A
green generator answer can never hide a retrieval failure here - every
golden question either retrieves its expected evidence (Recall@k) or is a
negative question that must retrieve NOTHING (the system refuses).

Also pins the chunk-ID provenance contract: a citation ID must point at the
text that carries it (regression for the bug where every section chunk was
labeled with the Nth document-wide [DOC-...] ID).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.ai.rag import RAGRetriever, MIN_SCORE
from missionmind.ai.rag_eval import (
    GOLDEN_QUESTIONS, evaluate_retrieval, format_report,
)

RETRIEVER = RAGRetriever()
ID_TO_CHUNK = {d["id"]: d for d in RETRIEVER.documents}


# ---- golden evaluation ------------------------------------------------------

def test_golden_expected_ids_all_exist_in_kb():
    """The golden dataset must only reference chunks that actually exist -
    otherwise the evaluation would silently test nothing."""
    missing = []
    for q in GOLDEN_QUESTIONS:
        for doc_id in q["expected"]:
            if doc_id not in ID_TO_CHUNK:
                missing.append((q["id"], doc_id))
    assert not missing, f"golden questions reference missing chunk ids: {missing}"


def test_golden_evaluation_overall_recall():
    res = evaluate_retrieval(RETRIEVER)
    overall = res["overall"]
    assert overall["recall@k"] >= 0.90, format_report(res)
    assert overall["retrieval_failures"] == 0, format_report(res)
    assert overall["no_answer_violations"] == 0, format_report(res)


def test_golden_evaluation_every_group_recalls():
    """Easy factual questions must not hide weak anomaly/multi-hop retrieval."""
    res = evaluate_retrieval(RETRIEVER)
    for qtype, agg in res["by_type"].items():
        assert agg["recall@k"] >= 0.90, (
            f"group {qtype} recall {agg['recall@k']} below 0.90\n{format_report(res)}")


def test_negative_questions_refuse():
    """Out-of-scope engineering questions must return NO evidence."""
    for q in GOLDEN_QUESTIONS:
        if q["qtype"] != "negative":
            continue
        hits = RETRIEVER.retrieve(q["question"], top_k=5)
        assert hits == [], (
            f"negative question returned evidence: {q['id']} -> {[h['id'] for h in hits]}")


def test_anomaly_query_path_retrieves_ground_truth():
    """The production anomaly->query->retrieval path (used by the mock and the
    alert API) must retrieve the signature doc for each fault mode."""
    solar = RETRIEVER.query_from_anomaly({
        "subsystem": "power", "physics_flag": "solar_degradation",
        "current_values": {"solar_power_w": 248, "battery_voltage_v": 24.6, "soc": 0.31},
    }, top_k=3)
    assert solar and solar[0]["id"] == "DOC-POWER-002", [h["id"] for h in solar]
    radiator = RETRIEVER.query_from_anomaly({
        "subsystem": "thermal", "physics_flag": "radiator_degradation",
        "current_values": {"temperature_c": 42.5, "heat_in_w": 60, "heat_out_w": 32},
    }, top_k=3)
    ids = [h["id"] for h in radiator]
    assert "DOC-THERM-002" in ids or "DOC-THERM-PROC-001" in ids, ids


# ---- provenance / citation integrity ---------------------------------------

def test_chunk_id_points_at_its_own_text():
    """Regression: before the fix, DOC-POWER-002 pointed at the 'Normal
    Operation' section and DOC-PROC-GEN-001 at 'Risk Levels'. A citation ID
    must label the text that actually carries it."""
    assert "Solar Array Degradation" in ID_TO_CHUNK["DOC-POWER-002"]["content"]
    assert "stuck panel" in ID_TO_CHUNK["DOC-POWER-002"]["content"]
    assert "Check Sun sensor alignment" in ID_TO_CHUNK["DOC-POWER-PROC-001"]["content"]
    assert "Radiator Degradation" in ID_TO_CHUNK["DOC-THERM-002"]["content"]
    assert "stuck louver" in ID_TO_CHUNK["DOC-THERM-002"]["content"]
    assert "Verify internal power dissipation" in ID_TO_CHUNK["DOC-THERM-PROC-001"]["content"]
    assert "Generic Troubleshooting Flow" in ID_TO_CHUNK["DOC-PROC-GEN-001"]["content"]
    assert "Power Rules" in ID_TO_CHUNK["DOC-MISSION-POWER-001"]["content"]
    assert "Thermal Rules" in ID_TO_CHUNK["DOC-MISSION-THERM-001"]["content"]
    assert "Evidence Requirements" in ID_TO_CHUNK["DOC-EVIDENCE-001"]["content"]
    assert "State of charge" in ID_TO_CHUNK["DOC-TELEMETRY-SOC-001"]["content"]


def test_chunk_ids_are_unique():
    ids = [d["id"] for d in RETRIEVER.documents]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate chunk ids break citation provenance: {dupes}"


def test_every_chunk_has_source_and_system_metadata():
    for d in RETRIEVER.documents:
        assert d["source"], f"chunk {d['id']} missing source file"
        assert d["system"] in ("power", "thermal", "mission"), d
        assert os.path.basename(d["path"]) == d["source"]


def test_metadata_scope_prevents_cross_subsystem_retrieval():
    """A thermal query must never return power-only chunks and vice versa."""
    power_hits = RETRIEVER.retrieve(
        "solar array degradation battery SOC voltage", top_k=5)
    assert power_hits and all(h["system"] == "power" for h in power_hits), [
        h["id"] for h in power_hits]
    thermal_hits = RETRIEVER.retrieve(
        "radiator temperature heat rejection emissivity", top_k=5)
    assert thermal_hits and all(h["system"] == "thermal" for h in thermal_hits), [
        h["id"] for h in thermal_hits]


# ---- chunking preserves engineering information -----------------------------

def test_chunking_preserves_numbers_and_units():
    """Engineering chunks must keep units and numeric values verbatim."""
    solar_chunk = ID_TO_CHUNK["DOC-POWER-002"]["content"]
    # the degradation signature cites the 364W threshold and 0.7*P_max; the
    # 520W nominal lives in the Normal Operation section
    assert "364W" in solar_chunk and "0.7" in solar_chunk
    assert "520W" in ID_TO_CHUNK["DOC-POWER-SUBSYSTEM-3"]["content"]
    therm = ID_TO_CHUNK["DOC-THERMAL-SUBSYSTEM-2"]["content"]
    assert "0.85" in therm and "0.5 m2" in therm and "5000 J/K" in therm
    volt = ID_TO_CHUNK["DOC-TELEMETRY-VOLT-001"]["content"]
    assert "28 V" in volt and "24 V" in volt
    soc = ID_TO_CHUNK["DOC-TELEMETRY-SOC-001"]["content"]
    assert "100 Wh" in soc and "0.3" in soc and "0.2" in soc


def test_min_score_gate_is_reasonable():
    """The score gate must not be so loose that irrelevant chunks pass, nor
    so tight that genuine evidence is dropped."""
    assert 0.0 < MIN_SCORE <= 0.15, MIN_SCORE
    hits = RETRIEVER.retrieve("solar array degradation", top_k=3)
    assert hits and hits[0]["score"] >= 0.10, hits


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
    if failed:
        sys.exit(1)
    print("All rag-retrieval tests PASS")

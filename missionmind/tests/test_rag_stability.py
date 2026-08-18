"""
RAG stability: 10 consecutive clean runs.

The retriever is deterministic (TF-IDF + cosine over a fixed corpus, no RNG,
no network), so ten consecutive runs MUST produce byte-identical results.
Any nondeterminism here - from dict ordering, float noise, or singleton
state leaking between runs - is a real bug that could make retrieval
results unreproducible across judge sessions.

For the full 10-run validation across the ENTIRE RAG suite (including the
slow API test), see missionmind/ai/rag_validation.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from missionmind.ai.rag import RAGRetriever
from missionmind.ai.rag_eval import evaluate_retrieval, GOLDEN_QUESTIONS

N_RUNS = 10


def test_ten_consecutive_runs_are_identical():
    first = None
    for _ in range(N_RUNS):
        retriever = RAGRetriever()  # fresh instance every run
        result = evaluate_retrieval(retriever)
        if first is None:
            first = result
            continue
        assert result == first, (
            "RAG evaluation is nondeterministic between runs - retrieval "
            "results must be reproducible")


def test_ten_consecutive_runs_all_pass_acceptance():
    """Every one of the 10 runs must clear the same acceptance bar the
    retrieval suite enforces (overall recall >= 0.90, no failures, no
    no-answer violations)."""
    for _ in range(N_RUNS):
        retriever = RAGRetriever()
        result = evaluate_retrieval(retriever)
        overall = result["overall"]
        assert overall["recall@k"] >= 0.90, overall
        assert overall["retrieval_failures"] == 0, overall
        assert overall["no_answer_violations"] == 0, overall


if __name__ == "__main__":
    import json
    runs = []
    for i in range(N_RUNS):
        retriever = RAGRetriever()
        runs.append(evaluate_retrieval(retriever))
        print(f"run {i + 1}/{N_RUNS}: recall@k={runs[-1]['overall']['recall@k']} "
              f"fail={runs[-1]['overall']['retrieval_failures']} "
              f"noviol={runs[-1]['overall']['no_answer_violations']}")
    assert all(r == runs[0] for r in runs), "results differ between runs"
    print(f"ALL {N_RUNS} RUNS IDENTICAL - RAG evaluation is reproducible")

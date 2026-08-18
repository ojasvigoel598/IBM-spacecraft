"""
MissionMind - RAG validation runner.

Runs the ENTIRE RAG test suite N consecutive times (default 10) and requires
every run to be clean. This is the repository's reproducibility gate for the
RAG/retrieval layer:

    python -m missionmind.ai.rag_validation            # 10 full runs
    python -m missionmind.ai.rag_validation --runs 5
    python -m missionmind.ai.rag_validation --fast     # skip the slow API test

The suite needs NO Granite credentials and NO network: it validates
retrieval, chunking, provenance, telemetry grounding, adversarial defence,
Granite independence (modes A/B/C) and the production API path.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAG_TEST_FILES = [
    "missionmind/tests/test_rag_retrieval.py",
    "missionmind/tests/test_rag_telemetry_grounding.py",
    "missionmind/tests/test_rag_adversarial.py",
    "missionmind/tests/test_rag_granite_modes.py",
    "missionmind/tests/test_rag_stability.py",
    # the API test is part of the complete suite but slow (solves full
    # scenarios); excluded by --fast
    "missionmind/tests/test_api_server.py",
]


def run_once(files, quiet: bool) -> bool:
    cmd = [sys.executable, "-m", "pytest", "-q", *files]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if not quiet:
        print(result.stdout[-2000:] if result.returncode else result.stdout[-400:])
        if result.stderr.strip():
            print(result.stderr[-1500:])
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG validation runner")
    parser.add_argument("--runs", type=int, default=10,
                        help="consecutive clean runs required (default 10)")
    parser.add_argument("--fast", action="store_true",
                        help="skip the slow API-server test")
    args = parser.parse_args()

    files = RAG_TEST_FILES[:-1] if args.fast else RAG_TEST_FILES
    print(f"[rag-validation] {args.runs} consecutive runs over {len(files)} files")
    failed = []
    for i in range(args.runs):
        ok = run_once(files, quiet=False)
        mark = "CLEAN" if ok else "FAILED"
        print(f"[rag-validation] run {i + 1}/{args.runs}: {mark}")
        if not ok:
            failed.append(i + 1)
    if failed:
        print(f"[rag-validation] FAILED: {len(failed)}/{args.runs} runs not clean: "
              f"{failed}")
        return 1
    print(f"[rag-validation] PASS: {args.runs} consecutive clean runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

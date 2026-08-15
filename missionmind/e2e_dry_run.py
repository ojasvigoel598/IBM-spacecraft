"""
End-to-end dry run twice, no manual fixes between runs
Spec Row 13
"""
import os, sys, subprocess, time

# Use the same interpreter that launched this script (works from a venv: .venv/Scripts/python.exe)
PY = sys.executable

def run_cmd(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def main():
    print("=== MissionMind E2E Dry Run ===")
    for iteration in [1,2]:
        print(f"\n===== ITERATION {iteration} =====")
        run_cmd(f"{PY} -m missionmind.simulator.run_scenarios")
        run_cmd(f"{PY} -m missionmind.tests.test_physics")
        run_cmd(f"{PY} -m missionmind.physics_rules.test_rules")
        run_cmd(f"{PY} -m missionmind.ml.train")
        run_cmd(f"{PY} -m missionmind.ml.detect --input missionmind/data/run_normal.csv")
        run_cmd(f"{PY} -m missionmind.ml.detect --input missionmind/data/run_solar_failure.csv")
        run_cmd(f"{PY} -m missionmind.ml.detect --input missionmind/data/run_radiator_failure.csv")
        run_cmd(f"{PY} -m missionmind.ml.nasa_real_validation --quick")
        run_cmd(f"{PY} -m missionmind.ai.rag")
        run_cmd(f"{PY} -m missionmind.ai.granite_client")
        print(f"Iteration {iteration} PASS")
        time.sleep(1)

    print("\n=== E2E Dry Run DONE twice, no manual fixes ===")

if __name__ == "__main__":
    main()

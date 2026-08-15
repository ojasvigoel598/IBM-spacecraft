#!/usr/bin/env python3
"""MissionMind environment gate (project §24-26, §22).

Runs BEFORE the main application so the project never fails later with an
obscure import error. Verifies, in order:

    Python version
    Virtual environment (not global interpreter)
    Every REQUIRED third-party package  -> functional import test
    Jupyter + notebook kernel tooling
    Required data files, config files, model directories

Output follows the convention:

    ENVIRONMENT CHECK
    Python: PASS
    ...
    Missing:
    <name>
    Action:
    <exact command>

Exits 0 only when every required item passes. Optional items are reported
separately and never fail the gate.

Run:  python -m missionmind.check_environment
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
# (import_name, human_name, pip_package, required)
# `required=False` items are reported but never fail the gate.
PACKAGES: list[tuple[str, str, str, bool]] = [
    ("numpy", "NumPy", "numpy", True),
    ("pandas", "Pandas", "pandas", True),
    ("sklearn", "scikit-learn", "scikit-learn", True),
    ("joblib", "Joblib", "joblib", True),
    ("streamlit", "Streamlit", "streamlit", True),
    ("plotly", "Plotly", "plotly", True),
    ("ibm_watsonx_ai", "IBM watsonx.ai SDK", "ibm-watsonx-ai", True),
    ("scipy", "SciPy", "scipy", True),
    ("pyod", "PyOD", "pyod", True),
    ("xgboost", "XGBoost", "xgboost", True),
    ("torch", "PyTorch", "torch", True),
    ("matplotlib", "Matplotlib", "matplotlib", True),
    ("nbformat", "nbformat", "nbformat", True),
    ("nbclient", "nbclient", "nbclient", True),
    ("nbconvert", "nbconvert", "nbconvert", True),
    ("ipykernel", "ipykernel", "ipykernel", True),
    ("paho.mqtt", "paho-mqtt", "paho-mqtt", True),  # telemetry edge MQTT
    ("fastapi", "FastAPI", "fastapi", True),   # viz/api_server.py
    ("uvicorn", "uvicorn", "uvicorn", True),
    ("pydantic", "pydantic", "pydantic", True),
    # Optional / tooling only — reported, never fails the gate
    ("dotenv", "python-dotenv (optional)", "python-dotenv", False),
    ("PIL", "Pillow (optional)", "pillow", False),
    ("requests", "Requests (optional)", "requests", False),
    ("gmsh", "Gmsh (CAD STEP/STL export)", "gmsh", False),
]

# (relative_path, kind) — kind in {"data", "config", "model-dir"}
REQUIRED_PATHS: list[tuple[str, str]] = [
    ("missionmind/data/nasa_battery_sample.csv", "data"),
    ("missionmind/data/grounded_parameters.json", "config"),
    ("missionmind/ml/train.py", "source"),
    ("missionmind/ml/detect.py", "source"),
    ("missionmind/physics_rules/rules.py", "source"),
    ("missionmind/simulator/config.py", "source"),
    ("missionmind/ai/rag.py", "source"),
    ("missionmind/ai/knowledge_base/power_subsystem.md", "data"),
    ("missionmind/ai/knowledge_base/thermal_subsystem.md", "data"),
    ("missionmind/ai/knowledge_base/mission_rules.md", "data"),
]

MODEL_DIRS: list[tuple[str, str]] = [
    ("missionmind/models", "model-dir"),  # ml/train.py: MODEL_DIR
]

PYTHON_MIN = (3, 10)


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= PYTHON_MIN
    msg = f"Python {v.major}.{v.minor}.{v.micro} (min {PYTHON_MIN[0]}.{PYTHON_MIN[1]})"
    if not ok:
        msg += f" — too old"
    return ok, msg


def _check_venv() -> tuple[bool, str]:
    # A venv has a `pyvenv.cfg` in its home and sys.prefix != sys.base_prefix
    # (or the base is a virtual env too).
    prefix = Path(sys.prefix)
    is_venv = (prefix / "pyvenv.cfg").exists() or sys.prefix != sys.base_prefix
    msg = f"venv={prefix.name} prefix={sys.prefix}"
    return bool(is_venv), msg


def _check_package(imp: str, human: str, pip: str, required: bool) -> tuple[bool, str]:
    """Functional import test: importing is the test. Extra attribute probes
    for the heaviest libraries so a broken wheel cannot pass silently."""
    try:
        mod = importlib.import_module(imp)
    except Exception as e:  # noqa: BLE001 — report any failure loudly (§25)
        return False, f"{human}: {type(e).__name__}: {e}"
    ver = getattr(mod, "__version__", "?")
    detail = f"{human} {ver}"
    # functional probes
    if imp == "numpy":
        import numpy as np

        assert np.abs(np.array([-1.0, 2.0]).sum()) > 0
        detail += " (arith ok)"
    elif imp == "pandas":
        import pandas as pd

        _ = pd.DataFrame({"a": [1, 2]}).a.mean()
        detail += " (frame ok)"
    elif imp == "sklearn":
        from sklearn.ensemble import IsolationForest

        IsolationForest(n_estimators=2, random_state=0)
        detail += " (IF ok)"
    elif imp == "torch":
        import torch

        t = torch.tensor([1.0, 2.0], requires_grad=True)
        (t * t).sum().backward()
        assert t.grad is not None and float(t.grad.sum()) > 0
        detail += f" (autograd ok, {torch.get_num_threads()} thr)"
    elif imp == "paho.mqtt":
        import paho.mqtt.client as mqtt

        assert hasattr(mqtt, "Client")
        detail += " (client class ok)"
    elif imp == "ibm_watsonx_ai":
        detail += " (SDK importable)"
    return True, detail


def _check_jupyter() -> tuple[bool, str]:
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().get_all_specs()
        names = sorted(specs.keys())
        ok = len(names) > 0
        msg = f"kernels: {', '.join(names) or 'NONE'}"
        return ok, msg
    except Exception as e:  # noqa: BLE001
        return False, f"jupyter_client: {type(e).__name__}: {e}"


def main() -> int:
    rows: list[tuple[str, bool, str]] = []
    failures: list[str] = []

    def row(name: str, ok: bool, detail: str = "") -> None:
        rows.append((name, ok, detail))
        if not ok:
            failures.append(name)

    print("=" * 62)
    print("ENVIRONMENT CHECK — MissionMind")
    print("=" * 62)

    ok, msg = _check_python()
    row("Python", ok, msg)
    ok, msg = _check_venv()
    row("Virtual environment", ok, msg)
    ok, msg = _check_jupyter()
    row("Jupyter kernels", ok, msg)

    for imp, human, pip, required in PACKAGES:
        ok, msg = _check_package(imp, human, pip, required)
        if required:
            row(human, ok, msg)
            if not ok:
                failures.append(f"pip install {pip}")
        else:
            # optional: report but do not fail the gate
            rows.append((human + " (optional)", ok, msg))

    # files & dirs
    missing_paths: list[str] = []
    for rel, kind in REQUIRED_PATHS:
        p = ROOT / rel
        exists = p.exists()
        row(f"{kind}: {rel}", exists, "found" if exists else "MISSING")
        if not exists:
            missing_paths.append(rel)
    for rel, _ in MODEL_DIRS:
        p = ROOT / rel
        exists = p.is_dir()
        row(f"model-dir: {rel}", exists, "found" if exists else "MISSING")
        if not exists:
            missing_paths.append(rel + "/")

    print("-" * 62)
    n_pass = sum(1 for _, ok, _ in rows if ok)
    n_fail = sum(1 for _, ok, _ in rows if not ok)
    print(f"PASS {n_pass}   FAIL {n_fail}")

    if failures:
        print()
        print("Missing:")
        for f in failures:
            print(f"  {f}")
        print()
        print("Action:")
        if any("pip install" in f for f in failures):
            print("  .venv/Scripts/python.exe -m pip install <package>")
        if missing_paths:
            print("  Restore missing files (git checkout / regenerate):")
            for mp in missing_paths:
                print(f"    {mp}")
        print()
        print("ENVIRONMENT CHECK: FAIL")
        return 1

    print()
    print("ENVIRONMENT CHECK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

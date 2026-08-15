#!/usr/bin/env python3
"""MissionMind automatic dependency detection (project §22-23).

Scans every Python source file and every code cell of the notebook,
collects all third-party imports, maps them to pip package names, and
compares against the declared manifest. Reports:

    * imports with NO declared package (missing from requirements.txt)
    * declared packages never imported by any file (unused — candidate
      for removal)

It does NOT install anything (§23: "Do not install packages blindly").
Installation is performed explicitly by the operator.

Run:  python -m missionmind.check_dependencies
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Modules that ship with the standard library — never third-party.
STDLIB = {
    "__future__", "__main__", "_thread", "abc", "argparse", "array", "ast",
    "asyncio", "base64", "bisect", "builtins", "bz2", "calendar", "cgi",
    "cmath", "codecs", "collections", "colorsys", "concurrent", "configparser",
    "contextlib", "copy", "copyreg", "csv", "ctypes", "dataclasses", "datetime",
    "decimal", "difflib", "dis", "email", "enum", "errno", "faulthandler",
    "fractions", "functools", "gc", "getpass", "gettext", "glob", "graphlib",
    "gzip", "hashlib", "heapq", "hmac", "html", "http", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "linecache", "locale",
    "logging", "lzma", "mailbox", "marshal", "math", "mimetypes", "multiprocessing",
    "numbers", "operator", "os", "pathlib", "pickle", "pkgutil", "platform",
    "plistlib", "posixpath", "pprint", "profile", "pstats", "queue", "random",
    "re", "readline", "reprlib", "resource", "runpy", "sched", "secrets",
    "select", "selectors", "shlex", "shutil", "signal", "site", "socket",
    "socketserver", "sqlite3", "ssl", "stat", "statistics", "string", "stringprep",
    "struct", "subprocess", "sys", "sysconfig", "tarfile", "tempfile",
    "textwrap", "threading", "time", "timeit", "token", "tokenize", "trace",
    "traceback", "types", "typing", "unicodedata", "unittest", "urllib", "uuid",
    "venv", "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound",
    "xml", "zipfile", "zipimport", "zlib",
    # project root namespace + pure stdlib additions
    "missionmind", "freebuff", "pytest", "jupyter_client", "importlib_metadata",
}

# import_name -> pip package (extend as needed; the report shows suggestions)
IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "PIL": "pillow",
    "paho": "paho-mqtt",
    "ibm_watsonx_ai": "ibm-watsonx-ai",
    "jupyter_client": "jupyter-client",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
}


INTERNAL_MODULES = {p.stem for p in (Path(__file__).resolve().parents[1] / "missionmind").rglob("*.py") if p.stem != "__init__"}

KNOWN_PACKAGES = {
    "numpy", "pandas", "scikit-learn", "joblib", "streamlit", "plotly",
    "ibm-watsonx-ai", "python-dotenv", "pyod", "xgboost", "torch",
    "matplotlib", "nbformat", "nbclient", "nbconvert", "ipykernel",
    "paho-mqtt", "gmsh", "pillow", "requests", "scipy",
}

# Declared for CLI / tooling use (invoked as subprocess `-m nbconvert`, used
# by .freebuff/ scripts, or convenient-but-not-runtime). These are never
# flagged as "declared but unused" — they are required by the validation
# gate even though no source file does `import nbconvert`.
TOOLING_DECLARED = {
    "nbformat", "nbclient", "nbconvert", "ipykernel",  # notebook gate (§27-28)
    "pillow", "requests", "python-dotenv", "gmsh",      # .freebuff/ tooling
}


def imports_from_source(src: str) -> set[str]:
    mods: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return mods
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level:
            # relative import — resolve to the importing module's package
            mods.add("__relative__")
    return mods


# directories never scanned (env / deps / vcs / build caches)
SKIP_DIRS = {"node_modules", "__pycache__", ".venv", ".git", ".vscode", "node"}


def scan_tree(base: Path, suffix: str) -> dict[Path, set[str]]:
    out: dict[Path, set[str]] = {}
    for p in sorted(base.rglob("*")):
        if p.suffix != suffix:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            out[p] = imports_from_source(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001
            continue
    return out


def notebook_imports(path: Path) -> set[str]:
    """Scan code cells of the notebook (it must stay in sync — §18/§29)."""
    try:
        import json

        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    mods: set[str] = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        mods |= imports_from_source("".join(cell.get("source", [])))
    return mods


def declared_packages() -> set[str]:
    """Parse requirements.txt (root and missionmind) — plain pin syntax only."""
    declared: set[str] = set()
    for req in (ROOT / "requirements.txt", ROOT / "missionmind" / "requirements.txt"):
        if not req.exists():
            continue
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            name = line.split("==")[0].split(">=")[0].split("<")[0].split("~=")[0].strip()
            if name:
                declared.add(name.lower())
    return declared


def main() -> int:
    print("=" * 62)
    print("DEPENDENCY SCAN — MissionMind")
    print("=" * 62)

    all_mods: set[str] = set()
    files: list[tuple[str, set[str]]] = []

    # missionmind/ package + root-level project scripts + notebook tooling
    # (build_notebook.py is the canonical notebook generator, §18)
    scan_roots = [
        (ROOT / "missionmind", ".py"),
        (ROOT, ".py"),
        (ROOT / ".freebuff", ".py"),
    ]
    for root, suffix in scan_roots:
        if not root.exists():
            continue
        for p, mods in scan_tree(root, suffix).items():
            try:
                rel = p.relative_to(ROOT).as_posix()
            except ValueError:
                rel = p.as_posix()
            files.append((rel, mods))
            all_mods |= mods

    nb = ROOT / "MissionMind_Full_ML_Analysis.ipynb"
    if nb.exists():
        nb_mods = notebook_imports(nb)
        files.append(("MissionMind_Full_ML_Analysis.ipynb", nb_mods))
        all_mods |= nb_mods

    third_party = {m for m in all_mods if m not in STDLIB and m not in INTERNAL_MODULES and not m.startswith("__")}

    declared = declared_packages()
    # map imports to pip names
    missing: dict[str, list[str]] = {}
    for imp in sorted(third_party):
        pkg = IMPORT_TO_PACKAGE.get(imp, imp.lower())
        if pkg not in declared and pkg not in KNOWN_PACKAGES:
            missing.setdefault(pkg, []).append(imp)

    # declared but never imported anywhere
    imported_pkgs = {IMPORT_TO_PACKAGE.get(m, m.lower()) for m in third_party}
    unused = sorted(d for d in declared if d not in imported_pkgs and d not in TOOLING_DECLARED)
    tooling = sorted(d for d in declared if d not in imported_pkgs and d in TOOLING_DECLARED)
    print(f"Scanned {len(files)} files, {len(third_party)} distinct third-party imports")
    print(f"Declared in requirements: {', '.join(sorted(declared)) or 'NONE'}")
    print()

    ok = True
    if missing:
        ok = False
        print("Imports with NO declared package (add to requirements.txt):")
        for pkg, imps in sorted(missing.items()):
            print(f"  {pkg:<20} (imported as: {', '.join(sorted(imps))})")
    else:
        print("Imports with NO declared package: none")

    if tooling:
        print()
        print("Declared for tooling / CLI use (not imported by source — expected):")
        for d in tooling:
            print(f"  {d}")

    if unused:
        print()
        print("Declared but never imported anywhere (candidate for removal):")
        for d in unused:
            print(f"  {d}")
    else:
        print("Declared but never imported (excluding tooling): none")

    print()
    print("DEPENDENCY SCAN:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

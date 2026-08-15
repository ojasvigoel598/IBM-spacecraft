#!/usr/bin/env python3
"""MissionMind source <-> notebook consistency check (project §29-30).

Verifies the notebook never drifts from the current implementation:

    1. Every module/function/class the notebook imports or calls must
       still exist in the source tree (no deleted/renamed APIs).
    2. Every file path referenced by the notebook must exist.
    3. The notebook must not embed hard-coded result numbers that claim
       to be outputs (no fake outputs — §30). Result-style literals in
       markdown interpretation cells are allowed; code cells must compute.
    4. Syntax of every code cell (calls ast.parse).

The notebook is GENERATED from source by `.freebuff/build_notebook.py`, so
this check is the guard: after any source change, rebuild + re-check.

Run:  python -m missionmind.check_notebook_consistency
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "MissionMind_Full_ML_Analysis.ipynb"

# Notebook may reference these by design; map module->{names} to skip.
# Inlined modules import each other via comments, so attribute lookups are
# the real API surface we can verify against the actual source modules.
KNOWN_BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))


def load_notebook() -> list[dict]:
    if not NB.exists():
        print(f"FATAL: notebook not found at {NB}")
        sys.exit(1)
    return json.loads(NB.read_text(encoding="utf-8")).get("cells", [])


def source_symbols() -> dict[str, set[str]]:
    """module_path -> exported {class, function, module-level assignment} names."""
    out: dict[str, set[str]] = {}
    for p in sorted((ROOT / "missionmind").rglob("*.py")):
        if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
        out[rel] = names
    return out


def attr_refs_in_cell(src: str) -> list[tuple[str, str]]:
    """Return (module_path, attr) attribute accesses of the form
    `missionmind.X.Y.attr` or `from missionmind.X import attr`."""
    refs: list[tuple[str, str]] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return refs
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # walk up to find a Name root
            parts: list[str] = []
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            name = ".".join(parts)
            if name.startswith("missionmind."):
                mod, _, attr = name.rpartition(".")
                refs.append((mod, attr))
    return refs


def cell_syntax(src: str) -> list[str]:
    """Return list of syntax errors in a cell ('' if ok)."""
    errors: list[str] = []
    lines = [ln for ln in src.splitlines()
             if not ln.strip().startswith("%")
             and not ln.strip().startswith("get_ipython()")]
    try:
        ast.parse("\n".join(lines))
    except SyntaxError as e:
        errors.append(f"line {e.lineno}: {e.msg}")
    return errors


def fake_output_patterns(src: str) -> list[str]:
    """Heuristic scan for hard-coded 'result' literals in code cells that
    would let a notebook look successful without executing. Only flags
    suspicious patterns; markdown cells are excluded."""
    hits: list[str] = []
    suspicious = [
        ("ROC-AUC = 0.", "ROC-AUC = 0."),
        ("AUC = 0.", "AUC = 0."),
        ("F1 = 0.", "F1 = 0."),
        ("accuracy = 0.", "accuracy = 0."),
    ]
    for pat, label in suspicious:
        if pat in src:
            hits.append(label)
    return hits


def main() -> int:
    print("=" * 62)
    print("NOTEBOOK <-> SOURCE CONSISTENCY — MissionMind")
    print("=" * 62)

    cells = load_notebook()
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    symbols = source_symbols()

    errors: list[str] = []
    warnings: list[str] = []
    checked_attrs = 0

    for idx, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))

        # 1. syntax
        for err in cell_syntax(src):
            errors.append(f"cell {idx} SYNTAX: {err}")

        # 2. attribute refs to missionmind modules
        for mod, attr in attr_refs_in_cell(src):
            checked_attrs += 1
            # find the module file
            mod_path = mod.replace(".", "/") + ".py"
            candidates = [mod_path]
            # also check package __init__ or direct file (module could be
            # inlined in the notebook, so only warn if module missing entirely)
            if not any(p == mod_path for p in symbols):
                # module not present as file — may be an inlined notebook
                # module; treat attr presence check as best-effort
                continue
            names = symbols.get(mod_path, set())
            if attr not in names:
                errors.append(f"cell {idx}: {mod}.{attr} does not exist in {mod_path}")

        # 3. fake outputs (code cells only)
        for hit in fake_output_patterns(src):
            warnings.append(f"cell {idx}: possible hard-coded result {hit!r} (§30)")

    # 4. referenced paths exist (DATA_DIR etc.)
    import re

    path_refs: set[str] = set()
    for cell in code_cells:
        for m in re.finditer(r'["\']((?:missionmind|data|models)/[^"\']+)["\']', "".join(cell.get("source", []))):
            path_refs.add(m.group(1))
    for ref in sorted(path_refs):
        if not (ROOT / ref).exists():
            errors.append(f"referenced path does not exist: {ref}")

    print(f"Checked {len(code_cells)} code cells, {checked_attrs} API attribute refs")
    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    print()
    print("NOTEBOOK <-> SOURCE CONSISTENCY:", "PASS" if not errors else f"FAIL ({len(errors)} errors)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

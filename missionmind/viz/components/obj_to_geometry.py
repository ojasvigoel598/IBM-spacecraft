#!/usr/bin/env python3
"""Convert the IBM satellite OBJ (Autodesk ATF STEP export) into a compact
``satellite_geometry.js`` module for Three.js.

Why not load the OBJ directly in the browser? The dashboard embeds its 3D view
inside Streamlit's components.html, where the OBJ cannot be fetched (CORS), and
the raw OBJ is 9 MB of mostly-redundant per-corner data. This converter:

  * keeps only positions + normals (no UVs - the CAD has no textures),
  * deduplicates per-corner (v, vn) pairs so the mesh is properly indexed,
  * triangulates any quad faces,
  * reads the part colours from the ATF material names (``SC_210_215_225`` ->
    ``#d2d7e1``),
  * centres the model and normalises it to a 1.0-unit bounding box,
  * rounds floats to 4 decimals (0.1 mm at metre scale) to keep the JS small.

Usage (from this directory)::

    python obj_to_geometry.py models/ibm_satellite.obj [satellite_geometry.js]

The output file is generated; do not edit it by hand.
"""

import json
import re
import sys
from pathlib import Path

RE_V = re.compile(r"^v\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*$")
RE_VN = re.compile(r"^vn\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*$")
RE_FACE = re.compile(r"^f\s+(.+?)\s*$")
RE_MAT = re.compile(r"^([A-Za-z]+)_(\d+)_(\d+)_(\d+)$")


def parse_color(name):
    """ATF material names like 'SC_210_215_225' encode an RGB colour."""
    m = RE_MAT.match(name.strip())
    if m and all(0 <= int(x) <= 255 for x in m.groups()[1:]):
        r, g, b = (int(x) for x in m.groups()[1:])
        return "#%02x%02x%02x" % (r, g, b)
    return "#cfd6dd"  # neutral silver fallback


def convert(src: Path, out: Path) -> None:
    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []
    # (group, material) -> {pos, nrm, idx, corner_map}
    parts: dict[tuple[str, str], dict] = {}
    cur_group, cur_mat = "Body", "Default"

    bounds = [float("inf")] * 3 + [float("-inf")] * 3

    with open(src, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("v "):
                m = RE_V.match(line)
                if not m:
                    continue
                v = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                verts.append(v)
                for i in range(3):
                    bounds[i] = min(bounds[i], v[i])
                    bounds[i + 3] = max(bounds[i + 3], v[i])
            elif line.startswith("vn "):
                m = RE_VN.match(line)
                if m:
                    norms.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            elif line.startswith("g "):
                cur_group = line[2:].strip() or "Body"
            elif line.startswith("usemtl "):
                cur_mat = line[7:].strip() or "Default"
            elif line.startswith("f "):
                m = RE_FACE.match(line)
                if not m:
                    continue
                tokens = m.group(1).split()
                if len(tokens) < 3:
                    continue
                # OBJ faces may be quads/polygons -> fan triangulate around vertex 0:
                # (0,1,2),(0,2,3),(0,3,4)... (the naive (k,k+1,k+2) scan would produce
                # overlapping triangles for quads+)
                for k in range(1, len(tokens) - 1):
                    a, b, c = tokens[0], tokens[k], tokens[k + 1]
                    if a == b or b == c or a == c:
                        continue
                    key = (cur_group, cur_mat)
                    part = parts.setdefault(
                        key, {"pos": [], "nrm": [], "idx": [], "cmap": {}}
                    )
                    for tok in (a, b, c):
                        # tok = v/vt/vn  (vt optional)
                        fields = tok.split("/")
                        vi = int(fields[0]) - 1
                        nidx = int(fields[2]) - 1 if len(fields) > 2 and fields[2] else vi
                        if vi < 0 or vi >= len(verts):
                            continue
                        corner = (vi, nidx)
                        cid = part["cmap"].get(corner)
                        if cid is None:
                            cid = len(part["pos"])
                            part["cmap"][corner] = cid
                            v = verts[vi]
                            part["pos"].extend((v[0], v[1], v[2]))
                            if 0 <= nidx < len(norms):
                                n = norms[nidx]
                                part["nrm"].extend((n[0], n[1], n[2]))
                            else:
                                part["nrm"].extend((0.0, 0.0, 1.0))
                        part["idx"].append(cid)

    if not verts:
        raise SystemExit("No vertices parsed - is this really an OBJ file?")

    # Centre + normalise to a 1.0-unit bounding box.
    cx = (bounds[0] + bounds[3]) / 2
    cy = (bounds[1] + bounds[4]) / 2
    cz = (bounds[2] + bounds[5]) / 2
    size = max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2]) or 1.0

    def r4(x):
        return round(x, 4)

    lines = ["// Generated by obj_to_geometry.py - do not edit by hand.",
             f"// Source: {src.name}  ({len(parts)} parts)",
             "const SATELLITE_GEOMETRY = {",
             f"  center: [{r4(cx)}, {r4(cy)}, {r4(cz)}],",
             f"  size: {r4(size)},",
             "  parts: ["]
    for (group, mat), part in parts.items():
        name = f"{group}/{mat}"
        color = parse_color(mat)
        n_verts = len(part["pos"]) // 3
        lines.append("    {")
        lines.append(f'      name: {json.dumps(name)},')
        lines.append(f'      color: {json.dumps(color)},')
        lines.append("      positions: [" + ",".join(str(r4(x)) for x in part["pos"]) + "],")
        lines.append("      normals: [" + ",".join(str(r4(x)) for x in part["nrm"]) + "],")
        lines.append("      indices: [" + ",".join(str(i) for i in part["idx"]) + "],")
        lines.append("      vertexCount: %d," % n_verts)
        lines.append("    },")
    lines.append("  ]")
    lines.append("};")
    lines.append("if (typeof module !== 'undefined') module.exports = SATELLITE_GEOMETRY;")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(parts)} parts, "
          f"{sum(len(p['pos']) // 3 for p in parts.values())} vertices, "
          f"{sum(len(p['idx']) for p in parts.values())} indices)")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "models" / "ibm_satellite.obj"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "satellite_geometry.js"
    convert(src, out)

"""Convert the real Fusion-exported satellite OBJ into STL + STEP for GitHub.

STL and STEP are the two formats the user wants in the repo so visitors can
view/download the actual spacecraft CAD without the code.  GitHub's built-in
3D viewer renders STL (and OBJ) directly; STEP is the standard exchange
format for CAD tools.

gmsh imports the OBJ triangle mesh and writes:
  - STL  (binary)  - directly viewable on GitHub
  - STEP (BRep)    - gmsh re-exports the mesh as a BRep shell so CAD tools
                     (Fusion, SolidWorks, FreeCAD) can open it

Run:
    .venv/Scripts/python.exe missionmind/viz/components/obj_to_step_stl.py
"""
import os
import sys

import gmsh

ROOT = os.path.dirname(os.path.abspath(__file__))
OBJ = os.path.join(ROOT, "models", "ibm_satellite.obj")
OUT_STL = os.path.join(ROOT, "models", "ibm_satellite.stl")
OUT_STEP = os.path.join(ROOT, "models", "ibm_satellite.step")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("Mesh.Binary", 1)

gmsh.open(OBJ)
gmsh.model.occ.synchronize()

# --- STL: mesh stays as-is, write binary ---
gmsh.write(OUT_STL)

# --- STEP: convert the surface mesh into a BRep shell ---
# gmsh.model.occ.addSurfaceLoop requires a closed surface; a satellite CAD
# mesh from Fusion is usually a closed manifold.  Use the occ kernel to
# fragment and export as one solid shell; fall back to plain write() which
# gmsh handles for surface meshes too.
try:
    dims = gmsh.model.getEntities()
    surfaces = [e for e in dims if e[0] == 2]
    loops = []
    for s in surfaces:
        loops.append([s[1]])
    if loops:
        loop_tag = gmsh.model.occ.addSurfaceLoop([l[0] for l in loops])
        solid = gmsh.model.occ.addVolume([loop_tag])
        gmsh.model.occ.synchronize()
    gmsh.write(OUT_STEP)
except Exception as exc:  # noqa: BLE001
    print(f"BRep shell export failed ({exc}); falling back to plain STEP write")
    gmsh.write(OUT_STEP)

gmsh.finalize()

for p in (OUT_STL, OUT_STEP):
    print(f"{'OK ' if os.path.exists(p) else 'MISSING'} {p} "
          f"({os.path.getsize(p) / 1e6:.2f} MB)" if os.path.exists(p) else
          f"MISSING {p}")

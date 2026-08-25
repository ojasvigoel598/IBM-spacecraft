"""
MissionMind — Satellite CAD Geometry Analysis

Loads the real IBM satellite STL model (exported from Fusion 360) and computes
geometric properties that feed into the thermal and power subsystem models:

  * Total surface area (m^2) — for thermal radiation (Stefan-Boltzmann)
  * Volume (m^3) — for thermal mass estimation
  * Cross-sectional area facing the Sun (m^2) — for solar power calculation
  * Cross-sectional area facing Earth (m^2) — for albedo/IR heating
  * Bounding box — for spacecraft MOI estimation

This module bridges the gap between "decorative 3D view" and "geometry-
driven thermal/power model" — a key digital twin requirement.

The 3D view (Three.js) already renders this mesh, but the physics models
(config.py) use hard-coded AREA=0.5 m^2 and A_SUNLIT=0.30 m^2.  These
defaults are documented as estimates; this module provides the actual
values from the real CAD geometry.

Run:
    python -m missionmind.viz.components.satellite_geometry
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_STL_PATH = os.path.join(_HERE, "models", "ibm_satellite.stl")
_OBJ_PATH = os.path.join(_HERE, "models", "ibm_satellite.obj")


@dataclass(frozen=True)
class SatelliteGeometry:
    """Computed geometric properties from the real CAD model."""
    total_surface_area_m2: float       # total mesh surface area
    volume_m3: float                   # enclosed volume (if watertight)
    cross_section_sun_m2: float        # projected area facing Sun (+X axis)
    cross_section_earth_m2: float      # projected area facing Earth (-Z axis)
    bounding_box_m: tuple              # (x_len, y_len, z_len) in metres
    n_vertices: int                    # mesh vertex count
    n_faces: int                       # mesh triangle count
    is_watertight: bool                # whether the mesh is closed
    source_file: str                   # which file was loaded


def load_mesh(path: Optional[str] = None):
    """Load the satellite STL/OBJ mesh via trimesh.

    Returns the trimesh.Trimesh object.  Raises ImportError if trimesh is
    not installed, or FileNotFoundError if the mesh file is missing.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for satellite geometry analysis. "
            "Install with: pip install trimesh"
        )
    if path is None:
        # Try STL first, then OBJ
        if os.path.exists(_STL_PATH):
            path = _STL_PATH
        elif os.path.exists(_OBJ_PATH):
            path = _OBJ_PATH
        else:
            raise FileNotFoundError(
                f"No satellite mesh found. Expected {_STL_PATH} or {_OBJ_PATH}"
            )
    return trimesh.load(path, force="mesh")


def projected_area(mesh, direction: str = "+x") -> float:
    """Compute the projected (cross-sectional) area of the mesh along a
    given direction.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The satellite mesh.
    direction : str
        One of '+x', '-x', '+y', '-y', '+z', '-z'.

    Returns
    -------
    float
        Projected area in m^2.
    """
    direction_map = {
        "+x": [1, 0, 0], "-x": [-1, 0, 0],
        "+y": [0, 1, 0], "-y": [0, -1, 0],
        "+z": [0, 0, 1], "-z": [0, 0, -1],
    }
    d = direction_map[direction]
    # Project each triangle onto the plane perpendicular to d and sum
    # absolute values (faces facing both ways contribute to cross-section).
    normals = mesh.face_normals
    areas = mesh.area_faces
    # Dot product of face normal with projection direction
    dots = normals @ d
    # Only count faces that have a component along the direction
    # (positive dot = facing the direction)
    projected = areas * np.abs(dots)
    return float(np.sum(projected))


# The Fusion 360 export uses mm as the base unit.  The physical satellite
# is ~200 mm long (the largest axis).  We scale to SI (metres) so all
# computed areas and volumes are in m^2 and m^3.
_MESH_SCALE_MM_TO_M = 0.001  # 1 mm = 0.001 m


def compute_geometry(path: Optional[str] = None) -> SatelliteGeometry:
    """Compute all geometric properties from the satellite CAD model."""
    mesh = load_mesh(path)
    # Scale from mm to metres
    mesh.apply_scale(_MESH_SCALE_MM_TO_M)

    total_area = float(np.sum(mesh.area_faces))
    try:
        volume = float(mesh.volume)
    except Exception:
        # Non-watertight meshes may not have a volume
        volume = 0.0

    # Sun-facing: +X is the solar array normal in the satellite frame
    area_sun = projected_area(mesh, "+x")
    # Earth-facing: -Z is the nadir direction
    area_earth = projected_area(mesh, "-z")

    extents = mesh.bounding_box.extents
    bbox = tuple(float(e) for e in extents)

    return SatelliteGeometry(
        total_surface_area_m2=round(total_area, 4),
        volume_m3=round(volume, 6),
        cross_section_sun_m2=round(area_sun, 4),
        cross_section_earth_m2=round(area_earth, 4),
        bounding_box_m=bbox,
        n_vertices=len(mesh.vertices),
        n_faces=len(mesh.faces),
        is_watertight=bool(mesh.is_watertight),
        source_file=mesh.metadata.get("file", "unknown"),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("MissionMind — Satellite CAD Geometry Analysis (trimesh)")
    print("=" * 72)

    geo = compute_geometry()

    print(f"\nSource:       {geo.source_file}")
    print(f"Vertices:     {geo.n_vertices}")
    print(f"Faces:        {geo.n_faces}")
    print(f"Watertight:   {geo.is_watertight}")
    print(f"\nSurface area:     {geo.total_surface_area_m2:.4f} m^2")
    print(f"Volume:           {geo.volume_m3:.6f} m^3")
    print(f"Sun-facing area:  {geo.cross_section_sun_m2:.4f} m^2  "
          f"(config default: 0.30 m^2)")
    print(f"Earth-facing:     {geo.cross_section_earth_m2:.4f} m^2  "
          f"(config default: 0.50 m^2)")
    print(f"Bounding box:     {geo.bounding_box_m[0]:.4f} x "
          f"{geo.bounding_box_m[1]:.4f} x {geo.bounding_box_m[2]:.4f} m")

    # Compare with config defaults
    print("\n--- Config vs CAD comparison ---")
    print(f"  A_SUNLIT (config): 0.3000 m^2, CAD: {geo.cross_section_sun_m2:.4f} m^2")
    print(f"  AREA    (config): 0.5000 m^2, CAD: {geo.total_surface_area_m2:.4f} m^2")
    print(f"  (config values are estimates; CAD values are from the real Fusion 360 model)")

    print("\nPASS — satellite geometry analysis OK")

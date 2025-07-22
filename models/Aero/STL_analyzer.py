"""ChatGPT Generated Code: Generates a silhouette of a STEP file tp determine the cross-sectional area"""

import cadquery as cq
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os

IN2_PER_M2 = 1550.0031

# === STEP MESHING ===
def load_mesh_from_step(path: str):
    wp = cq.importers.importStep(path)
    shape = wp.objects[0]
    verts_raw, faces = shape.tessellate(0.5)

    verts = np.array([v.toTuple() if hasattr(v, "toTuple") else v for v in verts_raw])
    tris = np.array([[verts[i] for i in face] for face in faces if len(face) == 3])
    print("[INFO] Loaded triangles:", tris.shape)
    return tris

# === ROTATION ===
def rotate_points(points: np.ndarray, axis: str, angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return points @ R.T

# === PROJECTION ===
def project_points(points3: np.ndarray, plane: str) -> np.ndarray:
    if plane == 'XY': return points3[:, [0, 1]]
    if plane == 'XZ': return points3[:, [0, 2]]
    if plane == 'YZ': return points3[:, [1, 2]]
    raise ValueError("plane must be XY,XZ,YZ")

# === TRUE PROJECTED AREA ===
def compute_true_projected_area(triangles, angle_deg, axis='y', plane='XZ', debug=False):
    N = triangles.shape[0]
    rotated = rotate_points(triangles.reshape(-1, 3), axis, angle_deg).reshape(N, 3, 3)
    projected_2d = project_points(rotated.reshape(-1, 3), plane).reshape(N, 3, 2)

    polygons = [Polygon(tri) for tri in projected_2d if Polygon(tri).is_valid]
    union = unary_union(polygons)
    area_m2 = union.area
    area_in2 = area_m2 * IN2_PER_M2

    if debug:
        fig, ax = plt.subplots(figsize=(5, 5))
        for poly in polygons:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.1, edgecolor='k', linewidth=0.3)
        if isinstance(union, Polygon):
            ux, uy = union.exterior.xy
            ax.plot(ux, uy, 'r', linewidth=2)
        ax.set_aspect('equal')
        ax.set_title(f"Projected Area\n{area_m2:.4f} m² / {area_in2:.2f} in²")
        plt.grid(True)
        plt.show()

    return area_m2

# === MAIN ===
if __name__ == '__main__':
    step_file = "C:/Users/emcof/PycharmProjects/ControlTheorySimulation/models/Aero/Rocket Model (AERO) v1.STEP"
    triangles = load_mesh_from_step(step_file)
    triangles *= 0.001  # mm → m

    # Show debug silhouette at one example AoA (0° in XZ plane)
    _ = compute_true_projected_area(triangles, angle_deg=2, axis='z', plane='XZ', debug=True)

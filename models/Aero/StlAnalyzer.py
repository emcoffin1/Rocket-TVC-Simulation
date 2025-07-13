import trimesh
import numpy as np
import matplotlib.pyplot as plt


def load_stl(filepath: str) -> trimesh.Trimesh:
    return trimesh.load_mesh(filepath)


def project_to_plane(vertices: np.ndarray, plane: str) -> np.ndarray:
    if plane == "XY":
        return vertices[:, :2]
    elif plane == 'YZ':
        return vertices[:, 1:]
    elif plane == 'XZ':
        return vertices[:, [0, 2]]
    else:
        raise ValueError(f"Invalid plane '{plane}', use 'XY', 'YZ', or 'XZ'.")


def triangle_area_2d(pts: np.ndarray) -> float:
    """
    Calculate 2D area of a triangle given its 3 points
    Uses the shoelace formula
    """
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]
    return 0.5 * abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))


def projected_area(mesh: trimesh.Trimesh, plane: str) -> float:
    area = 0.0
    for face in mesh.faces:
        tri_3d = mesh.vertices[face]
        tri_2d = project_to_plane(tri_3d, plane)
        area += triangle_area_2d(tri_2d)
    return area


def rotate_mesh(mesh: trimesh.Trimesh, axis: str, angle_deg: float) -> trimesh.Trimesh:
    angle_rad = np.radians(angle_deg)
    rotation = trimesh.transformations.rotation_matrix(angle_rad, direction=axis_map[axis])
    return mesh.copy().apply_transform(rotation)

axis_map = {
    'x': [1, 0, 0],
    'y': [0, 1, 0],
    'z': [0, 0, 1]
}


def plot_3x4_projections(mesh: trimesh.Trimesh, plane: str = 'XY', axis: str = "y", show: bool = True):


    def get_projection_bounds(projected_vertices: np.ndarray, pad_scale: float = 1.1):
        """
        Return centered axis limits with optional padding.
        """
        min_xy = projected_vertices.min(axis=0)
        max_xy = projected_vertices.max(axis=0)

        center = (min_xy + max_xy) / 2
        size = (max_xy - min_xy) * pad_scale / 2

        xlim = (center[0] - size[0], center[0] + size[0])
        ylim = (center[1] - size[1], center[1] + size[1])

        return xlim, ylim

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(f"{plane} Plane Projections at Different Y-Axis Rotations", fontsize=16)

    angles = list(range(0, 360, 30))

    for idx, angle in enumerate(angles):
        row, col = divmod(idx, 4)
        ax = axes[row][col]

        rotated = rotate_mesh(mesh, axis=axis, angle_deg=angle)

        all_proj_pts = []
        for face in rotated.faces:
            tri_3d = rotated.vertices[face]
            tri_2d = project_to_plane(tri_3d, plane)
            all_proj_pts.append(tri_2d)
            patch = plt.Polygon(tri_2d, edgecolor='black', facecolor='lightgray', alpha=0.7)
            ax.add_patch(patch)

        # Compute bounding box of projected points
        all_pts_flat = np.vstack(all_proj_pts)
        xlim, ylim = get_projection_bounds(all_pts_flat)

        area = projected_area(rotated, plane)
        ax.set_title(f"{angle}° | Area: {area:.2f}")
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axis('off')
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if show:
        plt.show()



def plot_reference_area_vs_yaw(angles, areas, plane="XZ", axis="y", show: bool = True):

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(np.radians(angles), areas, label=f"{plane} Projection")
    ax.fill(np.radians(angles), areas, alpha=0.3)
    ax.set_title(f"Projected Area vs Yaw (rot about {axis.upper()}, proj on {plane}")
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.legend()
    if show:
        plt.show()


def get_reference_area_vs_yaw(mesh: trimesh.Trimesh, plane="XZ", axis='y'):
    angles = np.arange(0, 360, 30)
    areas = []

    for angle in angles:
        rotated = rotate_mesh(mesh, axis=axis, angle_deg=angle)
        area = projected_area(rotated, plane)
        areas.append(area)

    return angles, areas








if __name__ == "__main__":
    # filepath = "C:/Users/emcof/OneDrive/Documents/PEAR - Trajectory Main/circle_disk.stl"
    # filepath = "C:/Users/emcof/PycharmProjects/ControlTheorySimulation/models/Aero/circle_disk.stl"
    filepath = 'C:/Users/emcof/PycharmProjects/ControlTheorySimulation/models/Aero/model_rocket.stl'
    mesh = load_stl(filepath)
    # mesh.show()
    for angle in range(0, 360, 30):
        rotated = rotate_mesh(mesh,  axis='z', angle_deg=angle)
        area_xy = projected_area(rotated, 'XY')
        area_xz = projected_area(rotated, 'XZ')
        area_yz = projected_area(rotated, 'YZ')
        print(f"Angle {angle}° Y-axis | XY: {area_xy:.4f}, XZ: {area_xz:.4f}, YZ: {area_yz:.4f}")

    plot_3x4_projections(mesh, plane="XZ", axis="z")
    # plot_reference_area_vs_yaw(mesh)

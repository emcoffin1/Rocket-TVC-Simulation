import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# --- Initial pose (wrong) ---
p_current = np.array([1.0, 2.0, 0.0])
v_current = np.array([0.0, 0.435, 0.9])
v_current = v_current / np.linalg.norm(v_current)

# --- Desired pose ---
p_target = np.array([0.0, 0.0, 0.0])
v_target = np.array([0.0, 0.0, 1.0])  # "Up"

# --- Rotation: compute quaternion to align orientation vectors ---
def compute_rotation_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    if np.isclose(dot, -1.0):
        orthogonal = np.array([1, 0, 0]) if not np.isclose(v1[0], 1.0) else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal)
        axis /= np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis)
    elif np.isclose(dot, 1.0):
        return R.from_quat([0, 0, 0, 1])
    else:
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(dot)
        return R.from_rotvec(axis * angle)

# Get rotation correction
rotation_correction = compute_rotation_between(v_current, v_target)
v_corrected = rotation_correction.apply(v_current)

# --- Visualization ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original pose
ax.quiver(*p_current, *v_current, color='red', label='Current Pose', linewidth=2)
ax.scatter(*p_current, color='red')

# Plot target pose
ax.quiver(*p_target, *v_target, color='green', label='Target Pose', linewidth=2)
ax.scatter(*p_target, color='green')

# Plot corrected orientation at translated position
ax.quiver(*p_target, *v_corrected, color='blue', label='Corrected Orientation', linewidth=2, linestyle='dashed')

# Display setup
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-1, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Translation + Rotation Correction to Target Pose")
ax.legend()
ax.grid(True)
plt.show()

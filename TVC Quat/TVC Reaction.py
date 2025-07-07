import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Current pose ---
p_current = np.array([1.0, 2.0, 0.0])
v_current = np.array([0.0, 0.0697, 0.997])  # Must be unit vector
v_current = v_current / np.linalg.norm(v_current)

# --- Target pose ---
p_target = np.array([0.0, 0.0, 0.0])
v_target = np.array([0.0, 0.0, 1.0])  # pointing up

# --- Compute translation vector ---
translation_vec = p_target - p_current
translation_dir = translation_vec / np.linalg.norm(translation_vec)

# --- Compute orientation correction (quaternion error) ---
def compute_quaternion_error(v1, v2):
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

rotation_error = compute_quaternion_error(v_current, v_target)
rot_angle = rotation_error.magnitude()
rot_axis = rotation_error.as_rotvec()
rot_axis_unit = rot_axis / rot_angle if rot_angle > 1e-6 else np.array([0, 0, 0])

# --- Combine for total TVC correction vector (for a simple first-step controller) ---
# Idea: fire in direction of translation, but apply rotation about axis to align
# This could be used to weight thrust vector angle vs. magnitude
combined_thrust_direction = translation_dir  # Simplified, could blend with rotation correction

# --- Output ---
print("🚀 TVC + Translational Correction Outputs")
print("=========================================")
print(f"🔁 Rotation Angle Needed: {np.rad2deg(rot_angle):.2f} degrees")
print(f"↻ Rotation Axis (unit vector): {rot_axis_unit}")
print(f"🧭 Translation Vector: {translation_vec}")
print(f"➡️ Translation Direction (unit): {translation_dir}")
print(f"🧮 Combined Thrust Direction (point toward target): {combined_thrust_direction}")

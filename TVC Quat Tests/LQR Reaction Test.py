import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# --- Define simulation parameters ---
dt = 0.1  # time step
steps = 30

# Initial pose
p_current = np.array([1.0, 2.0, 0.0])
v_current = np.array([0.0, 0.435, 0.9])
v_current = v_current / np.linalg.norm(v_current)

# Target pose
p_target = np.array([0.0, 0.0, 0.0])
v_target = np.array([0.0, 0.0, 1.0])

# LQR-like gain matrices (mocked for simplicity)
Kp_pos = 1.5
Kp_att = 2.0

# Storage for plotting
angle_errors = []
pos_errors = []

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

# Simulate over time
for _ in range(steps):
    # --- Compute errors ---
    pos_error = p_target - p_current
    rot_error = compute_quaternion_error(v_current, v_target)
    angle = rot_error.magnitude()
    axis = rot_error.as_rotvec()

    # Log errors
    angle_errors.append(np.rad2deg(angle))
    pos_errors.append(np.linalg.norm(pos_error))

    # --- Apply simple "LQR" correction ---
    # Position correction (move toward target)
    p_current += Kp_pos * pos_error * dt

    # Orientation correction (rotate toward target)
    if angle > 1e-6:
        delta_angle = Kp_att * angle * dt  # reduce angle over time
        delta_rot = R.from_rotvec(axis / np.linalg.norm(axis) * delta_angle)
        v_current = delta_rot.apply(v_current)
        v_current = v_current / np.linalg.norm(v_current)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(angle_errors, label='Orientation Error (deg)', color='blue')
ax1.set_ylabel('Angle Error (deg)')
ax1.grid(True)
ax1.legend()

ax2.plot(pos_errors, label='Position Error (m)', color='green')
ax2.set_xlabel('Step [0.1s]')
ax2.set_ylabel('Position Error (m)')
ax2.grid(True)
ax2.legend()

plt.suptitle("TVC Correction Over Time (LQR-like Control)")
plt.tight_layout()
plt.show()

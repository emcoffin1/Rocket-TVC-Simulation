import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
dt = 0.1
steps = 100
times = []
time = 0

# Initial pose
p_current = np.array([1.0, 2.0, 0.0])
v_current = np.array([0.0, 0.435, 0.9])
v_current = v_current / np.linalg.norm(v_current)

# Target pose
p_target = np.array([0.0, 0.0, 0.0])
v_target = np.array([0.0, 0.0, 1.0])

# PID gains
Kp_pos = 1.2
Ki_pos = 0.1
Kd_pos = 0.1

Kp_att = 1.5
Ki_att = 1.0
Kd_att = 1.0

# Initialize PID state
pos_integral = np.zeros(3)
pos_prev_error = p_target - p_current

att_integral = 0.0
att_prev_error = 0.0

# Logs for plotting
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

# Simulation loop
for _ in range(steps):
    # --- Position PID Control ---
    pos_error = p_target - p_current
    pos_integral += pos_error * dt
    pos_derivative = (pos_error - pos_prev_error) / dt
    pos_correction = Kp_pos * pos_error + Ki_pos * pos_integral + Kd_pos * pos_derivative
    p_current += pos_correction * dt
    pos_prev_error = pos_error

    # --- Orientation PID Control ---
    rot_error = compute_quaternion_error(v_current, v_target)
    angle = rot_error.magnitude()
    axis = rot_error.as_rotvec()

    att_integral += angle * dt
    att_derivative = (angle - att_prev_error) / dt
    att_correction_angle = Kp_att * angle + Ki_att * att_integral + Kd_att * att_derivative
    att_prev_error = angle

    if angle > 1e-6:
        axis_normalized = axis / np.linalg.norm(axis)
        delta_rot = R.from_rotvec(axis_normalized * att_correction_angle * dt)
        v_current = delta_rot.apply(v_current)
        v_current = v_current / np.linalg.norm(v_current)

    # --- Log Errors ---
    angle_errors.append(np.rad2deg(angle))
    pos_errors.append(np.linalg.norm(pos_error))
    times.append(time)
    time+=dt

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(times, angle_errors,label='Orientation Error (deg)', color='blue')
ax1.set_ylabel('Angle Error (deg)')
ax1.grid(True)
ax1.legend()

ax2.plot(times, pos_errors, label='Position Error (m)', color='green')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position Error (m)')
ax2.grid(True)
ax2.legend()

plt.suptitle("TVC Correction Over Time (PID Control)")
plt.tight_layout()
plt.show()

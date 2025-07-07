# Long-duration TVC simulation with velocity-dominant control, gimbal limits, and smoothed thrust vector
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# --- Constants ---
g = 32.174  # ft/s^2 (gravity)
dt = 0.05
total_time = 30.0
steps = int(total_time / dt)

# --- Rocket Parameters ---
mass = 375.0 / 32.174  # lbm to slugs
thrust = 2000.0  # lbf
I = np.diag([10.0, 10.0, 1.0])  # slug*ft^2

# --- Initial State ---
pos = np.array([1.0, 1.0, 0.0], dtype=np.float64)
vel = np.zeros(3)
angle_offset = np.deg2rad(7)
orientation = R.from_euler('y', angle_offset)
angular_velocity = np.zeros(3)

# --- Desired Orientation ---
target_orientation = R.from_quat([0, 0, 0, 1])

# --- LQR Gains (velocity-focused for damping) ---
K_pos = np.diag([1.2,1.2,0])
K_vel = np.diag([2.2, 2.2, 0.0])
K_ang = np.diag([2.0, 2.0, 0.0])
K_rate = np.diag([1.0, 1.0, 0.0])

# --- Gimbal Limit and Filtering ---
max_gimbal_rad = np.deg2rad(10)
alpha = 0.1  # low-pass filter factor
prev_thrust_dir = np.array([0.0, 0.0, 1.0])

# --- Logs ---
log_pos, log_vel, log_orient_error, log_ang_vel = [], [], [], []

# --- Simulation Loop ---
for step in range(steps):
    time = step * dt
    target_pos = np.array([0.0, 0.0, time * 40.0])  # vertical climb

    # --- Translation Control ---
    pos_error = target_pos - pos
    vel_error = -vel
    trans_cmd = K_pos @ pos_error + K_vel @ vel_error
    trans_cmd[2] = 0.0

    desired_thrust_dir = np.array([0.0, 0.0, 1.0]) + trans_cmd
    desired_thrust_dir /= np.linalg.norm(desired_thrust_dir)

    # Clamp gimbal angle
    vertical = np.array([0.0, 0.0, 1.0])
    dot = np.dot(desired_thrust_dir, vertical)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    if angle > max_gimbal_rad:
        axis = np.cross(vertical, desired_thrust_dir)
        axis = axis / np.linalg.norm(axis)
        limited_rot = R.from_rotvec(axis * max_gimbal_rad)
        thrust_direction = limited_rot.apply(vertical)
    else:
        thrust_direction = desired_thrust_dir

    # Low-pass filter
    thrust_direction = (1 - alpha) * prev_thrust_dir + alpha * thrust_direction
    thrust_direction /= np.linalg.norm(thrust_direction)
    prev_thrust_dir = thrust_direction

    # --- Rotation Control ---
    orientation_error_rot = orientation.inv() * target_orientation
    angle_axis = orientation_error_rot.as_rotvec()
    ang_vel_error = -angular_velocity
    torque_cmd = K_ang @ angle_axis + K_rate @ ang_vel_error

    # --- Angular Dynamics ---
    angular_accel = np.linalg.inv(I) @ torque_cmd
    angular_velocity += angular_accel * dt
    delta_rot = R.from_rotvec(angular_velocity * dt)
    orientation = orientation * delta_rot

    # --- Apply Rotated Thrust ---
    rotated_thrust = orientation.apply(thrust_direction) * thrust
    accel = rotated_thrust / mass - np.array([0.0, 0.0, g])
    vel += accel * dt
    pos += vel * dt

    # --- Log ---
    log_pos.append(pos.copy())
    log_vel.append(vel.copy())
    log_orient_error.append(np.rad2deg(np.linalg.norm(angle_axis)))
    log_ang_vel.append(angular_velocity.copy())

# --- Convert Logs ---
log_pos = np.array(log_pos)
log_vel = np.array(log_vel)
log_orient_error = np.array(log_orient_error)
log_ang_vel = np.array(log_ang_vel)
times = np.arange(0, total_time, dt)

# --- Print Summary ---
print("🚀 Final Position:", log_pos[-1])
print("🎯 Final Velocity:", log_vel[-1])
print("🌀 Final Angular Velocity:", log_ang_vel[-1])
print("✅ Final Orientation Error (deg):", log_orient_error[-1])

# --- Plotting ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(times, log_pos[:, 0], label='X')
axs[0].plot(times, log_pos[:, 1], label='Y')
axs[0].plot(times, log_pos[:, 2], label='Z')
axs[0].set_ylabel("Position (ft)")
axs[0].legend()
axs[0].grid()

axs[1].plot(times, log_vel[:, 0], label='Vx')
axs[1].plot(times, log_vel[:, 1], label='Vy')
axs[1].plot(times, log_vel[:, 2], label='Vz')
axs[1].set_ylabel("Velocity (ft/s)")
axs[1].legend()
axs[1].grid()

axs[2].plot(times, log_orient_error, color='orange')
axs[2].set_ylabel("Orientation Error (deg)")
axs[2].grid()

axs[3].plot(times, log_ang_vel[:, 0], label='wx')
axs[3].plot(times, log_ang_vel[:, 1], label='wy')
axs[3].plot(times, log_ang_vel[:, 2], label='wz')
axs[3].set_ylabel("Angular Velocity (rad/s)")
axs[3].set_xlabel("Time (s)")
axs[3].legend()
axs[3].grid()

plt.suptitle("Smoothed TVC Simulation (30s, Velocity-Based Damping, 5° Gimbal Limit)")
plt.tight_layout()
plt.show()

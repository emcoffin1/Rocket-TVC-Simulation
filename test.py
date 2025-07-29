import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def compute_gimbal_deflected_thrust(thrust_mag, theta_x, theta_y):
    """
    Compute thrust vector in body frame after gimbal deflection via quaternions.
    :param thrust_mag: Magnitude of thrust force
    :param theta_x: Gimbal angle around X-axis (radians)
    :param theta_y: Gimbal angle around Y-axis (radians)
    :return: Deflected thrust vector in body frame
    """
    # Rotate +Z by Rx then Ry (yaw then pitch in body frame)
    rot_x = R.from_rotvec(theta_x * np.array([1, 0, 0]))
    rot_y = R.from_rotvec(theta_y * np.array([0, 1, 0]))
    combined_rot = rot_y * rot_x

    thrust_dir = combined_rot.apply([0, 0, 1])
    return thrust_mag * thrust_dir

# Parameters
thrust = 10000.0  # N
lever_arm = 2.0   # m

# Torque commands
torque_cmds = [
    np.array([1000.0, 0.0, 0.0]),  # +τx only
    np.array([0.0, 1000.0, 0.0]),  # +τy only
]

def compute_gimbal_angles(torque_cmd, thrust, lever):
    τx, τy = torque_cmd[0], torque_cmd[1]
    ratio_x = -τy / (thrust * lever)
    ratio_y =  τx / (thrust * lever)

    ratio_x = np.clip(ratio_x, -1.0, 1.0)
    ratio_y = np.clip(ratio_y, -1.0, 1.0)

    θx = np.arcsin(ratio_x)
    θy = np.arcsin(ratio_y)

    return θx, θy

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original thrust vector
origin = np.zeros(3)
ax.quiver(*origin, 0, 0, thrust, color='k', label='Original +Z Thrust', linewidth=2)

# Plot deflected vectors for torque_cmds
colors = ['r', 'b']
labels = ['+τx (pitch up)', '+τy (yaw right)']

for torque, color, label in zip(torque_cmds, colors, labels):
    θx, θy = compute_gimbal_angles(torque, thrust, lever_arm)
    T_vec = compute_gimbal_deflected_thrust(thrust, θx, θy)
    ax.quiver(*origin, *T_vec, color=color, label=f'{label}', linewidth=2)

# Formatting
ax.set_xlim([-thrust/5, thrust/5])
ax.set_ylim([-thrust/5, thrust/5])
ax.set_zlim([0, thrust])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Thrust Vector Deflection due to Gimbal Torques')
ax.legend()
ax.view_init(elev=20, azim=135)

plt.tight_layout()
plt.show()


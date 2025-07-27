import numpy as np
from scipy.spatial.transform import Rotation as R
#
# # Desired Z direction (trajectory tangent, for example)
# z_axis = np.array([0,1,1])
# z_axis = z_axis / np.linalg.norm(z_axis)  # ✅ normalize to unit vector
#
# # Define a consistent "world up" reference
# world_up = np.array([0, 0, 1])
#
# # Prevent degenerate case if z_axis ≈ world_up
# if np.allclose(np.abs(np.dot(z_axis, world_up)), 1.0):
#     world_up = np.array([0, 1, 0])  # pick a new reference if aligned
#
# # Build orthonormal basis
# x = np.cross(world_up, z_axis)
# x = x / np.linalg.norm(x)  # ✅ normalize
#
# y = np.cross(z_axis, x)
#
# # Stack into rotation matrix (columns are body axes in world frame)
# r_no_roll = np.column_stack((x, y, z_axis))
#
# # Convert to quaternion
# q_no_roll = R.from_matrix(r_no_roll).as_quat()  # [x, y, z, w]
#
# print("Quaternion without roll:", q_no_roll)
# roll_angle = np.radians(30)
# q_roll = R.from_rotvec(roll_angle * z_axis).as_quat()
#
# # Combine: q_des = q_roll ⊗ q_no_roll
# q_des = R.from_quat(q_roll) * R.from_quat(q_no_roll)
# print("Quaternion with roll:", q_des.as_quat())


current_vector = np.array([0.5, 0.25, 1.0])
current_vector /= np.linalg.norm(current_vector)

target_vector = np.array([0.25,0.75,1])
target_vector /= np.linalg.norm(target_vector)


rotaiton_axis = np.linalg.cross(current_vector, target_vector)
rotaiton_axis /= np.linalg.norm(rotaiton_axis)
dot_product = np.clip(np.dot(current_vector, target_vector), -1.0, 1.0)
angle = np.arccos(dot_product)
angle /= 2
sin = np.sin(angle)
rotation_quaternion = np.array([rotaiton_axis[0]*sin, rotaiton_axis[1]*sin, rotaiton_axis[2]*sin, np.cos(angle)])
print(rotation_quaternion)

body_frame = np.array([0.0, 0.0, 1.0])

rotation = R.from_quat(quat=rotation_quaternion)
world_forward = rotation.apply(body_frame)

print(world_forward)
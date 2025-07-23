import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def generate_missile_lookup_table(filename='missile_profile.csv', num_points=100000):
    # Altitude range (no duplicates)
    z_vals = np.linspace(0, 100000, num_points)

    # Example 3D trajectory (sinusoidal arc)
    x_vals = 1000 * np.sin(np.pi * z_vals / z_vals[-1])
    y_vals = 300 * np.sin(2 * np.pi * z_vals / z_vals[-1])

    # Force initial point to be exactly (0, 0, 0)
    x_vals[0], y_vals[0], z_vals[0] = 0, 0, 0

    # Position vectors
    positions = np.stack([x_vals, y_vals, z_vals], axis=1)

    # Direction vectors (velocity estimate)
    vels = np.gradient(positions, axis=0)
    vels /= np.linalg.norm(vels, axis=1, keepdims=True)

    # Calculate orientation quaternion from forward velocity vector
    quats = []
    z_axis = np.array([0, 0, 1])
    for vel in vels:
        r, _ = R.align_vectors([vel], [z_axis])
        q = r.as_quat()  # x, y, z, w
        quats.append(q)

    quats = np.array(quats)

    # Build DataFrame
    df = pd.DataFrame({
        'z': z_vals,
        'x': x_vals,
        'y': y_vals,
        'qx': quats[:, 0],
        'qy': quats[:, 1],
        'qz': quats[:, 2],
        'qw': quats[:, 3],
    })

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Missile profile saved to '{filename}'")

    return df

# Generate table and preview trajectory
df = generate_missile_lookup_table()

# Plot trajectory for visual check
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['x'], df['y'], df['z'], label='Missile Trajectory', linewidth=2)
ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.set_zlabel('Altitude Z [m]')
ax.set_title('Missile Trajectory Path (3D)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()


# LOOK UP METHOD
# import pandas as pd
# import numpy as np
#
# # Load table
# df = pd.read_csv('missile_profile.csv')
#
# def get_pose_by_altitude(z_query):
#     # Interpolate x, y, and quaternion components
#     x = np.interp(z_query, df['z'], df['x'])
#     y = np.interp(z_query, df['z'], df['y'])
#     qx = np.interp(z_query, df['z'], df['qx'])
#     qy = np.interp(z_query, df['z'], df['qy'])
#     qz = np.interp(z_query, df['z'], df['qz'])
#     qw = np.interp(z_query, df['z'], df['qw'])
#     return (x, y, z_query), (qx, qy, qz, qw)


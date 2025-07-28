import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def generate_missile_lookup_table(
    filename: str = 'missile_profile_.csv',
    num_points: int = 100000,
    max_altitude: float = 100000.0,
    pitch_rate_deg_per_km: float = 35.0
) -> pd.DataFrame:
    """
    Generate a vertical climb profile where the vehicle pitches by
    `pitch_rate_deg_per_km` degrees every 1000 m of ascent.

    Columns:
      z   – altitude [m]
      x,y – horizontal position (zero)
      qx,qy,qz,qw – body‑to‑world quaternion (x, y, z, w)
    """
    # 1) Altitude grid
    z_vals = np.linspace(0.0, max_altitude, num_points)

    # 2) Horizontal positions (vertical climb)
    x_vals = np.zeros_like(z_vals)
    y_vals = np.zeros_like(z_vals)

    # 3) Compute pitch angle (in radians) at each altitude
    pitch_deg = pitch_rate_deg_per_km * (z_vals / 1000.0)
    pitch_rad = np.deg2rad(pitch_deg)

    # 4) Build quaternions for rotation about local X‑axis
    half = pitch_rad / 2.0
    qx = np.sin(half)
    qy = np.zeros_like(half)
    qz = np.zeros_like(half)
    qw = np.cos(half)

    # 5) Assemble into a DataFrame
    df = pd.DataFrame({
        'z':   z_vals,
        'x':   x_vals,
        'y':   y_vals,
        'qx':  qx,
        'qy':  qy,
        'qz':  qz,
        'qw':  qw,
    })

    # 6) Save and return
    df.to_csv(filename, index=False)
    print(f"✅ Slow‑pitch profile saved to '{filename}'")
    return df


def generate_orbital_insertion_lookup_table(
    filename: str = 'orbital_insertion_profile.csv',
    num_points: int = 100000,
    max_altitude: float = 100000.0,
    sweep_altitude: float = 50000.0
) -> pd.DataFrame:
    """
    Build a lookup table where:
      • z (m) goes from 0 → max_altitude
      • x, y remain zero (vertical climb path)
      • orientation quaternion rotates about the Y‑axis from 0° → 90°
        between z=0 and z=sweep_altitude, then holds at 90°.

    Quaternion convention: (qx, qy, qz, qw), for a rotation φ about Y:
      q = [ 0, sin(φ/2), 0, cos(φ/2) ]
    """
    # 1) Altitude grid
    z = np.linspace(0.0, max_altitude, num_points)

    # 2) Horizontal path (vertical ascent)
    x = np.zeros_like(z)
    y = np.zeros_like(z)

    # 3) Compute sweep angle φ(z):
    #    ramp linearly from 0 → π/2 over [0, sweep_altitude], then clamp
    φ = np.clip((z / sweep_altitude) * (np.pi / 2), 0.0, np.pi / 2)

    # 4) Quaternion components for rotation about Y‑axis by φ:
    half = φ / 2.0
    qx = np.zeros_like(half)
    qy = np.sin(half)
    qz = np.zeros_like(half)
    qw = np.cos(half)

    # 5) Assemble DataFrame
    df = pd.DataFrame({
        'z':   z,
        'x':   x,
        'y':   y,
        'qx':  qx,
        'qy':  qy,
        'qz':  qz,
        'qw':  qw,
    })

    # 6) Save to CSV and return
    df.to_csv(filename, index=False)
    print(f"✅ Orbital‑insertion profile saved to '{filename}'")
    return df


def generate_cubic_sweep_tangent_table(
    filename: str = 'cubic_sweep_profile.csv',
    num_points: int = 100_000,
    max_altitude: float = 100_000.0,      # total climb
    sweep_altitude: float = 75000.0,     # where horizontal range is reached
    horizontal_range: float = 20_000.0    # X_max at z = sweep_altitude
) -> pd.DataFrame:
    """
    Build a 3D climb with:
      - z from 0 → max_altitude
      - x(z) = horizontal_range*(z/sweep_altitude)^3 (clamped)
      - y = 0

    Then compute quaternions that align body‑z to the path tangent.
    """
    # 1) Altitude vector
    z = np.linspace(0.0, max_altitude, num_points)

    # 2) Cubic‑sweep x(z)
    frac = np.clip(z / sweep_altitude, 0.0, 1.0)
    x = horizontal_range * frac**3
    y = np.zeros_like(z)

    # 3) Stack positions and finite‑difference to get velocity vectors
    pos = np.stack([x, y, z], axis=1)
    vels = np.gradient(pos, axis=0)                # approximate dpos/dindex
    tangents = vels / np.linalg.norm(vels, axis=1, keepdims=True)

    # 4) Align each tangent to the world‑z axis ([0,0,1]) → quaternion
    forward = np.array([0.0, 0.0, 1.0])
    quats = []
    for d in tangents:
        # note: align_vectors(target_vectors, source_vectors)
        r, _ = R.align_vectors([d], [forward])
        quats.append(r.as_quat())  # x, y, z, w
    quats = np.array(quats)

    # 5) Build DataFrame
    df = pd.DataFrame({
        'z':   z,
        'x':   x,
        'y':   y,
        'qx':  quats[:,0],
        'qy':  quats[:,1],
        'qz':  quats[:,2],
        'qw':  quats[:,3],
    })

    # 6) Save & return
    df.to_csv(filename, index=False)
    print(f"✅ Cubic‑sweep tangent profile saved to '{filename}'")
    return df


def generate_pingpong_lookup_table(
        filename: str = 'missile_pingpong_profile.csv',
        num_points: int = 10000,
        max_altitude: float = 10000.0,
        lateral_shift_m: float = 25.0,
        pitch_interval_m: float = 200.0
) -> pd.DataFrame:
    """
    Generate a vertical climb profile with horizontal X-axis ping-pong
    motion every `pitch_interval_m` of vertical gain.

    The rocket alternates between +X and –X by `lateral_shift_m`, creating
    a zig-zag pattern. Orientation is adjusted to face the current direction.

    Columns:
      z   – altitude [m]
      x,y – horizontal position (zero y)
      qx,qy,qz,qw – body‑to‑world quaternion (x, y, z, w)
    """
    # Altitude grid
    z_vals = np.linspace(0.0, max_altitude, num_points)

    # Ping-pong state (alternating every N meters of altitude)
    segment = np.floor(z_vals / pitch_interval_m).astype(int)
    x_vals = ((segment % 2) * 2 - 1) * lateral_shift_m  # +50, -50, +50, ...
    y_vals = np.zeros_like(x_vals)

    # Heading direction: compute unit vector from previous to current point
    dx = np.gradient(x_vals)
    dy = np.gradient(y_vals)
    dz = np.gradient(z_vals)

    directions = np.vstack([dx, dy, dz]).T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Rocket default forward axis is +X, so compute rotation from +X to direction
    body_forward = np.array([1.0, 0.0, 0.0])
    quats = []

    for d in directions:
        # Avoid NaN rotation if d is degenerate
        if np.linalg.norm(d) < 1e-6:
            quat = [0, 0, 0, 1]
        else:
            rot = R.from_rotvec(np.cross(body_forward, d))
            quat = rot.as_quat()
        quats.append(quat)

    quats = np.array(quats)  # [N x 4] in (x, y, z, w) order

    # Final output
    df = pd.DataFrame({
        'z': z_vals,
        'x': x_vals,
        'y': y_vals,
        'qx': quats[:, 0],
        'qy': quats[:, 1],
        'qz': quats[:, 2],
        'qw': quats[:, 3],
    })

    df.to_csv(filename, index=False)
    print(f"✅ Ping-pong trajectory saved to '{filename}'")
    return df

# Generate table and preview trajectory
df = generate_cubic_sweep_tangent_table()

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


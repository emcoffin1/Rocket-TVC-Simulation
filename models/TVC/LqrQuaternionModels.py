"""Program that takes positional data and determines the quaternion error, passes through an LQR, and provides
a corrective quaternion to change the angular velocity
https://ntrs.nasa.gov/api/citations/20110015701/downloads/20110015701.pdf
"""
import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_are
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import os


class LQR:
    def __init__(self, q: np.ndarray = None, r: np.ndarray = None, max_torque: np.ndarray = None, k_pos: float = None,
                 k_vel: float = None):
        # Places emphasis on quaternion correction
        self.q = q if q is not None else np.diag([1, 1, 1, 1, 1, 1, 1, 1])
        self.r = r if r is not None else np.eye(3) * 1
        self.max_t = max_torque if max_torque is not None else np.array([1740, 1740, 37.2])
        self.k_pos = k_pos if k_pos is not None else 1
        self.k_vel = k_vel if k_vel is not None else 1

    def _compute_gain(self, inertia_matrix: np.ndarray):
        # System matrices (8,6 and 6x3)
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)

        B = np.zeros((6, 3))
        B[3:6, :] = np.linalg.inv(inertia_matrix)

        P = solve_continuous_are(A, B, self.q, self.r)
        K = np.linalg.inv(self.r) @ B.T @ P

        return K

    def get_torque(self, x: np.ndarray, inertia_matrix: np.ndarray, drag_torque: np.ndarray) -> np.ndarray:
        """
        Takers state variables and returns torque command
        :param x: 6x state vector [qx, qy, qz, wx, wy, wz, vx, vy]
        :param inertia_matrix: rocket inertia matrix [Ixx, Iyy, Izz]
        :return: Torque command in body frame [Tx, Ty, Tz]
        """

        k = self._compute_gain(inertia_matrix=inertia_matrix)
        torque = -k @ x + drag_torque

        if self.max_t is not None:
            torque = np.clip(torque, -self.max_t, self.max_t)

        return torque


class QuaternionFinder:
    """
    Generates commanded body‐rates (omega_cmd) for trajectory tracking.
    Body frame: +Z through nose, +X to right, +Y down (right‐handed).
    """

    def __init__(self, lqr=None, profile_csv: str = "cubic_sweep_profile.csv"):
        # Physical parameters & aerodynamic models
        self.lqr = lqr if lqr is not None else LQR()


        self.pos_error = []

        # Load altitude->(pos,quat) profile
        self._load_lookup_table(profile_csv)

    def compute_command_torque(self, time, rocket_pos, rocket_quat, rocket_omega, rocket_vel,
                               inertia_matrix, dt, accel_base, drag_torque, side_effect=None):

        alt = rocket_pos[2]

        # Expected positions
        pos_exp, _ = self.get_pose_by_altitude(alt=alt)
        # pos_exp = np.array([1, 0 , alt])
        pos_fut, _ = self.get_pose_by_altitude(alt=alt+2.0)
        # pos_fut = np.array([1.5, 0, alt])

        # Desired acceleration vector
        a_des = self.compute_a_des(rocket_pos=rocket_pos, rocket_vel=rocket_vel, r_target=pos_exp,
                                   r_future=pos_fut, accel_base=accel_base, side_effect=side_effect)
        a_des_unit = self.normalize(a_des)
        # Desired quaternion
        # q_des = self.compute_q_des_fixed_roll(a_des=a_des, angle_rad=np.deg2rad(90), side_effect=side_effect)
        q_des = self.compute_q_des_from_accel(a_des=a_des_unit, roll_angle_rad=0.0)

        # Error quaternion
        q_err = self.quat_mult(self.quat_conj(q=rocket_quat), q_des)

        # Separate eror quaternion into components
        axis = q_err[:3]
        w = q_err[3]

        # Compute angle components
        theta_err = 2 * np.sign(w) * axis

        x = np.concatenate([theta_err, rocket_omega])
        torque = self.lqr.get_torque(x=x, inertia_matrix=inertia_matrix, drag_torque=drag_torque)

        if side_effect:
            pass

        return torque

    def compute_q_des_from_accel(self, a_des, roll_angle_rad=0.0):
        """
        Align +Z body with a_des vector.
        Use roll_angle_rad to define rotation about new Z (optional).
        """
        if np.linalg.norm(a_des) < 1e-6:
            return np.array([0, 0, 0, 1])  # Identity

        # New Z axis: direction of desired thrust
        z_axis = -a_des / np.linalg.norm(a_des)

        # Reference 'up' direction — pick world X or Y to define roll
        up_ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(z_axis, up_ref)) > 0.95:
            up_ref = np.array([1.0, 0.0, 0.0])  # Avoid near-parallel

        # Create orthonormal basis (x, y, z)
        x_axis = np.linalg.cross(up_ref, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.linalg.cross(z_axis, x_axis)

        # Apply roll about z_axis if needed
        cos_r = np.cos(roll_angle_rad)
        sin_r = np.sin(roll_angle_rad)
        x_rot = cos_r * x_axis + sin_r * y_axis
        y_rot = -sin_r * x_axis + cos_r * y_axis

        # Construct rotation matrix: body axes as columns
        R_world_to_body = np.stack((x_rot, y_rot, z_axis), axis=1)

        # Convert to quaternion
        q_des = R.from_matrix(R_world_to_body).as_quat()
        return q_des

    def compute_a_des(self, rocket_pos, rocket_vel, r_target, r_future, accel_base, side_effect):
        """
        Computes the expected acceleration using velocity vectors between current and future positions
        Can be updated if velocities are introduced into trajectory
        Currently uses path following trajectories
        :return:
        """
        # Tangent direction of path
        v_path = self.normalize(r_future - r_target)

        # Expected speed
        v_mag_expected = np.linalg.norm(rocket_vel)

        # Desire velocity vector
        v_des = v_path * v_mag_expected

        # Positional drift
        e_pos = r_target - rocket_pos
        e_vel = v_des - rocket_vel

        # PD Style correction
        k_pos = self.lqr.k_pos
        k_vel = self.lqr.k_vel
        a_des = k_pos * e_pos + k_vel * e_vel

        # Extra Force Compensation
        a_des += accel_base

        if side_effect:
            self.pos_error.append(e_pos)

        return a_des

    # --- Lookup Trajectories ---------------------------------------------------

    def _load_lookup_table(self, csv_path: str):
        df = pd.read_csv(csv_path)
        z = df['z'].values
        # position splines
        self._ix = interp1d(z, df['x'], kind='cubic', fill_value='extrapolate')
        self._iy = interp1d(z, df['y'], kind='cubic', fill_value='extrapolate')
        # # quaternion splines
        # self._iqx = interp1d(z, df['qx'], kind='cubic', fill_value='extrapolate')
        # self._iqy = interp1d(z, df['qy'], kind='cubic', fill_value='extrapolate')
        # self._iqz = interp1d(z, df['qz'], kind='cubic', fill_value='extrapolate')
        # self._iqw = interp1d(z, df['qw'], kind='cubic', fill_value='extrapolate')

    def get_pose_by_altitude(self, alt):
        """
        Returns pos at the given altitude via cubic interpolation.
        """
        try:
            # p=1/0
            pos = np.array([ self._ix(alt),
                             self._iy(alt),
                             alt ])
            quat = np.array([0, 0, 0, 1])
            return pos, self.normalize(quat)
        except Exception:
            pos = np.array([0,0,alt])
            quat = np.array([0,0,0,1])
            return pos, quat

    # --- Quaternion Helpers ---------------------------------------------------

    @staticmethod
    def quat_conj(q: np.ndarray) -> np.ndarray:
        """
        - Performs conjugation of a passed quaternion
        - Inverts the sign
        """
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=q.dtype)

    @staticmethod
    def quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Performs quaternion multiplication (Hamilton Product)
        """
        av, aw = a[:3], a[3]
        bv, bw = b[:3], b[3]
        vec = aw*bv + bw*av + np.linalg.cross(av, bv)
        scl = aw*bw - av.dot(bv)
        return np.hstack((vec, scl))

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """
        Performs a safe normalization of a given vector or quaternion
        """
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return vector  # Avoid division by zero
        return vector / norm

    @staticmethod
    def align_sign(q: np.ndarray) -> np.ndarray:
        """
        Performs a sign alignment (ensures path taken to new quaternion
        is not the longest path)
        """
        return q if q[3] >= 0 else -q

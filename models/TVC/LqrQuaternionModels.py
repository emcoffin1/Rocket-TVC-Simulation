"""Program that takes positional data and determines the quaternion error, passes through an LQR, and provides
a corrective quaternion to change the angular velocity"""
import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_are
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import os

class LQR:
    def __init__(self, q: np.ndarray = None, r: np.ndarray = None):
        # Places emphasis on quaternion correction
        self.q = q if q is not None else np.diag([1, 1, 1, 0.5, 0.5]) * 10

        self.r = r if r is not None else np.eye(3) * 1

        self.k = self._compute_gain()

    def _compute_gain(self):
        A = np.zeros((5, 5))
        # B = np.array([
        #     [2.0, 0.0, 0.0],  # ωₓ → qₓ
        #     [0.0, 2.0, 0.0],  # ωᵧ → qᵧ
        #     [0.0, 0.0, 1.0],  # ω_z → q_z
        #     [0.0, 1.0, 0.0],  # pitch → x drift correction
        #     [0.0, 0.0, 1.0],  # yaw   → y drift correction
        # ])
        B = np.zeros((5, 3))
        B[0, 0] = 1.0     # ωₓ → q_x
        B[1, 1] = 1.0     # ωᵧ → q_y
        B[2, 2] = 1.0     # ω_z → q_z

        # drift correction coupling
        B[3, 1] = 1.0     # pitch →  x‑drift
        B[4, 2] = 1.0     #  yaw  →  y‑drift



        P = solve_continuous_are(A, B, self.q, self.r)
        K = np.linalg.inv(self.r) @ B.T @ P
        return K

    def get_Qdot(self, q_err: np.ndarray, pos_err2d: np.ndarray) -> np.ndarray:
        """
        Takers quaternion error and return desired quaternion derivative
        :param q_err: error quaternion [x y z w]
        :param pos_err2d: positional error in 2 dimensions [x y]
        :return: quaternion derivative [x y z w]
        """
        q_vec = q_err[:3]
        x = np.concatenate([q_vec, pos_err2d])
        dq_vec = -self.k @ x
        return np.array([dq_vec[0], dq_vec[1], dq_vec[2], 0.0])


class QuaternionFinder:
    """
    Generates commanded body‐rates (omega_cmd) for trajectory tracking.
    Body frame: +Z through nose, +X to right, +Y down (right‐handed).
    """

    def __init__(self, mass: float, lqr=None, profile_csv: str = "cubic_sweep_profile.csv"):
        # Physical parameters & aerodynamic models
        self.mass = mass
        self.lqr = lqr if lqr is not None else LQR()

        self.quat_error = []
        self.pos_error = []

        # Load altitude->(pos,quat) profile
        self._load_lookup_table(profile_csv)

    def compute_command_omega(self, rocket_pos: np.ndarray, rocket_quat: np.ndarray, side_effect:bool = False):
        """
         1) Get desired position and quaternion
         2) Compute attitude error q_err
         3) Call LQR for q_dot
         4) Invert kinematics -> omega_cmd
        """
        # Get expected trajectory components
        alt_m = rocket_pos[2]
        pos_des, _ = self.get_pose_by_altitude(alt=alt_m)
        pos_future, _ = self.get_pose_by_altitude(alt=alt_m+10)

        tangent = pos_future - pos_des
        tangent = self._safe_normalize(tangent)
        # if side_effect:
        #     print(f" {np.round(pos_des, 3)} || {np.round(pos_future, 3)} || {np.round(tangent, 3)}")
            # print(np.round(tangent,3))

        att_des = self._rotation_between_vectors(np.array([0,0,1]), tangent)

        # Determine attitude error: q_err = q_body^-1 (quat-mult) q_des
        q_err = self._quat_mult(self._quat_conj(rocket_quat), att_des)
        q_err[3] = 0
        q_err = self._align_sign(self._safe_normalize(q_err))

        # Determine lateral drift error (only worry about XY)
        pos_err2d = pos_des[:2] - rocket_pos[:2]

        # Get qdot from lqr
        q_dot = self.lqr.get_Qdot(q_err=q_err, pos_err2d=pos_err2d)

        # get omega w_cmd = 2·vec( q_dot ⊗ q_body⁻¹ )
        omega_cmd = 2.0 * self._quat_mult(q_dot, self._quat_conj(rocket_quat))[:3]
        omega_cmd[2] = 0
        if side_effect:
            self.pos_error.append(pos_err2d)
            self.quat_error.append(q_err)
            # print(f"COMPUTED: {np.round(omega_cmd,3)}")

        return omega_cmd, q_err, pos_err2d

    # --- Lookup Trajectories ---------------------------------------------------

    def _load_lookup_table(self, csv_path: str):
        df = pd.read_csv(csv_path)
        z = df['z'].values
        # position splines
        self._ix = interp1d(z, df['x'], kind='cubic', fill_value='extrapolate')
        self._iy = interp1d(z, df['y'], kind='cubic', fill_value='extrapolate')
        # quaternion splines
        self._iqx = interp1d(z, df['qx'], kind='cubic', fill_value='extrapolate')
        self._iqy = interp1d(z, df['qy'], kind='cubic', fill_value='extrapolate')
        self._iqz = interp1d(z, df['qz'], kind='cubic', fill_value='extrapolate')
        self._iqw = interp1d(z, df['qw'], kind='cubic', fill_value='extrapolate')

    def get_pose_by_altitude(self, alt):
        """
        Returns (pos, quat) at the given altitude via cubic interpolation.
        """
        try:
            # p=1/0
            pos = np.array([ self._ix(alt),
                             self._iy(alt),
                             alt ])
            quat = np.array([ self._iqx(alt),
                              self._iqy(alt),
                              self._iqz(alt),
                              self._iqw(alt) ])
            return pos, self._safe_normalize(quat)
        except Exception:
            pos = np.array([0,0,alt])
            quat = np.array([0,0,0,1])
            return pos, quat

    def _rotation_between_vectors(self, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
        axis = np.cross(v0, v1)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0])
        axis /= norm
        angle = np.arccos(np.dot(v0, v1))
        return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])

    # --- Quaternion Helpers ---------------------------------------------------

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=q.dtype)

    @staticmethod
    def _quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        av, aw = a[:3], a[3]
        bv, bw = b[:3], b[3]
        vec = aw*bv + bw*av + np.linalg.cross(av, bv)
        scl = aw*bw - av.dot(bv)
        return np.hstack((vec, scl))

    @staticmethod
    def _safe_normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(q)
        return q if n < eps else q / n

    @staticmethod
    def _align_sign(q: np.ndarray) -> np.ndarray:
        return q if q[3] >= 0 else -q

    @staticmethod
    def _quat_from_axis_angle(axis, angle: float) -> np.ndarray:
        a = np.array(axis, dtype=float)
        a = a / np.linalg.norm(a)
        s = np.sin(angle/2)
        return np.hstack((a * s, np.cos(angle/2)))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    q = QuaternionFinder(lqr=LQR())
    h = 10000
    pos = []
    for i in range(h):

        p, quat = q.get_path(alt=i)
        pos.append(quat)


    pos = np.array(pos)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:,0], pos[:,1], pos[:,2], label='Missile Trajectory', linewidth=2)
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_zlabel('Altitude Z [m]')
    ax.set_title('Missile Trajectory Path (3D)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


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
        self.q = q if q is not None else np.diag([1, 1, 1, 1, 1, 1])
        self.r = r if r is not None else np.eye(3) * 1

    def _compute_gain(self, inertia_matrix: np.ndarray):
        # System matrices (6x6 and 6x3)
        A = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 3)), np.zeros((3, 3))]
        ])

        B = np.block([
            [np.zeros((3, 3))],
            [np.linalg.inv(inertia_matrix)]
        ])

        P = solve_continuous_are(A, B, self.q, self.r)
        K = np.linalg.inv(self.r) @ B.T @ P
        return K

    def get_torque(self, x: np.ndarray, inertia_matrix: np.ndarray) -> np.ndarray:
        """
        Takers state variables and returns torque command
        :param x: 6x state vector [qx, qy, qz, wx, wy, wz]
        :param inertia_matrix: rocket inertia matrix [Ixx, Iyy, Izz]
        :return: Torque command in body frame [Tx, Ty, Tz]
        """
        inertia_matrix = np.diag(inertia_matrix)
        k = self._compute_gain(inertia_matrix=inertia_matrix)
        torque = -k @ x
        return torque


class QuaternionFinder:
    """
    Generates commanded body‐rates (omega_cmd) for trajectory tracking.
    Body frame: +Z through nose, +X to right, +Y down (right‐handed).
    """

    def __init__(self, lqr=None, profile_csv: str = "cubic_sweep_profile.csv"):
        # Physical parameters & aerodynamic models
        self.lqr = lqr if lqr is not None else LQR()

        self.quat_error = []
        self.pos_error = []

        # Load altitude->(pos,quat) profile
        self._load_lookup_table(profile_csv)

        self.iter = 0
        self.cur_ang = np.deg2rad(0)

    def compute_command_torque(self, rocket_pos, rocket_quat, rocket_omega, inertia_matrix, side_effect = None):
        """TAKE 2
        step 1: detect lateral drift
                find drift with location_exp - location_current

                use a k value to tune how aggressive to treat the drift (correction vector = -k * drift)

                determine desire_thrust in world frame (cor[0], cor[1], 1.0) and normalize

                compute rotation axis using cross product of body forward frame and desired thrust in world frame, normalize
                        if axis is super-small, vehicle is already aligned so q_des = [0,0,0,1]
                        otherwise find angle (radians) using dot product of bodyframe, thrust_world
                        q_desired is rotation vector to quat of axis * angle

                        pass to attitude error


        step 2: detect attitude drift
        need to determine: what to feed into the LQR to get the torque requirement
            attitude error [radx rady radz, wx, wy, wz]

            determine attitude error:
                q_desired rotation matrix * inv of q_current rotation matrix
                and then rotate back to quat and only take first 3 values (we don't want q_w)

            combine current w with q_error

            combine these to form 6 item array (6,)

            pass to lqr
        """
        alt_m       = rocket_pos[2]
        k           = 0.5

        # =================== #
        # -- LATERAL DRIFT -- #
        # =================== #

        # Direction of up on the rocket (through the nose cone)
        body_frame  = np.array([0.0, 0.0, 1.0])

        target_pos, target_quat = self.get_pose_by_altitude(alt=alt_m)

        drift       = target_pos - rocket_pos
        corr_v      = -k * drift

        if np.linalg.norm(corr_v) > 1e-6:
            corr_v /= np.linalg.norm(corr_v)
        else:
            corr_v = np.zeros_like(corr_v)

        desired_thrust_world = np.array([
            corr_v[0],
            corr_v[1],
            1.0
        ])
        desired_thrust_world /= np.linalg.norm(desired_thrust_world)

        rotation_axis = np.linalg.cross(body_frame, desired_thrust_world)

        if np.linalg.norm(rotation_axis) < 1e-8:
            q_drift = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            angle = np.arccos(np.clip(np.dot(body_frame, desired_thrust_world), -1.0, 1.0))
            q_drift = R.from_rotvec(rotvec=(rotation_axis * angle)).as_quat()

        # ===================== #
        # -- TARGET ATTITUDE -- #
        # ===================== #

        q_target    = R.from_quat(quat=target_quat)
        q_drift     = R.from_quat(quat=q_drift)
        q_desired   = q_drift * q_target

        # ============== #
        # -- ROTATION -- #
        # ============== #

        q_current   = R.from_quat(quat=rocket_quat)

        q_err       = q_desired * q_current.inv()
        q_err_v     = q_err.as_quat()[:3]

        # State vector
        x_lqr       = np.concatenate([q_err_v, rocket_omega])

        torque      = self.lqr.get_torque(x=x_lqr, inertia_matrix=inertia_matrix)

        if side_effect:
            print(f"EXPECTED TORQUE: {np.round(torque, 2)}")
            # print(f"DRIFT: {np.round(drift,2)}")
            self.quat_error.append(q_err)
            self.pos_error.append(drift)
            pass

        return torque

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
        # if self.iter == 800:
        #     self.cur_ang += np.deg2rad(90)
        #     self.iter = 0
        # quat = np.array([0, 0, np.sin(self.cur_ang/2), np.cos(self.cur_ang/2)])
        # self.iter += 1
        #
        # return np.array([0,0,alt]), quat
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


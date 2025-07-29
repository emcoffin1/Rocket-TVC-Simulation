"""Program that takes positional data and determines the quaternion error, passes through an LQR, and provides
a corrective quaternion to change the angular velocity"""
import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_are
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import os
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v  # Avoid division by zero
    return v / norm
class LQR:
    def __init__(self, q: np.ndarray = None, r: np.ndarray = None):
        # Places emphasis on quaternion correction
        self.q = q if q is not None else np.diag([1, 1, 1, 1, 1, 1, 1, 1])
        self.r = r if r is not None else np.eye(3) * 1

    def _compute_gain(self, inertia_matrix: np.ndarray, acc_mag: float):
        # System matrices (8,6 and 6x3)
        A = np.zeros((8, 8))
        A[0:3, 3:6] = np.eye(3)

        A[6, 0] = -acc_mag
        A[7, 1] = -acc_mag

        B = np.zeros((8, 3))
        B[3:6, :] = np.linalg.inv(inertia_matrix)

        P = solve_continuous_are(A, B, self.q, self.r)
        K = np.linalg.inv(self.r) @ B.T @ P

        return K

    def get_torque(self, x: np.ndarray, inertia_matrix: np.ndarray, acc_mag: float) -> np.ndarray:
        """
        Takers state variables and returns torque command
        :param x: 6x state vector [qx, qy, qz, wx, wy, wz, vx, vy]
        :param inertia_matrix: rocket inertia matrix [Ixx, Iyy, Izz]
        :return: Torque command in body frame [Tx, Ty, Tz]
        """
        inertia_matrix = np.diag(inertia_matrix)
        k = self._compute_gain(inertia_matrix=inertia_matrix, acc_mag=acc_mag)
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


    def compute_command_torque(self, time, rocket_pos, rocket_quat, rocket_omega, rocket_vel,
                               inertia_matrix, acc_mag, dt, side_effect = None):
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

        alt_m = rocket_pos[2]

        # ================== #
        # -- DESIRED PATH -- #
        # ================== #

        target_pos, target_quat = self.get_pose_by_altitude(alt=alt_m)
        target_pos_future, _ = self.get_pose_by_altitude(alt=alt_m + 2.0)

        # Position and velocity error
        # drift_pos = target_pos - rocket_pos
        drift_pos  = rocket_pos - target_pos
        drift_vel = (target_pos_future - target_pos)
        drift_vel /= np.linalg.norm(drift_vel) + 1e-8

        # Estimate expected speed
        expected_speed = np.linalg.norm(rocket_vel)
        desired_vel = drift_vel * expected_speed
        vel_err = desired_vel - rocket_vel

        # ========================= #
        # -- DESIRED ACCEL VECTOR --#
        # ========================= #

        k_pos = 0.05  # Tune these
        k_vel = 0.5

        desired_acc = k_pos * drift_pos + k_vel * vel_err
        desired_acc[2] += acc_mag  # Add vertical thrust compensation (optional)

        if np.linalg.norm(desired_acc) > 1e-5:
            thrust_dir = desired_acc / np.linalg.norm(desired_acc)
        else:
            thrust_dir = np.array([0, 0, 1.0])  # Default to up

        # ============== #
        # -- ROTATION -- #
        # ============== #

        # Rotate rocket +Z to desired thrust direction
        body_z = np.array([0, 0, 1])
        axis = np.cross(body_z, thrust_dir)
        angle = np.arccos(np.clip(np.dot(body_z, thrust_dir), -1.0, 1.0))

        if np.linalg.norm(axis) < 1e-6:
            q_drift = R.from_quat([0, 0, 0, 1])
        else:
            axis_normalized = axis / np.linalg.norm(axis)
            q_drift = R.from_rotvec(axis_normalized * angle)

        # Target orientation from trajectory
        q_target = R.from_quat(target_quat)
        # q_desired = q_drift * q_target
        q_desired = q_target * q_drift
        q_current = R.from_quat(rocket_quat)
        q_err = q_desired * q_current.inv()

        q_err_v = q_err.as_quat()[:3]

        # State vector: [attitude error, angular velocity, lateral velocity error]
        x_lqr = np.concatenate([q_err_v, rocket_omega, vel_err[:2]])

        torque = self.lqr.get_torque(x=x_lqr, inertia_matrix=inertia_matrix, acc_mag=acc_mag)

        if side_effect:
            self.quat_error.append(q_err)
            self.pos_error.append(q_drift)

        if side_effect and (8.0 < time < 8.1):

            # print(f"EXPECTED: {np.round(torque,2)}")
            # nose_vec = R.from_quat(rocket_quat).apply([0, 0, 1])  # +Z in body
            # trajectory_vec = target_pos_future - target_pos
            # trajectory_vec /= np.linalg.norm(trajectory_vec)
            #
            # alignment_error = np.arccos(np.clip(np.dot(nose_vec, trajectory_vec), -1.0, 1.0))
            # print(f"Alignment angle error: {np.degrees(alignment_error):.2f} deg")
            #
            # dir_trajectory = normalize(target_pos_future - target_pos)
            # e_xy = target_pos[:2] - rocket_pos[:2]
            # corr_vec = k_pos * normalize(np.array([e_xy[0], e_xy[1], 0.0]))  # No vertical drift correction
            # blended_dir = normalize(dir_trajectory + corr_vec)
            # dot = np.dot(nose_vec, blended_dir)
            # print(f"Thrust alignment cosine: {dot:.6f}, angle: {np.degrees(np.arccos(dot)):.2f}°")
            pass

        if side_effect:
            # print(f"EXPECTED: {np.round(torque,2)}")
            # print(f"DEV: {np.round(drift_pos,3)}")
            print(f"CUR: {np.round(rocket_pos,2)} || EXP: {np.round(target_pos,2)} || ERR: DEV: {np.round(drift_pos,3)}")
            print(f"EXPECTED TORQUE: {np.round(torque,3)}")
            # print(f"EXPECTED: {np.round(target_pos, 2)}")
            # print(f"ACTUAL:   {np.round(rocket_pos, 2)}")
            pass

        if side_effect and torque[1] > 1e-5:
            # print(round(time,2), torque[1])
            pass


        return torque

    def compute_q_des_fixed_roll(self, a_des, angle_rad):
        z_axis = normalize(a_des)

        y_ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(z_axis, y_ref)) > 0.95:
            y_ref = np.array([1.0, 0.0, 0.0])

        y_proj = normalize(y_ref - np.dot(y_ref, z_axis) * z_axis)

        x_proj = np.linalg.cross(y_proj, z_axis)

        cos_r = np.cos(angle_rad)
        sin_r = np.sin(angle_rad)

        x_axis = cos_r * x_proj + sin_r * y_proj
        y_axis = -sin_r * x_axis + cos_r * y_proj

        R_world_to_body = np.stack((x_axis, y_axis, z_axis), axis=1)
        q_des = R.from_matrix(matrix=R_world_to_body).as_quat()

        return q_des

    def compute_a_des(self):
        """
        Computes the expected acceleration using velocity vectors between current and future positions
        Can be updated if velocities are introduced into trajectory
        Currently uses path following trajectories
        :return:
        """

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
    def normalize(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)

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


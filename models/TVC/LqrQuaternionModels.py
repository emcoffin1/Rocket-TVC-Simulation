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
        self.q = q if q is not None else np.eye(3)
        self.r = r if r is not None else np.eye(3)
        self.k = self._compute_gain()


    def _compute_gain(self):
        """Computes gain for the LQR"""
        A = np.zeros((3, 3))  # System is static (error evolves through q_dot)
        B = np.eye(3)  # Control input maps directly
        P = solve_continuous_are(A, B, self.q, self.r)
        return np.linalg.inv(self.r) @ B.T @ P

    def get_Qdot(self, q_e: np.ndarray):
        """
        Takers quaternion error and return desired quaternion derivative
        :param q_e: error quaternion
        :return: quaternion derivative
        """
        q_vec = q_e[:3]
        dq_vec = -self.k @ q_vec
        return np.array([dq_vec[0], dq_vec[1], dq_vec[2], 0.0])


class QuaternionFinder:
    """Quaternions of the form [x, y, z, w]"""
    def __init__(self, lqr:object = None):
        self.earth_frame = np.array([0, 0.0, 0.0, 0.0])
        self.LQR = lqr if lqr is not None else LQR()
        self.error = []
        self.omega_command = []
        self.iteration = 0
        self.drift = 2
        self.prev_w_cmd = 0
        self.prev_w_body = 0

        self._load_lookup_table()


    def getAngularVelocityCorrection(self, time: float,
                                     dt: float,
                                     rocket_loc: np.ndarray,
                                     rocket_quat: np.ndarray,
                                     rocket_omega: np.ndarray,
                                     side_effect: bool = True) -> np.ndarray:
        """
        Computes angular velocity correction with PD damping and cross-track bias.
        """
        alt_m = rocket_loc[2]

        # ————————————————————————————————————————————————
        # 1) Build blended direction exactly as before
        pos_des, _ = self._get_pose_by_altitude(alt=alt_m)
        pos_fut, _ = self._get_pose_by_altitude(alt=alt_m + 5)

        dir_tan = pos_fut - pos_des
        dir_tan /= np.linalg.norm(dir_tan)

        e_xy = pos_des[:2] - rocket_loc[:2]
        k_pos = 0.01  # up from 0.005—stronger pull
        corr_xy = (k_pos * e_xy / np.linalg.norm(e_xy)) if np.linalg.norm(e_xy) > 1e-6 else np.zeros(2)

        blended = dir_tan + np.array([corr_xy[0], corr_xy[1], 0.0])
        blended /= np.linalg.norm(blended)

        r_blend, _ = R.align_vectors(a=[blended], b=[[0, 0, 1]])
        q_t = self._safe_normalize(r_blend.as_quat())

        # ————————————————————————————————————————————————
        # 2) Quaternion‐error
        q_r = self._safe_normalize(rocket_quat)
        q_e = self._quat_mult(self._quat_conj(q_r), q_t)
        q_e = self._safe_normalize(q_e)

        # ————————————————————————————————————————————————
        # 3) Proportional command (via LQR)
        qdot = self.LQR.get_Qdot(q_e=q_e)
        omega_quat = self._quat_mult(qdot, self._quat_conj(q_e))
        w_p = 2.0 * omega_quat[:3]  # P‐term

        # ————————————————————————————————————————————————
        # 4) Derivative (damping) term on measured body rates
        Kd = 0.1  # tune this
        w_d = -Kd * self.prev_w_body  # self.prev_w_body is rocket’s last ω from state

        # Combine P + D
        w_cmd = w_p + w_d

        # ————————————————————————————————————————————————
        # 5) Saturate to max body‑rate
        max_rate = np.deg2rad(2.5)
        norm = np.linalg.norm(w_cmd)
        if norm > max_rate:
            w_cmd *= (max_rate / norm)

        # ————————————————————————————————————————————————
        # 6) Low‑pass filter on the command
        alpha = dt / (0.05 + dt)  # 50 ms time constant
        w_cmd = alpha * w_cmd + (1 - alpha) * self.prev_w_cmd
        self.prev_w_cmd = w_cmd

        # ————————————————————————————————————————————————
        # 7) Logging & state update
        if side_effect:
            angle_err = 2 * np.degrees(np.arccos(np.clip(q_e[3], -1, 1)))
            self.error.append([angle_err, alt_m])
            self.omega_command.append([time, w_cmd])
            print(self.omega_command[-1])

        # store last body‐rate for D‐term next step
        self.prev_w_body = rocket_omega  # pull from your state

        return w_cmd

    def _find_quaternion_error(self, trajectory: np.ndarray, rocket: np.ndarray):
        """Determines the quaternion error, performs safe normalize"""
        rocket_i = self._quat_conj(rocket)
        q_e = self._quat_mult(trajectory, rocket_i)
        q_e = self._safe_normalize(q=q_e)
        return q_e

    def _quat_conj(self, q):
        """
        Performs a conjugation on a quaternion
        :param q: quaternion
        :return: conjugated quaternion
        """
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def _quat_mult(self, q1, q2):
        """
        Performs quaternion multiplication
        :param q1: quaternion
        :param q2: quaternion
        :return: multiplied quaternion
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ])


    def _get_pose_by_altitude(self, alt: float):

        # pos = np.array([0, 0, alt])
        # x = np.sin(np.deg2rad(self.drift)/2)
        # w = np.cos(np.deg2rad(self.drift)/2)
        # quat = np.array([x, 0, 0, w])
        #
        # self.iteration += 1
        # if self.iteration == 1600:
        #     self.drift *= -1
        #     self.iteration = 0
        #
        # return pos, quat

        try:
            p = 1/0
            pos = np.array([
                self.interp_x(alt),
                self.interp_y(alt),
                alt
            ])
            quat = np.array([
                self.interp_qx(alt),
                self.interp_qy(alt),
                self.interp_qz(alt),
                self.interp_qw(alt)
            ])
            return pos, quat
        except Exception as e:
            pos = np.array([0, 0, alt])
            quat = np.array([0, 0, 0, 1])
            # print(f"QuaternionFinder lookup error: {e}")
            return pos, quat

    def _load_lookup_table(self):
        try:
            PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(PROJECT_ROOT, "cubic_sweep_profile.csv")

            df = pd.read_csv(filename)
            # Build interpolators
            self.interp_x = interp1d(df['z'], df['x'], kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.interp_y = interp1d(df['z'], df['y'], kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.interp_qx = interp1d(df['z'], df['qx'], kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.interp_qy = interp1d(df['z'], df['qy'], kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.interp_qz = interp1d(df['z'], df['qz'], kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.interp_qw = interp1d(df['z'], df['qw'], kind='cubic', bounds_error=False, fill_value='extrapolate')
        except Exception as e:
            print(f"ERROR:      {e}")
            print(f"LOCATION:   QuaternionFinder._load_lookup_table()")

    def _safe_normalize(self, q, eps = 1e-6):
        """Safely normalizes quaternion"""
        norm = np.linalg.norm(q)
        if norm < eps:
            return q
        return q / norm

    def get_path(self, alt):
        return self._get_pose_by_altitude(alt=alt)

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


"""Program that takes positional data and determines the quaternion error, passes through an LQR, and provides
a corrective quaternion to change the angular velocity"""
import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_are
from scipy.interpolate import interp1d
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

        self._load_lookup_table()


    def getAngularVelocityCorrection(self, rocket_loc: np.array, rocket_quat: np.array, side_effect=True):
        """
        Computes and returns the angular velocity correction to maintain trajectory
        w = 2 * q^-1 * qdot
        :param rocket_loc: rocket location
        :param rocket_quat: rocket quaternion [
        :return: angular velocity to correct quaternion error
        """
        # Get Target Paths
        alt_m = rocket_loc[2]
        l_t, q_t = self._get_pose_by_altitude(alt=alt_m)
        q_t = self._safe_normalize(q=q_t)

        # Attitude Correction Quaternion
        q_e = self._find_quaternion_error(trajectory=q_t, rocket=rocket_quat)

        # Translational Correction
        l_e = l_t - rocket_loc
        if np.linalg.norm(l_e) == 0:
            q_trans_e = np.array([0,0,0,1])
        else:

            target_v_body = np.array([0, 0, 1])   # Nose up z axis
            c_v = np.cross(target_v_body, l_e)    # correction vector should be based on the quaternion correction
            c = np.dot(target_v_body, l_e)
            s = np.sqrt((1 + c) * 2)
            q_trans_e = np.array([c_v[0]/s, c_v[1]/s, c_v[2]/s, s/2])
            q_trans_e = self._safe_normalize(q=q_trans_e)
            if np.isclose(c, -1.0):
                q_trans_e = np.array([1, 0, 0, 0])  # 180 deg flip around x


        q_e_combined = self._quat_mult(q_trans_e, q_e)     # Multiply to first apply the translation and then the attitude
        q_e_combined = self._safe_normalize(q=q_e_combined)
        qdot = self.LQR.get_Qdot(q_e=q_e_combined)
        q_corr_i = self._quat_conj(q=q_e_combined)

        omega_quat = self._quat_mult(qdot, q_corr_i)
        # omega_quat = self._safe_normalize(q=omega_quat)
        w = 2 * omega_quat[:3]

        # if side_effect:
        #     angle_rad = 2 * np.arccos(np.clip(q_e[3], -1.0, 1.0))  # q_e[3] is the scalar part
        #     angle_deg = np.degrees(angle_rad)
        #     print(f"Angle deviation: {angle_deg:.2f}°")


        return w

    def _find_translational_error(self, trajectory: np.array, rocket: np.array):
        """
        Determines the translational error in trajectory vs current location
        :param trajectory: trajectory location [x,y,z]
        :param rocket: rocket location [x,y,z]
        :return: translational error [x,y,z]
        """

    def _find_quaternion_error(self, trajectory: np.ndarray, rocket: np.ndarray):

        rocket_i = self._quat_conj(rocket)
        q_e = self._quat_mult(trajectory, rocket_i)
        q_e = self._safe_normalize(q=q_e)
        return q_e


    def _get_trajectory_attitude(self, alt_m: float = 0):
        """Accesses attitude trajectory based on altitude and acquires quaternion"""
        if alt_m == 0:
            return np.array([0, 0, 0, 1])

        return np.array([0, 0, 0, 1])

    def _get_trajectory_location(self, alt_m: float = 0):
        """Access location trajectory based on altitude"""
        if alt_m == 0:
            return np.array([0, 0, alt_m])
        return np.array([0, 0, alt_m])

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
            # print(f"QuaternionFinder lookup error: {e}")
            return np.array([0, 0, alt]), np.array([0, 0, 0, 1])

    def _load_lookup_table(self):
        try:
            PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(PROJECT_ROOT, "missile_profile.csv")
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R

    # --- Simulation Setup ---
    q = np.eye(3) * 1
    r = np.eye(3) * 1
    traj = np.array([0, 0, 0, 1])
    att = np.array([1, 4, 0, 1])
    loc = np.array([0, 0, 1])

    print(f"TARGET TRAJECTORY: {traj }")

    thrust = 5000
    p_y_inertia = 156.4
    vehicle_height = 5.0
    dt = 0.01
    quat_tol = 1e-3
    max_iters = 500

    quat = QuaternionFinder(lqr=LQR(q=q, r=r))

    # --- Data Logging ---
    quaternions = []
    angular_vels = []
    errors = []

    # --- Main Loop ---
    for step in range(max_iters):
        att = att / np.linalg.norm(att)
        w = quat.getAngularVelocityCorrection(rocket_loc=loc, rocket_quat=att)

        theta_1 = p_y_inertia / (thrust * vehicle_height)
        theta_x = theta_1 * w[0] / dt
        theta_y = theta_1 * w[1] / dt

        omega_vec = np.array([-theta_x, -theta_y, 0.0])
        rotation_increment = R.from_rotvec(omega_vec)

        current_rot = R.from_quat(att)
        new_rot = rotation_increment * current_rot
        att = new_rot.as_quat()

        # Log data
        quaternions.append(att.copy())
        angular_vels.append(np.linalg.norm(w))
        errors.append(np.linalg.norm(att - traj))

        # Debug print
        print(f"[{step}] q: {np.round(att, 4)} | ω: {np.round(w, 3)}")

        if errors[-1] < quat_tol:
            print(f"\n✅ Aligned in {step} steps.")
            break
    else:
        print("\n❌ Did not converge.")

    # --- Plotting ---
    quaternions = np.array(quaternions)
    angular_vels = np.array(angular_vels)
    errors = np.array(errors)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(quaternions[:, 0], label="x")
    axs[0].plot(quaternions[:, 1], label="y")
    axs[0].plot(quaternions[:, 2], label="z")
    axs[0].plot(quaternions[:, 3], label="w")
    axs[0].set_ylabel("Quaternion")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(angular_vels, label="|ω|", color="darkorange")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].grid(True)

    axs[2].plot(errors, label="Quat Error", color="crimson")
    axs[2].set_ylabel("Quaternion Error")
    axs[2].set_xlabel("Iteration")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()



"""Program that takes positional data and determines the quaternion error, passes through an LQR, and provides
a corrective quaternion to change the angular velocity"""
import numpy as np
from scipy.linalg import solve_continuous_are


class LQR:
    def __init__(self, q: np.ndarray = None, r: np.ndarray = None):
        self.q = q if q is not None else np.eye(3)
        self.r = r if q is not None else np.eye(3)
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

    def getAngularVelocityCorrection(self, rocket_loc: np.array, rocket_quat: np.array):
        """
        Computes and returns the angular velocity correction to maintain trajectory
        w = 2 * q^-1 * qdot
        :param rocket_loc: rocket location
        :param rocket_quat: rocket quaternion [
        :return: angular velocity to correct quaternion error
        """
        alt_m = rocket_loc[2]

        # Attitude Correction Quaternion
        q_t = self._get_trajectory_attitude(alt_m=alt_m)
        q_e = self._find_quaternion_error(trajectory=q_t, rocket=rocket_quat)

        # Translational Correction
        l_t = self._get_trajectory_location(alt_m=alt_m)
        l_e = l_t - rocket_loc
        l_e = l_e / np.linalg.norm(l_e)         # Normalized error displacement

        target_v_body = np.array([0, 0, 1])   # Nose up z axis
        c_v = np.linalg.cross(target_v_body, l_e)    # correction vector should be based on the quaternion correction
        c = np.dot(target_v_body, l_e)
        s = np.sqrt((1 + c) * 2)
        q_trans_e = np.array([c_v[0]/s, c_v[1]/s, c_v[2]/s, s/2])
        q_trans_e = q_trans_e / np.linalg.norm(q_trans_e)

        q_e_combined = self._quat_mult(q_trans_e, q_e)     # Multiply to first apply the translation and then the attitude

        qdot = self.LQR.get_Qdot(q_e=q_e_combined)

        q_corr_i = self._quat_conj(q=q_e_combined)

        omega_quat = self._quat_mult(qdot, q_corr_i)

        w = 2 * omega_quat[:3]
        return w

    def _find_translational_error(self, trajectory: np.array, rocket: np.array):
        """
        Determines the translational error in trajectory vs current location
        :param trajectory: trajectory location [x,y,z]
        :param rocket: rocket location [x,y,z]
        :return: translational error [x,y,z]
        """


    def _get_trajectory_attitude(self, alt_m: float = 0):
        """Accesses attitude trajectory based on altitude and acquires quaternion"""
        if alt_m == 0:
            return np.array([0, 0, 0, 1])

    def _get_trajectory_location(self, alt_m: float = 0):
        """Access location trajectory based on altitude"""
        if alt_m == 0:
            return np.array([0, 0, 0])

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



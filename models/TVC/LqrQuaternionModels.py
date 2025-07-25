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
    """
    Generates commanded body‐rates (omega_cmd) for trajectory tracking.
    Body frame: +Z through nose, +X to right, +Y down (right‐handed).
    """

    def __init__(self,
                 mass: float,
                 lift_func,
                 drag_func,
                 d_lift_dalpha,
                 d_drag_dalpha,
                 lqr=None,
                 profile_csv: str = "cubic_sweep_profile.csv"):
        # Physical parameters & aerodynamic models
        self.mass = mass
        self.L       = lift_func
        self.D       = drag_func
        self.dL_da   = d_lift_dalpha
        self.dD_da   = d_drag_dalpha

        # Your quaternion‐LQR controller (must implement compute_qdot(q_err, pos_err2d))
        self.lqr = lqr if lqr is not None else LQR()

        # State tracking (if you later want integrators or filters)
        self.q_err_prev = np.zeros(4)

        # Load altitude→(pos,quat) profile
        self._load_lookup_table(profile_csv)

    def solve_alpha_thrust(self, V, Vdot, y,
                           w_Ny, w_Nz,
                           tol=1e-6, max_iter=5):
        """
        Newton solve for AoA (α), bank (μ), and thrust T per NASA eqns (54)-(55).
        Returns (α, μ, T).
        """
        m, L, D, dL, dD = self.mass, self.L, self.D, self.dL_dα, self.dD_dα

        # initial guesses
        a = 0.01
        T = m * 9.81
        u = np.arctan2(V*w_Ny + 9.81, V*w_Nz * np.cos(y))

        for _ in range(max_iter):
            f1 = L(a) + T*np.sin(u)        - m*V*w_Ny
            f2 = D(a)*np.cos(u) + T*np.cos(u) - m*(Vdot + 9.81)

            J = np.array([
                [ dL(a),       np.sin(u)],
                [ dD(a)*np.cos(u), np.cos(u)]
            ])

            delta = np.linalg.solve(J, -np.array([f1, f2]))
            a += delta[0];  T += delta[1]

            if np.linalg.norm(delta) < tol:
                break

        return a, u, T

    def compute_desired_quaternion(self,
                                   V: float,
                                   Vdot: float,
                                   y: float,
                                   w_Ny: float = 0.0,
                                   w_Nz: float = 0.0):
        """
        Step 2: Outer guidance → desired quaternion q_des and thrust T.
        """
        a, u, T = self.solve_alpha_thrust(V, Vdot, y, w_Ny, w_Nz)

        # Bank about N‑x
        q_NW = self._quat_from_axis_angle([1, 0, 0], u)
        # Sideslip about W‑y (β≈0 ⇒ identity)
        q_WC = self._quat_from_axis_angle([0, 1, 0], 0.0)
        # AoA about C‑y
        q_CB = self._quat_from_axis_angle([0, 1, 0], a)

        # Compose: q_NC = q_NW ⊗ q_WC ⊗ q_CB
        q_des = self._quat_mult(q_NW, self._quat_mult(q_WC, q_CB))
        q_des = self._align_sign(self._safe_normalize(q_des))

        return q_des, T

    def compute_command_omega(self,
                               rocket_pos: np.ndarray,
                               rocket_quat: np.ndarray,
                               V: float,
                               Vdot: float,
                               gamma: float):
        """
        Full pipeline:
         1) Get q_des from guidance
         2) Compute attitude error q_err
         3) Call LQR for q_dot
         4) Invert kinematics → omega_cmd (3,)
        """
        alt_m = rocket_pos[2]
        # 1) desired
        q_des, _ = self.compute_desired_quaternion(V, Vdot, gamma)

        # 2) attitude error: q_err = q_body⁻¹ ⊗ q_des
        q_err = self._quat_mult(self._quat_conj(rocket_quat), q_des)
        q_err = self._align_sign(self._safe_normalize(q_err))

        # 3) lateral drift error in XY (if your LQR needs it)
        pos_des, _ = self.get_pose_by_altitude(alt=float(alt_m))
        pos_err2d = pos_des[:2] - rocket_pos[:2]

        # 4) LQR → quaternion derivative
        q_dot = self.lqr.compute_qdot(q_err, pos_err2d)

        # 5) ω_cmd = 2·vec( q_dot ⊗ q_body⁻¹ )
        omega_cmd = 2.0 * self._quat_mult(q_dot, self._quat_conj(rocket_quat))[:3]
        return omega_cmd

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

    def get_pose_by_altitude(self, alt: float):
        """
        Returns (pos, quat) at the given altitude via cubic interpolation.
        """
        pos = np.array([ self._ix(alt),
                         self._iy(alt),
                         alt ])
        quat = np.array([ self._iqx(alt),
                          self._iqy(alt),
                          self._iqz(alt),
                          self._iqw(alt) ])
        return pos, self._safe_normalize(quat)

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


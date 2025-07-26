import numpy as np

from ..TVC.LqrQuaternionModels import *
from scipy.spatial.transform import Rotation as R
# from ..Structural.StructuralModel import StructuralModel    # DELETE ME

class TVCStructure:
    """
    Thrust Vector Control + Gimbal Dynamics.
    Integrates QuaternionFinder ω_cmd → gimbal angles.
    Body frame: +Z nose, +X right, +Y down.
    """

    def __init__(self,
                 lqr_controller: object,
                 structure_model: object,
                 aero_model: object,
                 trajectory_csv: str = "cubic_sweep_profile.csv"):
        self.struct = structure_model
        self.aero   = aero_model
        self.height = self.struct.length

        self.r_inertia = self.struct.I[2]
        self.p_y_inertia = self.struct.I[0]

        self.quaternion = QuaternionFinder(
            mass=self.struct.mass_current,
            lqr=lqr_controller,
            profile_csv=trajectory_csv
        )

        # State
        self.thrust = 0.0
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.d_theta_x = 0.0
        self.d_theta_y = 0.0
        self.gimbal_orientation = R.identity()

        # Gimbal limits & dynamics
        self.gimbal_tau = 0.2
        self.max_rate   = np.deg2rad(100)
        self.max_angle  = np.deg2rad(10)

        # Log
        self.gimbal_log = []

    def update_variables(self, thrust: float):
        """Call each time step to refresh thrust & inertias."""
        self.thrust = thrust
        # reload inertias in case structure_model changes
        self.I_roll     = self.struct.roll_inertia
        self.I_pitchyaw = self.struct.pitch_yaw_inertia


    def compute_gimbal_orientation(self,
                                   time: float,
                                   dt: float,
                                   rocket_pos: np.ndarray,
                                   rocket_quat: np.ndarray,
                                   side_effect: bool = False):
        """
        1) ω_cmd via QuaternionFinder
        2) map ω_cmd → raw gimbal angles
        3) clamp angles, apply first‐order filter & rate limits
        4) update cumulative gimbal_orientation (Rotation)
        """
        # Reset if no thrust
        if self.thrust <= 0.0:
            self.theta_x    = 0.0
            self.theta_y    = 0.0
            self.d_theta_x  = 0.0
            self.d_theta_y  = 0.0
            return np.zeros(3)

        # =================== #
        # -- OMEGA COMMAND -- #
        # =================== #

        omega_cmd, q_err, pos_err       = self.quaternion.compute_command_omega(
            rocket_pos=rocket_pos, rocket_quat=rocket_quat, side_effect=side_effect)

        # if side_effect:
        #     print(f"[{time:.2f}s] ω_cmd = {omega_cmd}")

        # ======================= #
        # -- GIMBAL CORRECTION -- #
        # ======================= #

        # Determine the raw angle changes
        gain            = self.struct.I[0] / (self.thrust * (self.height - self.struct.cm_current))
        theta_x_raw     = gain * omega_cmd[0]
        theta_y_raw     = gain * omega_cmd[1]

        # if abs(q_err[2]) < 1e-3:  # replace [2] with whatever index holds your roll‑error
        #     theta_x_raw = 0.0  # zero out nozzle roll if no roll error

        # gate roll unless there’s roll error (q_err_x)
        if abs(q_err[0]) < 1e-3:
            theta_x_raw = 0.0

        # gate pitch unless there’s pitch error (q_err_y)
        if abs(q_err[1]) < 1e-3:
            theta_y_raw = 0.0

        # Clamp to physical gimbal limits
        theta_x_raw     = np.clip(theta_x_raw, -self.max_angle, self.max_angle)
        theta_y_raw     = np.clip(theta_y_raw, -self.max_angle, self.max_angle)

        # Pass through first order filter and rate limit
        alpha           = dt / (self.gimbal_tau + dt)
        tx_filt         = (1-alpha)*self.theta_x + alpha*theta_x_raw
        ty_filt         = (1-alpha)*self.theta_y + alpha*theta_y_raw
        dtx             = np.clip(tx_filt - self.theta_x, -self.max_rate*dt, self.max_rate*dt)
        dty             = np.clip(ty_filt - self.theta_y, -self.max_rate*dt, self.max_rate*dt)

        # ================== #
        # -- STORE ANGLES -- #
        # ================== #

        self.theta_x    += dtx
        self.theta_y    += dty
        self.d_theta_x  = dtx / dt
        self.d_theta_y  = dty / dt

        # =================== #
        # -- UPDATE GIMBAL -- #
        # =================== #

        sx = np.sin(self.theta_x)
        cx = np.cos(self.theta_x)
        sy = np.sin(self.theta_y)
        cy = np.cos(self.theta_y)
        thrust_dir_body = np.array([
            + sy,  # X‐component from pitch
            - sx,  # Y‐component from roll
            cx * cy  # Z‐component
        ])

        if side_effect:
            self.gimbal_log.append([np.rad2deg(self.theta_x), np.rad2deg(self.theta_y)])
            # print( np.round(np.rad2deg(self.theta_x),2), np.round(np.rad2deg(self.theta_y),2), np.round(omega_cmd,2))
            # print(self.gimbal_orientation)

        return thrust_dir_body

    def _compute_gimbal_orientation(self,
                                   time: float,
                                   dt: float,
                                   rocket_pos: np.ndarray,
                                   rocket_quat: np.ndarray,
                                   side_effect: bool = False) -> R:
        """
        1) ω_cmd via QuaternionFinder
        2) map ω_cmd → raw gimbal angles
        3) clamp angles, apply first‐order filter & rate limits
        4) update cumulative gimbal_orientation (Rotation)
        """
        # Reset if no thrust
        if self.thrust <= 0.0:
            self.theta_x    = 0.0
            self.theta_y    = 0.0
            self.d_theta_x  = 0.0
            self.d_theta_y  = 0.0
            return self.gimbal_orientation

        # =================== #
        # -- OMEGA COMMAND -- #
        # =================== #

        omega_cmd, q_err, pos_err       = self.quaternion.compute_command_omega(
            rocket_pos=rocket_pos, rocket_quat=rocket_quat, side_effect=side_effect)

        # if side_effect:
        #     print(f"[{time:.2f}s] ω_cmd = {omega_cmd}")

        # ======================= #
        # -- GIMBAL CORRECTION -- #
        # ======================= #

        # Determine the raw angle changes
        gain            = self.struct.I[0] / (self.thrust * (self.height - self.struct.cm_current))
        theta_x_raw     = gain * omega_cmd[0]
        theta_y_raw     = gain * omega_cmd[1]

        # if abs(q_err[2]) < 1e-3:  # replace [2] with whatever index holds your roll‑error
        #     theta_x_raw = 0.0  # zero out nozzle roll if no roll error

        # # gate roll unless there’s roll error (q_err_x)
        # if abs(q_err[0]) < 1e-3:
        #     theta_x_raw = 0.0
        #
        # # gate pitch unless there’s pitch error (q_err_y)
        # if abs(q_err[1]) < 1e-3:
        #     theta_y_raw = 0.0

        # Clamp to physical gimbal limits
        theta_x_raw     = np.clip(theta_x_raw, -self.max_angle, self.max_angle)
        theta_y_raw     = np.clip(theta_y_raw, -self.max_angle, self.max_angle)

        # Pass through first order filter and rate limit
        alpha           = dt / (self.gimbal_tau + dt)
        tx_filt         = (1-alpha)*self.theta_x + alpha*theta_x_raw
        ty_filt         = (1-alpha)*self.theta_y + alpha*theta_y_raw
        dtx             = np.clip(tx_filt - self.theta_x, -self.max_rate*dt, self.max_rate*dt)
        dty             = np.clip(ty_filt - self.theta_y, -self.max_rate*dt, self.max_rate*dt)

        # ================== #
        # -- STORE ANGLES -- #
        # ================== #

        self.theta_x    += dtx
        self.theta_y    += dty
        self.d_theta_x  = dtx / dt
        self.d_theta_y  = dty / dt

        # =================== #
        # -- UPDATE GIMBAL -- #
        # =================== #

        r_step_x        = R.from_rotvec(rotvec=[dtx, 0, 0])
        # r_step_x        = R.from_rotvec(rotvec=[0, 0, 0])
        r_step_y        = R.from_rotvec(rotvec=[0, -dty, 0])
        self.gimbal_orientation = (r_step_y * r_step_x) * self.gimbal_orientation


        if side_effect:
            self.gimbal_log.append([np.rad2deg(self.theta_x), np.rad2deg(self.theta_y)])
            print( np.round(np.rad2deg(self.theta_x),2), np.round(np.rad2deg(self.theta_y),2), np.round(omega_cmd,2))
            # print(self.gimbal_orientation)

        return self.gimbal_orientation

    def update_variables_(self, thrust: float):
        """
        Function to update variables, called every time step, most values accessed passively
        :param thrust: current effective thrust [N]
        """
        self.thrust = thrust
        self.r_inertia = self.struct.roll_inertia
        self.p_y_inertia = self.struct.pitch_yaw_inertia

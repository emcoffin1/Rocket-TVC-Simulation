import numpy as np

from ..TVC.LqrQuaternionModels import *
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
        self.height = self.struct.height

        self.quaternion = QuaternionFinder(
            mass=self.struct.mass,
            lift_func=self.aero.lift,
            drag_func=self.aero.drag,
            d_lift_dalpha=self.aero.d_lift,
            d_drag_dalpha=self.aero.d_drag,
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
        self.max_rate   = np.deg2rad(10)
        self.max_angle  = np.deg2rad(5)

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
                                   rocket_vel: np.ndarray,
                                   rocket_acc: np.ndarray,
                                   side_effect: bool = False) -> R:
        """
        1) ω_cmd via QuaternionFinder
        2) map ω_cmd → raw gimbal angles
        3) clamp angles, apply first‐order filter & rate limits
        4) update cumulative gimbal_orientation (Rotation)
        """
        # Reset if no thrust
        if self.thrust <= 0.0:
            self.theta_x = 0.0
            self.theta_y = 0.0
            self.d_theta_x = 0.0
            self.d_theta_y = 0.0
            return self.gimbal_orientation

        # =============================== #
        # -- COMPUTE FLIGHT KINEMATICS -- #
        # =============================== #

        # Compute flight kinematics
        V           = np.linalg.norm(rocket_vel)

        # projection of accel onto velocity
        Vdot        = np.dot(rocket_acc, rocket_vel) / (V + 1e-8)

        # flight‐path angle γ = arctan2(v_z, sqrt(v_x²+v_y²))
        gamma       = np.arctan2(rocket_vel[2], np.linalg.norm(rocket_vel[:2]) + 1e-8)

        # ====================== #
        # -- ATTITUDE COMMAND -- #
        # ====================== #

        omega_cmd = self.quaternion.compute_command_omega(
            rocket_pos=rocket_pos,
            rocket_quat=rocket_quat,
            V=float(V),
            Vdot=Vdot,
            gamma=gamma
        )
        if side_effect:
            print(f"[{time:.2f}s] ω_cmd = {omega_cmd}")

        # =========================== #
        # -- GIMBAL CORRECTION -- #
        # =========================== #

        # --- 3) Map ω_cmd → raw gimbal deflections θ_raw = I/(T·h) · ω_cmd ---
        gain            = self.I_pitchyaw / (self.thrust * self.height + 1e-8)
        theta_x_raw     = gain * omega_cmd[0]
        theta_y_raw     = gain * omega_cmd[1]

        # clamp to physical gimbal limits
        theta_x_raw     = np.clip(theta_x_raw, -self.max_angle, self.max_angle)
        theta_y_raw     = np.clip(theta_y_raw, -self.max_angle, self.max_angle)

        alpha           = dt / (self.gimbal_tau + dt)
        tx_filt         = (1-alpha)*self.theta_x + alpha*theta_x_raw
        ty_filt         = (1-alpha)*self.theta_y + alpha*theta_y_raw
        dtx             = np.clip(tx_filt - self.theta_x, -self.max_rate*dt, self.max_rate*dt)
        dty             = np.clip(ty_filt - self.theta_y, -self.max_rate*dt, self.max_rate*dt)

        # update stored angles
        self.theta_x    += dtx
        self.theta_y    += dty
        self.d_theta_x  = dtx / dt
        self.d_theta_y  = dty / dt

        # --- 6) Build incremental Rotation and update orientation ---
        # negative x‐step because gimbal rotation opposite body‐rate sign
        r_step_x        = R.from_rotvec([-dtx, 0, 0])
        r_step_y        = R.from_rotvec([0, dty, 0])
        self.gimbal_orientation = (r_step_y * r_step_x) * self.gimbal_orientation

        return self.gimbal_orientation

    def update_variables_(self, thrust: float):
        """
        Function to update variables, called every time step, most values accessed passively
        :param thrust: current effective thrust [N]
        """
        self.thrust = thrust
        self.r_inertia = self.struct.roll_inertia
        self.p_y_inertia = self.struct.pitch_yaw_inertia


# if __name__ == "__main__":
#     from models.Structural.StructuralModel import StructuralModel
#     from models.Engine.EngineModels import Engine
#     q = np.eye(3) * 5
#     r = np.eye(3) * 1
#
#     tvc = TVCStructure(lqr=LQR(q=q, r=r), structure_model=StructuralModel(engine_class=Engine(),  liquid_total_ratio=0.56425))





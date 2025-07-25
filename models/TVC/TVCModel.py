import numpy as np

from ..TVC.LqrQuaternionModels import *
# from ..Structural.StructuralModel import StructuralModel    # DELETE ME




class TVCStructure:
    def __init__(self, lqr: object, structure_model: object):
        self.structural_model = structure_model
        self.quaternion = QuaternionFinder(lqr=lqr)
        self.thrust = 0.0

        # Inertias and geometry
        self.r_inertia = structure_model.roll_inertia
        self.p_y_inertia = structure_model.pitch_yaw_inertia
        self.vehicle_height = structure_model.length

        # Gimbal actuation limits and dynamics
        self.gimbal_tau = 0.2  # First-order lag time constant [s]
        self.max_rate = np.deg2rad(10)  # Max gimbal slew rate [rad/s]

        # Gimbal state (persistent)
        self.theta_x_prev = 0.0  # Previous filtered gimbal angle around x
        self.theta_y_prev = 0.0  # Previous filtered gimbal angle around y
        self.d_theta_x = 0.0     # Change in angle (used for derivative update)
        self.d_theta_y = 0.0     # Change in angle (used for derivative update)
        self.gimbal_orientation = R.identity()  # Cumulative gimbal rotation
        self.max_angle = np.deg2rad(5)

    def compute_gimbal_orientation(self, time, dt: float, rocket_location: np.ndarray, rocket_quat: np.ndarray,
                                   rocket_omega: np.ndarray, side_effect=False) -> R:
        """
        Computes and returns the cumulative gimbal orientation as a Rotation object.
        This should be applied to the thrust vector in body frame.

        :param dt: Time step [s]
        :param rocket_location: Current position [x, y, z]
        :param rocket_quat: Current attitude quaternion [x, y, z, w]
        :param side_effect: Debug flag
        :return: Updated gimbal orientation (Rotation object)
        """
        # If no thrust, reset rates and return
        if self.thrust is None or self.thrust <= 0.0:
            self.d_theta_x = 0.0
            self.d_theta_y = 0.0
            return self.gimbal_orientation

        # 1) Get controller command (body-rate) in rad/s
        w_cmd = self.quaternion.getAngularVelocityCorrection(
            time=time,
            dt=dt,
            rocket_loc=rocket_location,
            rocket_quat=rocket_quat,
            rocket_omega=rocket_omega,
            side_effect=side_effect
        )
        # Optional logging of commanded rate
        # if side_effect and 8 < time < 20:
        #     print(f"[{time:.2f}s] ω_cmd = ({w_cmd[0]:.3f}, {w_cmd[1]:.3f}, {w_cmd[2]:.3f}) rad/s")

        # 2) Map commanded body-rate to raw gimbal angles (no extra dt)
        factor = self.p_y_inertia / (self.thrust * self.vehicle_height)
        theta_x_raw = factor * w_cmd[0]
        theta_y_raw = factor * w_cmd[1]

        # 3) Clamp raw angles to physical gimbal limits
        theta_x_raw = np.clip(theta_x_raw, -self.max_angle, self.max_angle)
        theta_y_raw = np.clip(theta_y_raw, -self.max_angle, self.max_angle)

        # 4) First-order low-pass on gimbal angle
        alpha = dt / (self.gimbal_tau + dt)
        theta_x_smooth = (1 - alpha) * self.theta_x_prev + alpha * theta_x_raw
        theta_y_smooth = (1 - alpha) * self.theta_y_prev + alpha * theta_y_raw

        # 5) Clamp gimbal rate (dtheta) to max_rate
        max_dtheta = self.max_rate * dt
        self.d_theta_x = np.clip(theta_x_smooth - self.theta_x_prev, -max_dtheta, max_dtheta)
        self.d_theta_y = np.clip(theta_y_smooth - self.theta_y_prev, -max_dtheta, max_dtheta)

        # 6) Update cumulative gimbal angles
        self.theta_x_prev += self.d_theta_x
        self.theta_y_prev += self.d_theta_y

        # Optional debug log for gimbal state
        # if side_effect and 8 < time < 20:
        #     print(f"  θ_raw=( {np.rad2deg(theta_x_raw):.2f}°, {np.rad2deg(theta_y_raw):.2f}°)  "
        #           f"θ_prev=( {np.rad2deg(self.theta_x_prev):.2f}°, {np.rad2deg(self.theta_y_prev):.2f}°)  "
        #           f"dθ=( {np.rad2deg(self.d_theta_x):.2f}°, {np.rad2deg(self.d_theta_y):.2f}°)")

        # 7) Build incremental rotation step and apply
        r_step_x = R.from_rotvec([-self.d_theta_x, 0.0, 0.0])
        r_step_y = R.from_rotvec([0.0, self.d_theta_y, 0.0])
        self.gimbal_orientation = (r_step_y * r_step_x) * self.gimbal_orientation

        return self.gimbal_orientation

    def _calculate_theta(self, dt: float, rocket_location: np.ndarray, rocket_quat: np.ndarray, side_effect = False):
        """Calculates the rotation required to meet the attitude and translational requirements"""
        # Angular velocity determine by quaternion error and lqr calculated qdot

        if self.thrust == 0:
            return 0, 0, np.zeros(3)

        w = self.quaternion.getAngularVelocityCorrection(rocket_loc=rocket_location, rocket_quat=rocket_quat, side_effect=side_effect)

        # Compute theta using THETA = I/LT (w/dt)
        theta_x = (self.p_y_inertia / (self.thrust * self.vehicle_height)) * (w[0] / dt)
        theta_y = (self.p_y_inertia / (self.thrust * self.vehicle_height)) * (w[1] / dt)

        # Smooth data using first order filter
        alpha = dt / (self.gimbal_tau + dt)
        theta_x_s = (1 - alpha) * self.theta_x_prev + alpha * theta_x
        theta_y_s = (1 - alpha) * self.theta_y_prev + alpha * theta_y

        # Apply max gimbal rate (rad/s)
        dtheta_x = np.clip((theta_x_s - self.theta_x_prev), -self.max_rate * dt, self.max_rate * dt)
        dtheta_y = np.clip((theta_y_s - self.theta_y_prev), -self.max_rate * dt, self.max_rate * dt)

        # Update new theta
        theta_x = self.theta_x_prev + dtheta_x
        theta_y = self.theta_y_prev + dtheta_y

        # Save new theta values for next dt
        self.theta_x_prev = theta_x
        self.theta_y_prev = theta_y

        return theta_x, theta_y, w

    def update_variables_(self, thrust: float):
        """
        Function to update variables, called every time step, most values accessed passively
        :param thrust: current effective thrust [N]
        """
        self.thrust = thrust
        self.r_inertia = self.structural_model.roll_inertia
        self.p_y_inertia = self.structural_model.pitch_yaw_inertia


if __name__ == "__main__":
    from models.Structural.StructuralModel import StructuralModel
    from models.Engine.EngineModels import Engine
    q = np.eye(3) * 5
    r = np.eye(3) * 1

    tvc = TVCStructure(lqr=LQR(q=q, r=r), structure_model=StructuralModel(engine_class=Engine(),  liquid_total_ratio=0.56425))





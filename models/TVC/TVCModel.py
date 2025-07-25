import numpy as np

from ..TVC.LqrQuaternionModels import *
# from ..Structural.StructuralModel import StructuralModel    # DELETE ME


class TVCStructure:
    def __init__(self, lqr: object, structure_model: object):
        self.structural_model = structure_model
        self.quaternion = QuaternionFinder(lqr=lqr)

        self.thrust = None
        self.r_inertia = self.structural_model.roll_inertia
        self.p_y_inertia = self.structural_model.pitch_yaw_inertia

        self.cg = self.structural_model.cm_current
        self.vehicle_height = self.structural_model.length

        self.theta_x_prev = 0.0         # Previous gimbal angle around x-axis
        self.theta_y_prev = 0.0         # Previous gimbal angle around y-axis
        self.gimbal_tau = 0.2           # Time constant to slow gimbal rotation
        self.max_rate = np.deg2rad(10)  # Maximum rate of angular change

    def calculate_theta(self, dt: float, rocket_location: np.ndarray, rocket_quat: np.ndarray, side_effect = False):
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
        self.cg = self.structural_model.cm_current


if __name__ == "__main__":
    from models.Structural.StructuralModel import StructuralModel
    from models.Engine.EngineModels import Engine
    q = np.eye(3) * 5
    r = np.eye(3) * 1

    tvc = TVCStructure(lqr=LQR(q=q, r=r), structure_model=StructuralModel(engine_class=Engine(),  liquid_total_ratio=0.56425))





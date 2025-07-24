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
        print(self.vehicle_height)


    def calculate_theta(self, dt: float, rocket_location: np.ndarray, rocket_quat: np.ndarray, side_effect = False):
        """Calculates the rotation required to meet the attitude and translational requirements"""
        # Angular velocity determine by quaternion error and lqr calculated qdot

        if self.thrust == 0:
            return 0, 0, np.zeros(3)

        w = self.quaternion.getAngularVelocityCorrection(rocket_loc=rocket_location, rocket_quat=rocket_quat, side_effect=side_effect)
        # Rotation about the x-axis
        theta_1 = self.p_y_inertia / (self.thrust * self.vehicle_height)
        theta_x = theta_1 * w[0] / dt

        # Rotation about the y-axis
        theta_y = theta_1 * w[1] / dt

        # theta_y = round(theta_y, 2)
        # theta_x = round(theta_x, 2)

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





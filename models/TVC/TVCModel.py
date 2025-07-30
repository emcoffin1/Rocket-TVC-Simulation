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
                 structure_model: object,
                 aero_model: object,):
        # -- OBJECTS -- #
        self.struct = structure_model
        self.aero   = aero_model

        # -- CONSTANTS -- #
        self.height = self.struct.length
        self.p_y_inertia = self.struct.I[0]

        # -- STATES -- #
        self.theta = np.array([0.0, 0.0])
        self.theta_prev = np.array([0.0, 0.0])
        self.d_theta = np.array([0.0, 0.0])
        self.thrust = 0.0

        # -- GIMBAL CONSTANTS -- #
        self.gimbal_tau = 0.2
        self.max_rate   = np.deg2rad(100)
        self.max_angle  = np.deg2rad(7.5)

        # -- LOG -- #
        self.gimbal_log = []

    def compute_gimbal_orientation_using_torque(self, torque_body_cmd: np.ndarray, dt: float, time:float,
                                                side_effect: bool = False):
        """
        Computes gimbal angle using torque commands computed by the LQR
        theta = Torque (of opposite axis) / (Thrust * HEIGHT-CG)
        derived from Torque = F*r, where r = L*sin(theta)

        Computes in an array for simpler code
        :return: array to be multiplied by thrust magnitude [x y z]
        """
        lever = self.height - self.struct.cm_current
        self.theta_prev = self.theta

        # ==================================== #
        # -- COMPUTE REQUIRED GIMBAL ANGLES -- #
        # ==================================== #

        if self.thrust > 1e-6:
            # Using theta = Torque / (Length * Thrust)
            self.theta = np.array([
                                   torque_body_cmd[1] / (self.thrust * lever),
                                   torque_body_cmd[0] / (self.thrust * lever)
                                   ])

            # Clip to ensure not out of arc-sin limits
            self.theta = np.clip(self.theta, -1.0, 1.0)

            # Apply arc-sin adjustment
            self.theta = np.arcsin(self.theta)

        else:
            self.theta = np.array([0.0, 0.0])

        # -- CLIP TO MAX GIMBAL ANGLE -- #
        # Clip gimbal to max angles
        self.theta = np.clip(self.theta, -self.max_angle, self.max_angle)

        # -- FIRST-ORDER FILTER -- #
        # Time constant
        alpha = dt / (self.gimbal_tau + dt)
        self.theta = (1 - alpha) * self.theta_prev + alpha * self.theta

        # Rate Limit
        d_theta = self.theta - self.theta_prev

        d_theta = np.clip(d_theta, -self.max_rate*dt, self.max_rate*dt)

        # ==================== #
        # -- UPDATED GIMBAL -- #
        # ==================== #

        # Save values
        self.d_theta = d_theta / dt
        self.theta_prev += d_theta
        self.theta = self.theta_prev

        # -- UPDATE GIMBAL ROTATION -- #
        rot_x = R.from_rotvec(self.theta[0] * np.array([0, 1, 0]))
        rot_y = R.from_rotvec(self.theta[1] * np.array([1, 0, 0]))
        combined_rot = rot_y * rot_x

        base_thrust = np.array([0, 0, 1])
        rotated_dir = combined_rot.apply(base_thrust)

        if side_effect:
            self.gimbal_log.append([np.rad2deg(self.theta[0]), np.rad2deg(self.theta[1])])

        return rotated_dir

    def update_variables_(self, thrust: float):
        """
        Function to update variables, called every time step, most values accessed passively
        :param thrust: current effective thrust [N]
        """
        self.thrust = thrust
        self.p_y_inertia = self.struct.pitch_yaw_inertia




class FinTab:
    def __init__(self, rocket_position: np.ndarray, name, positive_torque):
        """Controls roll authority through the x-axis using fin tabs"""

        # Location of the fin wrt the center of gravity
        self.location = rocket_position
        # Depicts the direction of the forces acting on the component using cross product and normalization
        force_direction = np.linalg.cross(np.array([0, 0, 1]), rocket_position)
        force_direction /= np.linalg.norm(force_direction)

        self.force_direction = np.abs(force_direction)
        self.positive_torque = positive_torque

        # self.force_direction = -1 * force_direction

        self.name = name

        self.radial_distance = rocket_position[0] if rocket_position[0] != 0 else rocket_position[1]

        self.tab_theta  = 0
        self.dtheta = 0
        self.previous_theta = 0
        self.area       = 0.05
        self.max_tab_angle = np.deg2rad(15)
        self.max_dtheta = np.deg2rad(750)     # Maximum dtheta per second
        self.motor_tau = 0.5



class RollControl:
    def __init__(self, fins: list, struct: object):
        """
        Handles all fins
        :param: List of fins [px, nx, py, ny]
        """
        self.fins = fins
        self.struct = struct
        self.angles = []

    def update_fin_angles(self, tab: FinTab, angle: float):
        """
        Updates tab angle
        [rads]
        :param tab: Tab object
        :param angle: Angle [rad]
        """
        tab.tab_theta = angle
        # print( np.round(np.degrees(angle),2))


    def calculate_theta(self, torque_cmd: np.ndarray, rho: float, vel: np.ndarray, dt: float, time: float,
                        side_effect=False):
        """
        Function to determine the tab angles to attain a specific roll rate
        :param torque_cmd: Torque command determined from LQR [Tx Ty Tz]
        :param rho: Density at current altitude [kg/m3]
        :param vel: Velocity of air in body frame [x y z]
        :param dt: Time step [s]
        :return: x_theta, -x_theta, y_theta, -y_theta
        """
        r_list = []
        vel_mag = np.linalg.norm(vel)

        for i, x in enumerate(self.fins):
            if vel_mag > 1e-6 and rho > 1e-6:
                # -- FIND THETA REQUIRED -- #
                # a negative deflection results in a positive torque

                theta_target_raw = torque_cmd[2] / (4 * rho * vel[2] ** 2 * x.area * np.pi * (np.abs(x.radial_distance)) * x.positive_torque)

                # Clip to maximum angle
                theta_target_raw_clipped = np.clip(theta_target_raw, -x.max_tab_angle, x.max_tab_angle)

                # Apply first order filter
                alpha = dt / (x.motor_tau + dt)
                theta_filter = (1 - alpha) * x.tab_theta + alpha * theta_target_raw_clipped

                # Ensure dtheta !> max dtheta
                dtheta = (theta_filter - x.tab_theta) / dt
                dtheta = np.clip(dtheta, -x.max_dtheta * dt, x.max_dtheta * dt)

                theta = x.tab_theta + dtheta

                x.previous_theta = x.tab_theta
                x.dtheta = dtheta
                self.update_fin_angles(x, theta)

                r_list.append(x.tab_theta)

            else:
                r_list.append(x.tab_theta)

        self.angles.append(np.degrees(r_list))

        return r_list[0], r_list[1], r_list[2], r_list[3]











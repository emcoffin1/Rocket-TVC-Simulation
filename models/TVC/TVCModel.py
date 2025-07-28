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
        self.struct = structure_model
        self.aero   = aero_model
        self.height = self.struct.length

        self.p_y_inertia = self.struct.I[0]

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

    def compute_gimbal_orientation_using_torque(self, torque_body_cmd: np.ndarray, dt: float, time:float,
                                                side_effect: bool = False):
        """
        Computes gimbal angle using torque commands computed by the LQR
        theta = Torque (of opposite axis) / (Thrust * HEIGHT-CG)
        :return:
        """
        lever = self.height - self.struct.cm_current

        # ==================================== #
        # -- COMPUTE REQUIRED GIMBAL ANGLES -- #
        # ==================================== #
        if self.thrust > 1e-6:
            theta_x_raw = torque_body_cmd[1] / (self.thrust * lever)
            theta_y_raw = -torque_body_cmd[0] / (self.thrust * lever)
        else:
            theta_x_raw = 0
            theta_y_raw = 0

        # Clip gimbal to max angles
        theta_x_raw_clip = np.clip(theta_x_raw, -self.max_angle, self.max_angle)
        theta_y_raw_clip = np.clip(theta_y_raw, -self.max_angle, self.max_angle)

        # First-order filter
        alpha = dt / (self.gimbal_tau + dt)
        tx_filter = (1 - alpha) * self.theta_x + alpha * theta_x_raw_clip
        ty_filter = (1 - alpha) * self.theta_y + alpha * theta_y_raw_clip

        # Rate Limit
        dtx = tx_filter - self.theta_x
        dty = ty_filter - self.theta_y

        dtx = np.clip(dtx, -self.max_rate * dt, self.max_rate * dt)
        dty = np.clip(dty, -self.max_rate * dt, self.max_rate * dt)

        # ==================== #
        # -- UPDATED GIMBAL -- #
        # ==================== #

        # Save values
        self.theta_x += dtx
        self.theta_y += dty
        self.d_theta_x = dtx / dt
        self.d_theta_y = dty / dt

        # Update gimbal
        r_step_x = R.from_rotvec([0, dtx, 0])
        r_step_y = R.from_rotvec([dty, 0, 0])


        # self.gimbal_orientation = (r_step_y * r_step_x) * self.gimbal_orientation

        self.gimbal_orientation = (r_step_x * r_step_y) * self.gimbal_orientation

        if side_effect:
            self.gimbal_log.append([np.rad2deg(self.theta_x), np.rad2deg(self.theta_y)])
            print(f"t: {round(time,2)} | GIMBAL X: {np.round(np.rad2deg(self.theta_x),3)} || Y: {np.round(np.rad2deg(self.theta_y),3)}")
        return self.gimbal_orientation



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

        print(self.name, self.location, self.force_direction)


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
            if vel_mag > 1e-6:

                # -- FIND THETA REQUIRED -- #
                # a negative deflection results in a positive torque
                theta_target_raw = torque_cmd[2] / (4 * rho * vel[2]**2 * x.area * np.pi * np.abs(x.radial_distance) * x.positive_torque)
                # theta_target_raw = np.arcsin(theta_target_raw)

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











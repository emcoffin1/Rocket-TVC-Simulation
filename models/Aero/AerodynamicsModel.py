"""
Will be used to provide a high fidelity aerodynamics model for a TVC simulation
Drag forces will be computed using lookup tables that use angle of attack and mach computed in Ansys Fluent
These values, if applicable, will be fitted to a spline for easy Cd determination

Empirical model for a rocket drag
https://kth.diva-portal.org/smash/get/diva2:1881325/FULLTEXT01.pdf

Free flight of Orion Crew Vehicle
https://ntrs.nasa.gov/api/citations/20220007178/downloads/Free_flight_of_MPCV.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
class Aerodynamics:
    def __init__(self, air: object):
        # -- Initialize other models -- #
        self.air = air

    def _get_drag_coeff(self, mach_v: float) -> float:
        """
        Returns drag coefficient based on mach and aoa wrt body frame
        :param mach_v: Mach value in rocket frame
        :return: Drag coefficient
        """
        # Get mach value
        x = mach_v
        cd = None
        if x <= 0.9:
            cd = 0.425 + -0.149*x + 0.129*x**2
        elif x <= 1.1:
            cd = 0.86*x + -0.378
        else:
            cd = -0.149*x + 0.733
        return cd

    def getDragForce(self, vel: np.ndarray, aoa: float = None) -> np.ndarray:
        """
        Gets drag force as a function of free stream velocity, cross-sectional area, density, and angle of attack
        D = 0.5 * rho * V^2 * A * Cd
        :param vel: Velocity in body frame [x y z] [m/s]
        :param aoa: Angle of attack of freestream velocity on body [rad]
        :return: Drag force [x y z] [N]
        """
        v_mag = np.linalg.norm(vel)
        if v_mag == 0:
            return np.zeros(3)

        mach = self.air.getMachNumber(velocity_mps=v_mag)
        drag_direction = -vel / v_mag
        cd = self._get_drag_coeff(mach_v=mach)
        drag = 0.5 * self.air.rho * vel ** 2 * cd * 0.0545
        return drag * drag_direction


    def getLiftForce(self, vel_ms, roll_tabs: object):
        """
        Determines the lift force in all 3 axis, that is to say that the
        roll authority will be handled within the return array

        A negative deflection angle will result in a positive torque

        Uses a thin plate approximation of lift where Cl = 2*pi*theta

        :param vel_ms: Velocity of wind in body frame m/s [x y z]
        :param roll_tabs: Control tab angle [rad]
        :returns: Lift force [x y z] [N]
        """
        v_mag = np.linalg.norm(vel_ms)
        lift = []
        for x in roll_tabs.fins:
            if v_mag <= 0:
                lift.append(np.zeros(3))
            else:
                cl = 2 * np.pi * x.tab_theta
                # Scalar
                lift_comp = 0.5 * self.air.rho * v_mag**2 * x.area * cl
                # Multiplied to force direction identity
                lift_value = lift_comp * x.force_direction
                lift.append(np.array(lift_value))
        return lift

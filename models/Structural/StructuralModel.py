"""
The structural model will represent the physical vehicle and the properties that may change during flight
such as CG, total mass, and all downstream variables. As the vehicle is not fully designed, all of these
will be approximations, and are subject to change or re-design.

Structural simulation
http://eqboxsolution.com/design-of-rockets-and-space-launch-vehicles-edberg-pdf-a829/

Structural Dynamics of Space Vehicles Simulations
https://ntrs.nasa.gov/api/citations/19650022554/downloads/19650022554.pdf

"""
import numpy as np

class StructuralModel:
    def __init__(self, engine_class: object, liquid_total_ratio: float):
        # -- Objects -- #
        self.engine = engine_class

        # -- Structural Measurements -- #
        # The dimensions of the fins are not included in the structural models due to
        # its low impact, it is however considered in the aerodynamics model
        self.length         = 5.4864                                # [m] -- 18 feet --- Total length of the vehicle
        self.diameter       = 0.254                                 # [m] -- 10 in   --- Diameter of vehicle
        self.cm_offset_roll = 0.0                                   # [m] Center of mass offset on roll axis

        # -- MASS CONSTANTS [kg] -- #
        liquid_total_mass_ratio = liquid_total_ratio                # ratio of liquid mass to total mass
        self.fluid_mass     = self.engine.getFluidMass()            # Total fluid mass
        self.wetMass        = self.fluid_mass / liquid_total_mass_ratio    # Total mass before launch
        self.dryMass        = self.wetMass - self.fluid_mass               # Mass at end of burn, void of liquid mass
        self.mass_current   = self.wetMass                          # [kg] Adaptive flight mass
        self.dm             = 0                                     # [kg] Change in mass per time step

        # ======================== #
        # -- CENTER OF MASS [m] -- #
        # ======================== #

        self.cm_initial     = 3.42392
        self.cm_final       = 3.5687
        self.cm_current     = self.cm_initial                       # [m] Adaptive center of mass

        # ============================ #
        # -- CENTER OF PRESSURE [m] -- #
        # ============================ #

        self.cp_initial     = 3.59
        self.cp_final       = 3.59
        self.cp_current     = 3.59

        # ================================= #
        # -- MOMENTS OF INERTIA [kg*m^2] -- #
        # ================================= #

        self.roll_inertia   = None                                  # [kg*m^2] Roll inertia (does not change)
        self.pitch_yaw_inertia = None                               # [kg8m^2] Pitch and yaw inertia (does change)
        self.I = np.array([
            self.pitch_yaw_inertia,
            self.pitch_yaw_inertia,
            self.roll_inertia
        ])

        # ================ #
        # -- FLEX MODES -- #
        # ================ #
        # not implemented

        self.flex_modes = [
            FlexModes(mode_num=1, frequency=8.0, damping=0.02, modal_mass=0.3 * self.mass_current),
            FlexModes(mode_num=2, frequency=24.0, damping=0.03, modal_mass=0.15 * self.mass_current)
        ]
        self._update_Inertia()

    def updateProperties(self):
        """
        Updates all structurally related properties and variables
        !UPDATE ONLY ONCE PER TIMESTEP!
        """
        self._update_Total_Mass()
        self._update_CM()
        self._update_Inertia()

    def _update_Total_Mass(self):
        """Method to get the mass change over previous time step"""
        # Mass update
        if self.engine.combustion_chamber.active:
            # Determine total mass of the vehicle from tank updates
            val = self.dryMass + self.engine.getFluidMass()

            # Subtract most recent saved mass from new mass to get dm
            self.dm = self.mass_current - val

            # Set new mass
            self.mass_current = val

    def _update_CM(self):
        """
        Method to get the new center of mass using linearity
        Assumes linear shift in CM
        """
        if self.engine.combustion_chamber.active:
            self.cm_current = (self.cm_initial + ((self.mass_current - self.wetMass) / (self.dryMass - self.wetMass)) *
                               (self.cm_final - self.cm_initial))

    def _update_Inertia(self):
        """
        Method to update the pitch, roll, and yaw mass moments of inertia
        Assumes the center of mass only changes axially through flight, not radially
        """
        # Roll inertia
        # m/2 * r^2
        roll_j = 0.5 * self.mass_current * (self.diameter / 2) ** 2
        self.roll_inertia = roll_j + (self.mass_current * self.cm_offset_roll**2)

        # Pitch/yaw inertia
        # Assumes pitch and yaw inertia are equal
        # Assumes thin-walled shell
        # m/12 * L^2
        self.pitch_yaw_inertia = (1/12) * self.mass_current * self.length**2

        self.I = np.array([
            self.pitch_yaw_inertia,
            self.pitch_yaw_inertia,
            self.roll_inertia
        ])

    def get_structural_moment(self):
        total = 0

        for mode in self.flex_modes:
            total += mode.get_moment()
        return total

class FlexModes:
    def __init__(self, mode_num: int, frequency: float, damping: float, modal_mass: float):
        self.mode_num = mode_num
        self.omega = 2 * np.pi * frequency      # Natural frequency (rad/s)
        self.damping = damping                  # Damping ratio
        self.mass = modal_mass                  # Effective modal mass
        self.x = 0                              # Displacement
        self.v = 0                              # Velocity

    def update(self, force_input, dt):
        # Solves m*x'' + c*x' + k*x = F using RK2
        # Initial values
        x0 = self.x
        v0 = self.v

        # Constants
        # k = m*w^2
        k = self.mass * self.omega ** 2
        # c = 2*m*w*damping
        c = 2 * self.mass * self.omega * self.damping

        # xdot = v
        # vdot = 1/m (F - cv - kx)

        # Step 1: first iteration
        k1 = (force_input - c * v0 - k * x0) / self.mass

        # Midpoint estimates using xdot and vdot updates
        x_half = x0 + v0 * (dt / 2)
        v_half = v0 + k1 * (dt / 2)

        # Step 2: second iteration using midpoint estimates
        k2 = (force_input - c * v_half - k * x_half) / self.mass

        # Step 3: New state
        self.v += k2 * dt
        self.x += v_half * dt

    def get_moment(self, force_input: float = None):

        k = self.mass * self.omega ** 2
        if force_input:
            # M = mal
            c = 2 * self.mass * self.omega * self.damping
            a = (force_input - c * self.v - k * self.x) / self.mass

            return self.mass * a * self._get_effective_length()

        else:
            # M = kx = mw^2 x
            return k * self.x
    def _get_effective_length(self):
        if self.mode_num == 1:
            return 5.0
        elif self.mode_num == 2:
            return 3.0
        return 5.0




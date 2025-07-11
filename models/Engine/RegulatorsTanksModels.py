

class DomeReg:
    """
    Simulates the pressure regulation between the ullage and propellant tanks
    """
    def __init__(self, outlet_pressure: float):
        self.outletPressure = outlet_pressure


class PropTank:
    """
    Model to contain the lox or fuel
    """
    def __init__(self, volume: float, fluid: object):
        self.volume = volume
        self.fluid = fluid
        self.mass = volume * self.fluid.density_liquid
        self.pressure = 0
        self.gas_volume = 0
        self.gas_mass = 0
        self.gas_temp = 0
        self.gas_pressure = 0
        print(f" Loaded: {self.fluid.name()} mass: {self.mass} kg")
        print(f" Loaded: {self.fluid.name()} volume: {self.volume} m2")
        self.step = 0

        self.log_P_U = []
        self.log_V_U = []
        self.log_T_U = []
        self.log_m_U = []

        self.log_P_L = []
        self.log_V_L = []
        self.log_T_L = []
        self.log_m_L = []

    def volumeChange(self, dm: float, ullage: object, dome_reg: object):
        """
        Models gas entering the tank as liquid exits.
        :param dm: change in mass from liquid flow [kg]
        :param ullage: the ullage tank object
        :param dome_reg: the dome reg object
        :return:
        """


        # Volume lost due to fluid outflow
        dv = dm / self.fluid.density_liquid
        self.gas_volume += dv
        self.volume -= dv
        self.mass -= dm

        # Calculate incoming gas temp (isentropic expansion)
        P_reg = dome_reg.outletPressure

        T_in = ullage.T * (P_reg / ullage.P) ** ((ullage.gamma - 1) / ullage.gamma)

        # New total gas mass needed to hold pressure at new volume
        m_total = (P_reg * self.gas_volume) / (ullage.R * T_in)
        m_new = m_total - self.gas_mass

        # Update tank gas temp with energy balance BEFORE updating gas_mass
        if m_new > 0:
            self.gas_temp = (
                                    self.gas_mass * self.gas_temp + m_new * T_in
                            ) / (self.gas_mass + m_new)

        # Pull gas from ullage
        ullage.gasLeaving(m_new)

        # Update gas mass
        self.gas_mass = m_total

        # # Assume incoming gas undergoes isentropic expansion through regulator
        # P_reg = dome_reg.outletPressure
        # T_in = ullage.T * (P_reg / ullage.P) ** ((ullage.gamma - 1) / ullage.gamma)
        #
        # # Gas needed to keep pressure constant in new total volume
        # m_total = (P_reg * self.gas_volume) / (ullage.R * T_in)
        # m_new = m_total - self.gas_mass
        #
        # # Pull gas from ullage
        # ullage.gasLeaving(m_new)
        #
        # self.gas_mass = m_total
        # self.gas_temp = T_in

        self.step += 1

        self.log_P_U.append(self.gas_pressure)
        self.log_V_U.append(self.gas_volume)
        self.log_T_U.append(self.gas_temp)
        self.log_m_U.append(self.gas_mass)

        self.log_P_L.append(self.pressure)
        self.log_V_L.append(self.volume)
        self.log_T_L.append(0.0)
        self.log_m_L.append(self.mass)


    @property
    def getVolumeFilled(self) -> float:
        """Return volume using m/rho"""
        return self.mass / self.fluid.density_liquid


class InjectorPlate:
    def __init__(self, injector_holes: int = 1, injection_area: float = 0.01, mdot_fuel: float = 0.0, mdot_lox: float = 0.0):
        self.injector_holes = injector_holes
        self.injection_area = injection_area    # m2
        self.mdot_fuel = mdot_fuel              # kg/s
        self.mdot_lox = mdot_lox                # kg/s

    @property
    def injectMass(self, dt: float = 0.1):
        return (self.mdot_lox + self.mdot_lox) * dt


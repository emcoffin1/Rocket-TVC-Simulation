

class DomeReg:
    """
    Simulates the pressure regulation between the ullage and propellant tanks
    """
    def __init__(self, outlet_pressure: float):
        self.outletPressure = outlet_pressure


class PropTank:
    def __init__(self, volume: float, fluid: object, dome_reg: object, ullage: object):
        """
        Handles all fluid flow related tasks including lox or fuel and ullage gas
        :param volume: volume of tank       [m3]
        :param fluid: type of fluid         [object]
        :param dome_reg: regulating body    [object]
        :param ullage: pressure feed system [object]
        """
        # Fluid
        self.volume = volume
        self.fluid = fluid
        self.mass = volume * self.fluid.density_liquid

        # Regulator/Pressurant
        self.dome_reg = dome_reg
        self.ullage = ullage
        self.pressure = self.dome_reg.outletPressure
        self.reg_pressure = self.pressure
        self.gas_volume = 0     # No gas in tank
        self.gas_mass = 1e-8    # small amount of mass to not divide by zero
        self.gas_temp = self.ullage.T  # Set for first iteration
        self.gas_pressure = 0   # initially starts with zero gas pressures as there is no gas

        # State
        self.blow_down = False

        # Logging
        self.log_P_U = []
        self.log_V_U = []
        self.log_T_U = []
        self.log_m_U = []

        self.log_P_L = []
        self.log_V_L = []
        self.log_T_L = []
        self.log_m_L = []

        print(f" Loaded: {self.fluid.name()} mass: {self.mass} kg")
        print(f" Loaded: {self.fluid.name()} volume: {self.volume} m2")

    def volumeChange(self, dm: float, dt: float):
        """
        Models gas entering the tank as liquid exits or the behavior of gas during blowdown
        :param dm: change in mass from liquid flow  [kg]
        :param dt: time step                        [s]
        :return:
        """
        # Volume lost due to fluid outflow
        dv = dm / self.fluid.density_liquid

        # Update fluid mass and volume, no effect directly on gas
        self.volume -= dv
        self.mass -= dm

        if self.blow_down:
            # System in blow-down
            # Constant mass, changing volume, new temp and pressure
            # P = P (V1 / V2) ^ y (GAS)
            new_P = self.gas_pressure * (self.gas_volume / (self.gas_volume + dv)) ** self.ullage.gamma
            # T = T (V1 / V2) ^ y-1 (GAS)
            new_T = self.gas_temp * (self.gas_volume / (self.gas_volume + dv)) ** (self.ullage.gamma - 1)

            # Store all new gas variables
            self.gas_pressure = new_P
            self.gas_temp = new_T
            self.gas_volume += dv


        else:
            # System is not in blow-down
            # Calculate the mass required to offset the volume lost
            # m = PV/RT (GAS)
            m_new = self.reg_pressure * dv / (self.ullage.R * self.gas_temp)

            # Calculate the new temperature of the gas
            # Change in energy formula T1 + ((y * T0 - T1) / m1) * m_new
            T_new = self.gas_temp + ((self.ullage.gamma * self.ullage.T - self.gas_temp) / self.gas_mass) * m_new

            # Store all new gas variables
            self.gas_temp = T_new
            self.gas_mass += m_new
            self.gas_volume += dv
            self.gas_pressure = self.reg_pressure

            # Update ullage tank
            self.ullage.gasLeaving(dm=m_new, dt=dt)

            # Check if blow down now
            if self.ullage.P <= self.reg_pressure:
                print("[STATE CHANGE] Entering blow-down")
                self.blow_down = True

        self.log_P_U.append(self.gas_pressure)
        self.log_V_U.append(self.gas_volume)
        self.log_T_U.append(self.gas_temp)
        self.log_m_U.append(self.gas_mass)

        self.log_P_L.append(self.pressure)
        self.log_V_L.append(self.volume)
        self.log_T_L.append(0)
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


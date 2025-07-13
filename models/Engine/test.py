class ProTank:
    def __init__(self, volume: float, fluid: object, dome_reg: object, ullage: object, tank_volume: float = None):
        """
        Handles all fluid flow related tasks including lox or fuel and ullage gas
        :param volume: volume of tank       [m3]
        :param fluid: type of fluid         [object]
        :param dome_reg: regulating body    [object]
        :param ullage: pressure feed system [object]
        """
        self.tank_volume = tank_volume if tank_volume is not None else round(volume+0.004, 2)

        # -- Fluid -- #
        self.volume = volume
        self.fluid = fluid
        self.temp = self.fluid.storing_temp
        self.mass = volume * self.fluid.density_liquid

        # -- Regulator/Pressurant -- #
        self.dome_reg = dome_reg
        self.ullage = ullage
        # self.pressure = self.dome_reg.outletPressure
        self.reg_pressure = self.dome_reg.outletPressure

        # -- Total Gas -- #
        self.total_gas_pressure = 0
        self.total_gas_volume = 0

        # -- Nitrogen Ullage Gas -- #
        self.ull_volume = 0     # No gas in tank
        self.ull_mass = 1.0e-6    # small amount of mass to not divide by zero
        self.ull_temp = self.ullage.T  # Set for first iteration
        self.ull_pressure = 0   # initially starts with zero gas pressures as there is no gas

        # -- Fluid as a gas -- #
        self.gas_fluid_mass = 0
        self.gas_fluid_volume = 0
        self.gas_fluid_temp = self.fluid.storing_temp
        self.gas_fluid_pres = 0

        # -- State -- #
        self.blow_down = False
        self.dm = 0  # The amount of change each cycle

        # -- Logging -- #
        self.log_P_U = []   # | Ullage gas logging
        self.log_T_U = []   # |
        self.log_m_U = []   # |

        self.log_P_L = []   # | Liquid logging
        self.log_V_L = []   # |
        self.log_T_L = []   # |
        self.log_m_L = []   # |

        self.log_P_Lg = []   # | Liquid as a gas logging
        self.log_T_Lg = []   # |
        self.log_m_Lg = []   # |

        self.log_V_total_gas = []
        self.log_P_total_gas = []
        self.log_M_total_gas = []

        # print(f" Loaded: {self.fluid.name()} mass: {self.mass} kg")
        # print(f" Loaded: {self.fluid.name()} volume: {self.volume} m2")

        self.heat_leak = 0.05  # W -- guestimate

        self._add_some_nitrogen()

    def volumeChange(self, dm: float, time: float, dt: float):
        """
        Since last time step: more fluid has left tank to fuel combustion, fluid has boiled off
        -------------------------------------------------------------------------------------------
        What needs to happen: calculate the volume of fluid lost to combustion and to boil off
                              recompute the temperature of gasses in the tank (ullage and boil-off)
                              from updated temperature, compute the new pressure of the gas
                              Determine if blow down or pressure regulated:
                              Blow down
                                Recompute temperature and pressure based on dV
                              Pressure regulated
                                Determine the mass required from the ullage tank to correct the pressure
                                Automatically add that mass and recompute all values
                                Save mass required from tank for external access

                              Log all updates
        -------------------------------------------------------------------------------------------
        Models gas entering the tank as liquid exits or the behavior of gas during blowdown
        U = m*cv*T`
        T = sum(U) / V_total
        :param dm: change in mass from liquid flow  [kg]
        :param time: Total flight time for logging  [s]
        :param dt: time step                        [s]
        :return:
        """


        dm = min(dm, self.mass)
        dv = dm / self.fluid.density_liquid

        self.volume -= dv
        self.mass -= dm

        self.total_gas_volume = self.tank_volume - self.volume


        print(self.volume, self.total_gas_volume)


import math


class DomeReg:
    """
    Simulates the pressure regulation between the ullage and propellant tanks
    Pressure drop is assumed to be zero due to comparatively high pressures
    """
    def __init__(self, outlet_pressure: float):
        self.outletPressure = outlet_pressure


class InjectorPlate:
    def __init__(self, injector_holes: int = 1, injection_area: float = 0.01, mdot_fuel: float = 0.0, mdot_lox: float = 0.0):
        self.injector_holes = injector_holes
        self.injection_area = injection_area    # m2
        self.mdot_fuel = mdot_fuel              # kg/s
        self.mdot_lox = mdot_lox                # kg/s

    @property
    def injectMass(self, dt: float = 0.1):
        return (self.mdot_lox + self.mdot_lox) * dt


class PropTank:
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
                                    -- Happens passively
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

        # Account for fluid boil off which happens passively
        # Should only happen to LOX but just in case
        if self.temp > self.fluid.boiling_point:
            # Adds mass to fluid gas and removes mass from fluid
            self._update_fluid_boil_off(dt=dt)

        # -- Fluid loss due to combustion -- #
        # Clamp mass change
        dm = min(dm, self.mass)

        # Volume lost due to fluid outflow
        dv = dm / self.fluid.density_liquid

        # Update fluid volume and mass
        self.volume -= dv
        self.mass -= dm

        # Update total gas volume
        self.total_gas_volume = self.tank_volume - self.volume

        # Update temps and pressures before determining mass needed
        self._update_partial_pressure()
        self._update_gas_temperatures()

        # -- Replenish Gas -- #
        # P,V,T,m have all updated, now pull ullage gas
        if not self.blow_down:
            # System not in blow-down
            self._pressure_feed_step()

            # Check if blow down now
            if self.ullage.P <= self.reg_pressure:
                print(f"[STATE CHANGE] Entering blow-down at: {time}s")
                self.blow_down = True

        else:
            self.dm = 0

        # -- Logging -- #
        self.log_P_U.append(self.ull_pressure)      # | Ullage gas logging
        self.log_T_U.append(self.ull_temp)          # |
        self.log_m_U.append(self.ull_mass)          # |

        self.log_P_L.append(0)                      # | Liquid logging
        self.log_V_L.append(self.volume)            # |
        self.log_T_L.append(self.temp)              # |
        self.log_m_L.append(self.mass)              # |

        self.log_P_Lg.append(self.gas_fluid_pres)       # | Liquid as a gas logging
        self.log_T_Lg.append(self.gas_fluid_temp)       # |
        self.log_m_Lg.append(self.gas_fluid_mass)       # |

        self.log_V_total_gas.append(self.total_gas_volume)      # | Total gas logging
        self.log_P_total_gas.append(self.total_gas_pressure)    # |

    def _update_gas_temperatures(self):
        """
        Updates temperature of gasses
        T = PV / mR
        """
        if self.ull_mass != 0:
            self.ull_temp = self.ull_pressure * self.total_gas_volume / (self.ullage.R * self.ull_mass)

        if self.gas_fluid_mass != 0:
            self.gas_fluid_temp = self.gas_fluid_pres * self.total_gas_volume / (self.fluid.R_vapor * self.gas_fluid_mass)

    def _update_fluid_boil_off(self, dt: float):
        """
        Uses dm = Q * dt / h to determine boil off mass and volume since last time step
        Only accounts for enclosure induced boil off, not boil off from hotter N gas
        """
        # Compute boiled off mass
        dm_bo = self.heat_leak * dt / self.fluid.vapor_energy

        # Clamp if not enough mass left
        dm_bo = min(dm_bo, self.mass)

        # Subtract from liquid mass
        self.mass -= dm_bo

        # Update fluid vapor mass
        self.gas_fluid_mass += dm_bo

        # Update volumes
        dv_bo = dm_bo / self.fluid.density_liquid
        self.volume -= dv_bo
        self.gas_fluid_volume += dv_bo

    def _pressure_feed_step(self):
        """
        Determines the mass of ullage required to return to set point
        Update partial pressures before
        """
        # Get pressure difference
        dP = self.reg_pressure - self.total_gas_pressure

        if dP > 0:
            # Pressure below set point, fill tanks
            # Prevent divide-by-zero
            denom = self.ullage.R * self.ull_temp
            if denom <= 0:
                raise RuntimeError(f"Bad ullage R/T: R={self.ullage.R}, T={self.ull_temp}")

            # Get m of pressure at ullage tank temp using m = PV/RT
            new_m = dP * self.total_gas_volume / denom

            # Add mass to ullage mass
            self.dm = new_m
            self.ull_mass = self.ull_mass + new_m

        else:
            self.dm = 0

    def _update_partial_pressure(self):
        """
        Uses ideal gas law to update the partial pressures and then the total pressure
        P = mRT / V
        """
        P_ullage = self.ull_mass * self.ullage.R * self.ull_temp / self.total_gas_volume
        # P_fluid_gas = self.gas_fluid_mass * self.fluid.R_vapor * self.gas_fluid_temp / self.total_gas_volume
        self.ull_pressure = P_ullage
        # self.gas_fluid_pres = P_fluid_gas
        self.total_gas_pressure = P_ullage #+ P_fluid_gas

    def _add_some_nitrogen(self):
        """Preloads tanks with boil off pressure or ullage pressure"""
        # Volume to fill
        # v_empty = self.tank_volume - self.volume

        # Place nitrogen with ullage temp
        self.ull_temp = self.ullage.T
        # self.gas_fluid_temp = self.fluid.boiling_point

        if self.temp > self.fluid.boiling_point:
            # -- BOIL OFF -- #
            # Give a small amount of boil off if temp is above
            m_bo = 0.000001 * self.mass
            # print(f"Boil off mass: {m_bo}")
            self.mass -= m_bo                               # Subtract boiled mass from liquid mass
            self.gas_fluid_mass = m_bo                      # Add boiled mass to gas mass

            # Determine volume of that lost mass
            v_lost_fluid = m_bo / self.fluid.density_liquid

            # Adjust fluid and fluid gas volumes
            self.volume = self.volume - v_lost_fluid        # Update the fluid volume level
            v_empty = self.tank_volume - self.volume        # Determines how much empty space is left for the gas
            self.total_gas_volume = v_empty                 # Declares the volume of gas in total

            # Determine the pressure using ideal
            p_bo = self.gas_fluid_mass * self.fluid.R_vapor * self.gas_fluid_temp / self.total_gas_volume
            self.total_gas_pressure = p_bo                  # Updates the total pressure of the gasses
            self.gas_fluid_pres = p_bo                      # Updates the PP of the liquid gas

            # -- ULLAGE GAS -- #
            # Pressure required
            # Clamped so can't be over pressured
            p_reg = max(0, self.reg_pressure - self.total_gas_pressure)
            self.total_gas_volume = self.tank_volume - self.volume

            # Get m needed from ullage tank using m = PV/RT
            self.ull_mass = p_reg * self.total_gas_volume / (self.ullage.R * self.ull_temp)

            # Final pressure using ideal
            p_ull = self.ull_mass * self.ullage.R * self.ull_temp / self.total_gas_volume

            self.total_gas_pressure += p_ull
            self.ull_pressure = p_ull

        else:
            # -- ULLAGE GAS -- #
            self.total_gas_volume = self.tank_volume - self.volume
            # Pressure required
            # Clamped so can't be over pressured
            p_reg = self.reg_pressure
            # Find m required to fill tank at required pressure using m = PV/RT
            self.ull_mass = p_reg * self.total_gas_volume / (self.ullage.R * self.ull_temp)

            # Final pressure using ideal
            p_ull = self.ull_mass * self.ullage.R * self.ull_temp / self.total_gas_volume
            self.total_gas_pressure += p_ull
            self.ull_pressure = p_ull

        # Remove mass from ullage tank
        self.ullage.gasLeaving(dm=self.ull_mass, dt=0.0001)

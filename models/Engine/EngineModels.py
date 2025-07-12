import numpy as np
import math
import matplotlib.pyplot as plt
from models.Engine.LiquidModels import LOX, RP1, BiProp, UllageGas
from models.Engine.RegulatorsTanksModels import DomeReg, PropTank
from models.EnvironmentalModels import AirProfile
import scipy.optimize


class Engine:
    def __init__(self, pc_desire: float = 551581, safety_margin: float = 0.2):
        self.Pc_desired = pc_desire
        self.safety_margin = safety_margin
        dome_reg = self.Pc_desired + (self.Pc_desired * self.safety_margin)

        mdot = 4.2134195

        lox_reg = DomeReg(outlet_pressure=dome_reg)
        fuel_reg = DomeReg(outlet_pressure=dome_reg)
        ullage = UllageGas(m0=1.6232, V0=0.05, mdot=mdot)
        self.combustion_chamber = CombustionChamber(
            biprop=BiProp(),
            fuel=RP1(),
            lox=LOX(),
            of_ratio=1.8,
            Pc=self.Pc_desired,
            mdot=mdot,
            eps=1.936,
            At=0.00825675768,  # m²
            ullage_tank=ullage,
            lox_reg=lox_reg,
            fuel_reg=fuel_reg,
            lox_tank=PropTank(volume=0.0474, fluid=LOX(), dome_reg=lox_reg, ullage=ullage),
            fuel_tank=PropTank(volume=0.0371, fluid=RP1(), dome_reg=fuel_reg, ullage=ullage)
        )

        self.nozzle = Nozzle(combustion_chamber=self.combustion_chamber)

    def runBurn(self, dt: float, alt_m: float = 0, side_effect: bool = True):
        """
        Processes the thrust time by firing combustion chamber for raw value
        and then passing to nozzle for pressure adjustment
        :param dt: time step [dt]
        :param alt_m: current altitude [m]
        :return:
        """
        if self.combustion_chamber.active:
            raw_thrust = self.combustion_chamber.burnStep(dt=dt, side_effect=side_effect)
            adjusted_thrust = self.nozzle.getThrust(thrust=raw_thrust, alt_m=alt_m)
            if side_effect and adjusted_thrust < 0:
                self.combustion_chamber.active = False
                return 0.0
            return adjusted_thrust
        else:
            return 0.0

    def getFluidMass(self) -> float:
        """
        Acquired total fluid and gas tank mass
        :return:
        """
        mass_lox = self.combustion_chamber.lox_tank.mass
        mass_fuel = self.combustion_chamber.fuel_tank.mass
        mass_ullage = self.combustion_chamber.ullage_tank.m_total
        # print(f"FUEL: {mass_fuel} - ULLAGE: {mass_ullage} - LOX: {mass_lox}")
        return mass_lox + mass_fuel + mass_ullage


class CombustionChamber:
    def __init__(self, biprop: BiProp, fuel: RP1, lox: LOX, of_ratio: float, Pc: float, mdot: float ,eps: float, At: float = 0.00825675768,
                 air: AirProfile = None, lox_tank: PropTank = None, fuel_tank: PropTank = None,
                 ullage_tank: UllageGas = None, lox_reg: DomeReg = None, fuel_reg: DomeReg = None):

        # Target mdot_total = 4.2134 kg/s
        # Target of ratio   = 1.8
        # Target mdot_fuel  = 1.4197195
        # Target mdot_lox   = 2.7207755
        # Target Pc_initial = 551581

        # Indicates if the engine is active, will be switched to false on loss of fluids or pressure
        self.active = True
        self.time = 0

        # Liquids
        self.lox = lox
        self.fuel = fuel
        self.biprop = biprop
        self.air = air if air is not None else AirProfile()

        # Tanks
        self.lox_tank = lox_tank
        self.fuel_tank = fuel_tank
        self.ullage_tank = ullage_tank

        # Regulators
        self.fuel_reg = fuel_reg if fuel_reg is not None else DomeReg(outlet_pressure=200e3)
        self.lox_reg = lox_reg if lox_reg is not None else DomeReg(outlet_pressure=200e3)

        # Combustion Chamber Properties
        self.mdot = mdot
        self.of_ratio = of_ratio
        self.Pc = Pc
        self.Pe = None
        self.Ve = 0
        self.eps = eps
        self.g = 9.08665       # m/s2
        self.At = At

        self._update_gas_properties()

    def _update_gas_properties(self):
        """Uses propellants CEA interface to update combustion properties"""
        props = self.biprop.gasProperties(self.Pc, self.of_ratio, self.eps)
        self.Tc = props["T_c"]
        self.gamma = props["gamma"]
        self.R = props["R_specific"] * 1000
        self.isp_vac = props["Isp_vac"]
        self.c_star = props["c_star"]
        self.updateExitPressure()
        self.updateExhaustVelocity()

    def getCurrentIsp(self, vacuum: bool = False) -> float:
        """
        Returns an isp based on vacuum
        :param vacuum: bool
        :return:
        """
        return self.isp_vac if vacuum else self.isp_sea

    def getMassFlowRate(self, pc: float = None) -> (float, float):
        """
        Uses chamber pressure, throat area, and characteristic velocity to compute mass flow rate
        mdot = Pc * At / c*
        mdot_f = mdot / (1 + o/f)
        mdot_l = mdot - mdot_f
        :param pc: chamber pressure passively updated
        :return:
        """
        Pc = pc if pc is not None else self.Pc
        mdot = Pc * self.At / self.c_star
        mdot_f = mdot / (1 + self.of_ratio)
        mdot_l = mdot - mdot_f
        # print(mdot_f, mdot_l)
        return mdot_f, mdot_l

    def updateExitPressure(self):
        """Updates exit pressure"""
        Me = self._solve_exit_mach()
        exp = (1 + (self.gamma - 1) / 2 * Me**2)
        self.Pe = self.Pc / (exp**(self.gamma / (self.gamma - 1)))
        # print(f"Exit Pressure: {self.Pe} -- Exit Mach: {Me}")

    def updateExhaustVelocity(self):
        """Exhaust velocity equation"""
        term1 = 2 * self.gamma * self.R * self.Tc / (self.gamma - 1)
        exp = (self.gamma - 1) / self.gamma
        term2 = (1 - ((self.Pe / self.Pc)**exp))
        self.Ve = math.sqrt(term1 * term2)
        # print(f"Exhaust Vel: {self.Ve}")

    def updateChamberPressure(self, feed_pressure: float = None):
        """
        Updates chamber pressure using mdot c* / At                                 -- more realistic
        Allows for mdot to vary after ullage pressure drop and blow-down begins
        This relies on the assumption that the chamber pressure is equal to the pressure that
        is held in the tanks
        """
        # print(feed_pressure)
        f, l = self.getMassFlowRate(feed_pressure)
        self.Pc = (f+l) * self.c_star / self.At

    def getThrust(self) -> float:
        """
        Gets raw engine thrust using: F = mdot * ve
        This is the raw thrust without nozzle affects
        :return: Thrust [N]
        """
        mdot_f, mdot_o = self.getMassFlowRate()
        mdot = mdot_o + mdot_f
        F = mdot * self.Ve
        return F

    def burnStep(self, dt: float, side_effect: bool = True) -> float:
        """
        An iteration of time step, changes the mass in each tank, returns the RAW thrust,
        will return no thrust if tanks are empty
        :param dt: time step [s]
        :return: Thrust [N]]
        """
        if not self.active:
            self.Pc = 0
            self.mdot = 0
            # print("Engine deactivated")
            return 0

        if not side_effect:
            # Return estimated thrust
            return self.getThrust()

        # STEP 1: Get fuel mass in tanks
        mf_mass = self.fuel_tank.mass
        mo_mass = self.lox_tank.mass


        # STEP 2: Check if tanks have liquid
        if mf_mass <= 0 or mo_mass <= 0:
            self.active = False
            print(f"[SHUTDOWN] Tanks empty: {mf_mass} --- {mo_mass}")
            print(f"[SHUTDOWN] Burn Stops: {self.time:.2f}s")
            return 0

        # STEP 3: Get current flow rates
        mdot_f, mdot_o = self.getMassFlowRate()

        # print(f"MDOT_F: {mdot_f}  -  MDOT_O: {mdot_o}  -  dt: {dt}")

        # STEP 4: Find liquid mass change in each tank
        mf_used = mdot_f * dt
        mo_used = mdot_o * dt

        # STEP 5: Update liquid tank pressure
        self.lox_tank.volumeChange(dm=mo_used, time=self.time)
        self.fuel_tank.volumeChange(dm=mf_used, time=self.time)

        # STEP 6: Update ullage tank
        dm_f = self.fuel_tank.dm
        dm_l = self.lox_tank.dm
        # print(f"Fuel: {dm_f}  --  LOX: {dm_l}")
        if not (self.lox_tank.blow_down or self.fuel_tank.blow_down):
            self.ullage_tank.gasLeaving(dm=(dm_l + dm_f), dt=dt)
        else:
            self.ullage_tank.gasLeaving(dm=0, dt=dt)

        # STEP 7: Update Chamber Pressure
        # Get the pressure that the fuel is being pushed out at
        feed_pressure = max(self.lox_tank.gas_pressure, self.fuel_tank.gas_pressure)
        # Update the chamber pressure using that
        self.updateChamberPressure(feed_pressure)

        # STEP 7: Update gas properties
        self._update_gas_properties()

        # Get thrust
        thrust = self.getThrust()
        if thrust < 0:
            self.active = False
            return 0
        # print(thrust)

        self.time += dt

        return thrust

    def _solve_exit_mach(self) -> float:
        """
        Solves for exit Mach number using isentropic area-Mach relation
        uses iterative method (brent)
        """
        gamma = self.gamma
        eps = self.eps

        def func(M):
            lhs = eps
            rhs = (1 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M ** 2)) ** ((gamma + 1) / (2 * (gamma - 1)))
            return lhs - rhs

        # Only supersonic exit flows are valid (e.g., M_e > 1)
        val = scipy.optimize.brentq(func, 1.01, 10.0)
        return val

    def _get_rawthrust_over_time(self, time_max=10, dt=1.0):
        """
        Performs a sweep over a set time to analyze the change in thrust
        :param time_max: Simulation duration
        :param dt: Time step in seconds
        :return: list: h, thrust
        """
        results = []
        time_val = 0.0

        while time_val <= (time_max / dt):
            thrust_vec = self.burnStep(dt=dt)

            if thrust_vec <= 0:
                break

            results.append([time_val, thrust_vec, self.Pc])

            time_val += dt

        return results

    def _get_tank_size(self, mdot_target: float, of_ratio_target: float, burntime:float) -> dict:

        # Fuel mass flow rate -- mdot_f = mdot / (1 + OF)
        mdot_f = mdot_target / (1 + of_ratio_target)
        # LOX mass flow rate -- mfot_l = mdot - mdot_f
        mdot_l = mdot_target - mdot_f

        rho_fuel = self.fuel.liquidDensity(self.fuel.storing_temp)
        rho_ox = self.lox.liquidDensity(self.lox.storing_temp)

        # Mass required -- mdot_f * burn time
        m_f = mdot_f * burntime
        m_o = mdot_l * burntime

        # Density equation -- V = m / rho
        v_f = m_f / rho_fuel
        v_o = m_o / rho_ox

        return {
            "fuel_mass": m_f,
            "fuel_volume": v_f,
            "lox_mass": m_o,
            "lox_volume": v_o
        }

    def _get_ullage_tank_pressure(self, ullage_volume: float, mdot_target: float, of_ratio_target: float,
                                  burntime:float, pressure_target: float) -> dict:

        tank_sizing = self._get_tank_size(mdot_target=mdot_target, of_ratio_target=of_ratio_target, burntime=burntime)

        m_f = tank_sizing["fuel_mass"]
        v_f = tank_sizing["fuel_volume"]
        m_o = tank_sizing["lox_mass"]
        v_o = tank_sizing["lox_volume"]

        # Using ideal gas law -- m = PV / RT
        m_ullage_f  = pressure_target * v_f / (self.ullage_tank.R * self.ullage_tank.T)
        m_ullage_o  = pressure_target * v_o / (self.ullage_tank.R * self.ullage_tank.T)
        m_ullage    = pressure_target * ullage_volume / (self.ullage_tank.R * self.ullage_tank.T)

        total_ullage_mass = m_ullage_o + m_ullage_f + m_ullage

        # Using ideal gas law -- P = mRT / V
        ullage_pressure = total_ullage_mass * self.ullage_tank.R * self.ullage_tank.T / ullage_volume

        return {
            "fuel_mass": m_f,
            "fuel_volume": v_f,
            "lox_mass": m_o,
            "lox_volume": v_o,
            "nitrogen_mass": total_ullage_mass,
            "nitrogen_pressure": ullage_pressure
        }

    def get_fluid_setup(self, ullage_volume: float, mdot_target: float, of_ratio_target: float, burntime:float,
                       pressure_target: float):
        """
        Prints all fluid setup information for tank pressurization, volume, and mass
        :param ullage_volume: Volume of ullage tank [m3]
        :param mdot_target: Target mass flow rate [kg/s]
        :param of_ratio_target: Ratio of oxygen and fuel
        :param burntime: Desired burn time [s]
        :param pressure_target: Desired chamber pressure [Pa]
        :return:
        """
        info = self._get_ullage_tank_pressure(ullage_volume=ullage_volume, mdot_target=mdot_target,
                                              of_ratio_target=of_ratio_target, burntime=burntime,
                                              pressure_target=pressure_target)

        print("=" * 15)
        print(f"RECOMMENDED -- FUEL MASS:         {info['fuel_mass']:.4f} kg")
        print(f"RECOMMENDED -- FUEL VOLUME:       {info['fuel_volume']:.4f} m3")
        print(f"RECOMMENDED -- LOX MASS:          {info['lox_mass']:.4f} kg")
        print(f"RECOMMENDED -- LOX VOLUME:        {info['lox_volume']:.4f} m3")
        print(f"RECOMMENDED -- NITROGEN MASS:     {info['nitrogen_mass']:.4f} kg")
        print(f"RECOMMENDED -- NITROGEN MASS(adj):{1.85*info['nitrogen_mass']:.4f} kg")
        print(f"RECOMMENDED -- NITROGEN PRESSURE: {info['nitrogen_pressure']:.4f} Pa")
        print("=" * 15)


class Nozzle:
    def __init__(self, air: AirProfile = None, combustion_chamber: CombustionChamber = None,
                 Ae: float = 0.0159851293):
        self.Ae = Ae
        self.air = air if air is not None else AirProfile()
        self.combustionChamber = combustion_chamber

    def getThrust(self, thrust: float, alt_m: float = 0) -> float:
        """
        Uses raw thrust (mdot Ve) + dP * Ae
        Calculates once at a time
        :param thrust: Raw thrust value [N]
        :param alt_m: Current altitude [m]
        :return:
        """
        if thrust != 0:
            pres_atm = self.air.getStaticPressure(alt_m=alt_m)
            pres_exit = self.combustionChamber.Pe
            thrust_total = thrust + ((pres_exit - pres_atm) * self.Ae)
            return thrust_total
        else:
            return 0




if __name__ == "__main__":

    engine = Engine()
    # engine.combustion_chamber.get_fluid_setup(burntime=20, ullage_volume=0.05, pressure_target=551581, of_ratio_target=1.8,
    #                                           mdot_target=4.21,)
    t = []
    T = []
    alt = []

    dt = 0.1
    time_val = 0.0
    # for x in range(round(3000 / dt)):
    #     if not engine.combustion_chamber.active:
    #         print(f"Engine shutdown at {time_val:.2f} seconds")
    #         break
    #
    #     y = engine.runBurn(dt=dt, alt_m=0)
    #     print(y)
    #     T.append(y)
    #     t.append(time_val)
    #     alt.append(x**2)
    #
    #     time_val += dt
    alts = 0
    while time_val < 20: # engine.combustion_chamber.active:
        y = engine.runBurn(dt=dt, alt_m=time_val **3)
        # print(y)
        T.append(y)
        t.append(time_val)
        time_val += dt


    print(f"Engine shutdown at {time_val:.2f} seconds")

    logs = [engine.combustion_chamber.ullage_tank.log_V,        # 0
            engine.combustion_chamber.ullage_tank.log_T,        # 1
            engine.combustion_chamber.ullage_tank.log_P,        # 2
            engine.combustion_chamber.ullage_tank.log_m,        # 3

            engine.combustion_chamber.fuel_tank.log_V_L,        # 4
            engine.combustion_chamber.fuel_tank.log_T_L,        # 5
            engine.combustion_chamber.fuel_tank.log_P_L,        # 6
            engine.combustion_chamber.fuel_tank.log_m_L,        # 7

            engine.combustion_chamber.lox_tank.log_V_U,         # 8
            engine.combustion_chamber.lox_tank.log_T_U,         # 9
            engine.combustion_chamber.lox_tank.log_P_U,         # 10
            engine.combustion_chamber.lox_tank.log_m_U,         # 11

            ]

    # for i,j,k,w in zip(logs[8], logs[9], logs[10], logs[11]):
    #     print(f"Volume: {i} -- Temp: {j}  --  Pres: {k}  --  Mass: {w}")
    # print(f"t len:{len(t)}   T len:{len(T)}")
    # for i,j in zip(t, T):
    #     print(f"t: {i}, T: {j}")

    # plt.subplot(3,2,1)
    # plt.plot(logs[0], label="Ullage Volume")
    # # plt.plot(logs[4], label="Liquid Volume")
    # plt.plot(logs[8], label="Gas Volume")
    # plt.legend()
    #
    # plt.subplot(3,2,2)
    # plt.plot(logs[1], label="Ullage Temp")
    # # plt.plot(logs[5], label="Liquid Temp")
    # plt.plot(logs[9], label="Gas Temp")
    # plt.legend()
    #
    # plt.subplot(3,2,3)
    # plt.plot(logs[2], label="Ullage Pressure")
    # # plt.plot(logs[6], label="Liquid Pressure")
    # plt.plot(logs[10], label="Gas Pressure")
    # plt.legend()
    #
    # plt.subplot(3,2,4)
    # plt.plot(logs[3], label="Ullage Mass")
    # # plt.plot(logs[7], label="Liquid Mass")
    # plt.plot(logs[11], label="Gas Mass")
    # plt.legend()
    #
    #
    # plt.subplot(3,2,5)
    plt.plot(t, T, label="Thrust ")
    plt.legend()

    # plt.subplot(2,2,4)
    # plt.plot(t, T, label="Thrust")

    #lt.xlabel("Time [s]")
    #plt.ylabel("Thrust [N]")
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import math
import matplotlib.pyplot as plt
from .LiquidModels import LOX, RP1, BiProp, UllageGas
from .RegulatorsTanksModels import DomeReg, PropTank

import scipy.optimize


class Engine:
    def __init__(self, air_profile: object, pc_desire: float = 551581, safety_margin: float = 0.2):
        self.Pc_desired = pc_desire
        self.safety_margin = safety_margin
        dome_reg = self.Pc_desired  # + (self.Pc_desired * self.safety_margin)

        mdot = 4.2134195

        lox_reg = DomeReg(outlet_pressure=dome_reg)
        fuel_reg = DomeReg(outlet_pressure=dome_reg)
        ullage = UllageGas(P0=1657895.4223417067, V0=0.05)
        self.combustion_chamber = CombustionChamber(
            biprop=BiProp(),
            fuel=RP1(),
            lox=LOX(),
            air_profile=air_profile,
            of_ratio=1.8,
            Pc=self.Pc_desired,
            mdot=mdot,
            eps=1.936,
            At=0.00825675768,  # m²
            ullage_tank=ullage,
            lox_reg=lox_reg,
            fuel_reg=fuel_reg,
            lox_tank=PropTank(volume=0.05625664329844635, fluid=LOX(), dome_reg=lox_reg, ullage=ullage),
            fuel_tank=PropTank(volume=0.04402911911375661, fluid=RP1(), dome_reg=fuel_reg, ullage=ullage)
        )

        self.nozzle = Nozzle(combustion_chamber=self.combustion_chamber,air=air_profile)

    def runBurn(self, dt: float, alt_m: float = 0) -> float:
        """
        Processes the thrust time by firing combustion chamber for raw value
        and then passing to nozzle for pressure adjustment
        :param dt: time step [dt]
        :param alt_m: current altitude [m]
        :return:
        """
        raw_thrust = self.combustion_chamber.burnStep(dt=dt)
        adjusted_thrust = self.nozzle.getThrust(thrust=raw_thrust, alt_m=alt_m)
        if adjusted_thrust <= 0:
            self.combustion_chamber.active = False
            return 0

        return adjusted_thrust

    def getFluidMass(self) -> float:
        """
        Acquired total fluid and gas tank mass
        :return:
        """
        mass_lox = self.combustion_chamber.lox_tank.mass
        mass_fuel = self.combustion_chamber.fuel_tank.mass
        mass_ullage = self.combustion_chamber.ullage_tank.m_total
        return mass_lox + mass_fuel + mass_ullage


# class RocketEngine:
#     def __init__(self):
#         # self.isp = 300
#         self.massFlowRate = 4.2134195   # average mdot [kg/s]
#         self.burnTime = 23.7            # [s]
#         self.totalImpulse = 210684      # [N-s]
#
#         self.chamberExitRatio = 1/8
#
#     def getThrust(self, time: float, pressure: float) -> np.ndarray:
#         if time < self.burnTime:
#
#             thrust = np.polyval([0.000127581, -0.0164375, 0.92882, -36.2125, 1998.08], time)
#             Pc = self.getChamberPressure(time)
#             Pe = Pc / 8
#             T = thrust * 4.448 + (Pe*6894.76 - pressure) * (5.362**2/4*math.pi/39.37**2) # lbf * N/lbf
#             return np.array([0, 0, T])
#         else:
#             return np.zeros(3)
#
#     def getMassFlowRate(self, time: float, pressure: float) -> float:
#         if time <= self.burnTime:
#             # 1) get current thrust
#             thrust_vec = self.getThrust(time, pressure)
#             F = np.linalg.norm(thrust_vec)
#
#             # 2) compute total propellant mass
#             m_prop = self.massFlowRate * self.burnTime
#
#             # 3) scale F(t) so integral mdot = m_prop
#             val = F * (m_prop / self.totalImpulse)
#             print(val)
#             return val
#         return 0.0
#
#     def getChamberPressure(self, time: float):
#         return np.polyval([5.10325e-06, -0.000657499, 0.0371528, -1.4485, 79.9233], time)
#
#     def getExitPressure(self, time: float):
#         return self.getChamberPressure(time) * self.chamberExitRatio
#
#     def getPropUsed(self, time: float):
#         return min(self.massFlowRate * time, self.massFlowRate * self.burnTime)
#
#     def getRemainingProp(self, time: float):
#         return max(0.0, self.massFlowRate * self.burnTime - self.getPropUsed(time))


class CombustionChamber:
    def __init__(self, biprop: BiProp, fuel: RP1, lox: LOX, air_profile: object,
                 of_ratio: float, Pc: float, mdot: float ,eps: float, At: float = 0.00825675768,
                 lox_tank: PropTank = None, fuel_tank: PropTank = None,
                 ullage_tank: UllageGas = None, lox_reg: DomeReg = None, fuel_reg: DomeReg = None):

        # Target mdot_total = 4.2134 kg/s
        # Target of ratio   = 1.8
        # Target mdot_fuel  = 1.4197195
        # Target mdot_lox   = 2.7207755
        # Target Pc_initial = 551581

        # Indicates if the engine is active, will be switched to false on loss of fluids or pressure
        self.active = True

        # Liquids
        self.lox = lox
        self.fuel = fuel
        self.biprop = biprop
        # self.air = air if air is not None else AirProfile()

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

    def getMassFlowRate(self, Pc: float = None) -> (float, float):
        """
        Uses chamber pressure, throat area, and characteristic velocity to compute mass flow rate
        mdot = Pc * At / c*
        mdot_f = mdot / (1 + o/f)
        mdot_l = mdot - mdot_f
        :param Pc: chamber pressure passively updated
        :return:
        """
        Pc = Pc if Pc is not None else self.Pc
        mdot = Pc * self.At / self.c_star
        # mdot = 4.2134195
        mdot_f = mdot / (1 + self.of_ratio)
        mdot_l = mdot - mdot_f
        # print(Pc, mdot, self.c_star, self.At)
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
        """

        x, y = self.getMassFlowRate(feed_pressure)
        self.Pc = (x+y) * self.c_star / self.At
        # print(feed_pressure, self.Pc)

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

    def burnStep(self, dt: float) -> float:
        """
        An iteration of time step, changes the mass in each tank, returns the RAW thrust,
        will return no thrust if tanks are empty
        :param dt: time step [s]
        :return: Thrust [N]]
        """
        if not self.active:
            self.Pc = 0
            self.mdot = 0
            #print("Engine deactivated")
            return 0

        # STEP 1: Get fuel mass in tanks
        mf_mass = self.fuel_tank.mass
        mo_mass = self.lox_tank.mass


        # STEP 2: Check if tanks have liquid
        if mf_mass <= 0 or mo_mass <= 0:
            self.active = False
            print(f"[SHUTDOWN] Tanks empty: {mf_mass} --- {mo_mass}")
            print(f"Remaining Ullage Mass: {self.ullage_tank.m} Pressure: {self.ullage_tank.P}")
            return 0

        # STEP 3: Get current flow rates
        mdot_f, mdot_o = self.getMassFlowRate()

        #print(f"MDOT_F: {mdot_f}  -  MDOT_O: {mdot_o}  -  dt: {dt}")

        # STEP 4: Subtract mass from tanks
        mf_used = mdot_f * dt
        mo_used = mdot_o * dt

        self.fuel_tank.mass -= mf_used
        self.lox_tank.mass -= mo_used

        # STEP 5: Update ullage tank pressure
        self.lox_tank.volumeChange(dm=mo_used, dt=dt)
        self.fuel_tank.volumeChange(dm=mf_used, dt=dt)
        required_pressure = max(self.lox_reg.outletPressure, self.fuel_reg.outletPressure)
        lowest_tank = min(self.lox_tank.gas_pressure, self.fuel_tank.gas_pressure)

        # # Determine degraded feed pressure
        feed_pressure = min(required_pressure, lowest_tank)


        # STEP 6: Update Chamber Pressure
        self.updateChamberPressure(feed_pressure)

        # STEP 7: Update gas properties
        self._update_gas_properties()

        # Get thrust
        thrust = self.getThrust()
        # print(thrust)

        if thrust <= 0:
            self.active = False
            return 0

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

    def _get_tank_sizes(self, burn_time: float, fuel_temp: float, ox_temp: float, mdot_desired: float = None,
                        of_ratio: float = None) -> dict:
        """
        Function to compute required fuel and oxidizer tank volumes [m³]
        :param burn_time: burn duration [s]
        :param fuel_temp: fuel temp for density lookup [K]
        :param ox_temp: oxidizer temp [K]
        :param mdot_desired: Desired mass flow rate
        :return: dict with fuel and oxidizer tank volumes in m3, mass in kg, and pressure
        """
        if (mdot_desired and of_ratio) is not None:
            mf = mdot_desired / (1 + of_ratio)
            mo = mdot_desired - mf
        else:
            mf, mo = self.getMassFlowRate()



        rho_fuel = self.fuel.liquidDensity(fuel_temp)
        rho_ox = self.lox.liquidDensity(ox_temp)

        Vf = (mf * burn_time) / rho_fuel
        Vo = (mo * burn_time) / rho_ox

        return {
            "fuel_mass": mf * burn_time,
            "oxidizer_mass": mo * burn_time,
            "fuel_volume": Vf,
            "oxidizer_volume": Vo
        }

    def _get_ullage_pressurization(self, burntime: float,
                                   target_pressure: float,
                                   pressurant_v: float,
                                   mdot_desired: float,
                                   of_ratio: float,
                                   fuel_temp: float = 288.15,
                                   ox_temp: float = 90.0,
                                   pressurant_temp: float = 285.0,
                                   pressurant_R: float = 296.8) -> dict:
        """
        Estimates ullage gas volume and pressure requirements
        :param burntime: burn duration [s]
        :param fuel_temp: fuel temp [K]
        :param ox_temp: oxidizer temp [K]
        :param pressurant_temp: helium/N2 temp [K]
        :param pressurant_R: specific gas constant of pressurant [J/kg·K]
        :param pressurant_v: Ullage Tank volume [m3]
        :param target_pressure: desired initial tank pressure [Pa] (defaults to Pc)
        :return: dict of ullage volumes and pressures
        """
        target_pressure = target_pressure or self.Pc
        print(f"Target pressure {target_pressure}")

        tanks = self._get_tank_sizes(burn_time=burntime, fuel_temp=fuel_temp, ox_temp=ox_temp,
                                     mdot_desired=mdot_desired, of_ratio=of_ratio)
        Vf = tanks["fuel_volume"]
        Vo = tanks["oxidizer_volume"]
        mf = tanks["fuel_mass"]
        mo = tanks["oxidizer_mass"]

        # Pressurant mass needed just to occupy the fuel/LOX tanks at Pc pressure
        fuel_press_mass = (target_pressure * Vf) / (pressurant_R * pressurant_temp)
        lox_press_mass = (target_pressure * Vo) / (pressurant_R * pressurant_temp)
        # tank_press_mass = (1.2 * target_pressure / (pressurant_R * pressurant_temp) * (Vf + Vo))

        # Pressurant needed to equalize empty tanks
        ullage_press_mass = ((target_pressure * pressurant_v) / (pressurant_R * pressurant_temp))

        # Total pressurant mass needed
        press_mass = fuel_press_mass + lox_press_mass + ullage_press_mass

        # Initial ullage tank pressure dependent on mass and volume
        init_press_p = press_mass * pressurant_R * pressurant_temp / pressurant_v

        return {
            "fuel_tank": {
                "fuel_volume": Vf,
                "fuel_mass": mf,
                "pressurant_mass": fuel_press_mass
            },
            "lox_tank": {
                "lox_volume": Vo,
                "lox_mass": mo,
                "pressurant_mass": lox_press_mass
            },
            "pressurant_tank": {
                "total_pressurant_mass": press_mass,
                "pressurant_pressure": init_press_p
            }
        }

    def get_fluid_setup_info(self, target_pressure: float, mdot_desired: float, of_ratio: float, burntime: float = 30, pressurant_v: float = 0.05):
        """
        Runs a script to determine the information needed for a proper functioning fluids system
        MUST HAVE AT LEAST MASS FLOW RATE, LOX, and FUEL IMPLEMENTED
        :return:
        """
        press_data = self._get_ullage_pressurization(burntime=burntime, pressurant_v=pressurant_v,
                                                     target_pressure=target_pressure, mdot_desired=mdot_desired,
                                                     of_ratio=of_ratio)
        print(f"Rec: Fuel volume -------------- : {press_data["fuel_tank"]["fuel_volume"]} m3")
        print(f"Rec: Fuel Mass ---------------- : {press_data["fuel_tank"]["fuel_mass"]} kg")
        print(f"Rec: LOX volume --------------- : {press_data["lox_tank"]["lox_volume"]} m3")
        print(f"Rec: LOX Mass ----------------- : {press_data["lox_tank"]["lox_mass"]} kg")
        print(f"Rec: Total Pressurant Mass ---- : {press_data['pressurant_tank']["total_pressurant_mass"]} kg")
        print(f"Rec: Total Pressurant Pressure  : {press_data["pressurant_tank"]["pressurant_pressure"]} Pa")


class Nozzle:
    def __init__(self, air: object, combustion_chamber: CombustionChamber = None,
                 Ae: float = 0.0159851293):
        self.Ae = Ae
        self.air = air
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
    from models.EnvironmentalModels import AirProfile
    engine = Engine(air_profile=AirProfile())
    engine.combustion_chamber.get_fluid_setup_info(target_pressure=551581, of_ratio=1.8,
                                                   mdot_desired=4.2134195, burntime=23.7)
    t = []
    T = []
    alt = []

    dt = 0.1
    time_val = 0.0
    for x in range(round(200 / 0.1)):
        if not engine.combustion_chamber.active:
            print(f"Engine shutdown at {time_val:.2f} seconds")
            break

        y = engine.runBurn(dt=dt, alt_m=0)
        T.append(y)
        t.append(time_val)
        alt.append(x**2)

        time_val += dt

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

    plt.subplot(3,2,1)
    # plt.plot(logs[0], label="Ullage Volume")
    # plt.plot(logs[4], label="Liquid Volume")
    plt.plot(logs[8], label="Gas Volume")
    plt.legend()

    plt.subplot(3,2,2)
    # plt.plot(logs[1], label="Ullage Temp")
    # plt.plot(logs[5], label="Liquid Temp")
    plt.plot(logs[9], label="Gas Temp")
    plt.legend()

    plt.subplot(3,2,3)
    # plt.plot(logs[2], label="Ullage Pressure")
    # plt.plot(logs[6], label="Liquid Pressure")
    plt.plot(logs[10], label="Gas Pressure")
    plt.legend()

    plt.subplot(3,2,4)
    # plt.plot(logs[3], label="Ullage Mass")
    # plt.plot(logs[7], label="Liquid Mass")
    plt.plot(logs[11], label="Gas Mass")
    plt.legend()


    plt.subplot(3,2,5)
    plt.plot(t, T, label="Thrust ")
    plt.legend()

    # plt.subplot(2,2,4)
    # plt.plot(t, T, label="Thrust")

    #lt.xlabel("Time [s]")
    #plt.ylabel("Thrust [N]")
    plt.legend()
    plt.tight_layout()
    plt.show()

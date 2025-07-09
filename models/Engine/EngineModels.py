import numpy as np
import math
import matplotlib.pyplot as plt
from rocketcea.cea_obj import CEA_Obj
from LiquidModels import LOX, RP1, UllageGas
from RegulatorsTanksModels import DomeReg, PropTank
from models.EnvironmentalModels import AirProfile
import scipy.optimize

class RocketEngine:
    def __init__(self):
        # self.isp = 300
        self.massFlowRate = 4.2134195   # average mdot [kg/s]
        self.burnTime = 23.7            # [s]
        self.totalImpulse = 210684      # [N-s]

        self.chamberExitRatio = 1/8

    def getThrust(self, time: float, pressure: float) -> np.ndarray:
        if time < self.burnTime:

            thrust = np.polyval([0.000127581, -0.0164375, 0.92882, -36.2125, 1998.08], time)
            Pc = self.getChamberPressure(time)
            Pe = Pc / 8
            T = thrust * 4.448 + (Pe*6894.76 - pressure) * (5.362**2/4*math.pi/39.37**2) # lbf * N/lbf
            return np.array([0, 0, T])
        else:
            return np.zeros(3)

    def getMassFlowRate(self, time: float, pressure: float) -> float:
        if time <= self.burnTime:
            # 1) get current thrust
            thrust_vec = self.getThrust(time, pressure)
            F = np.linalg.norm(thrust_vec)

            # 2) compute total propellant mass
            m_prop = self.massFlowRate * self.burnTime

            # 3) scale F(t) so integral mdot = m_prop
            val = F * (m_prop / self.totalImpulse)
            print(val)
            return val
        return 0.0

    def getChamberPressure(self, time: float):
        return np.polyval([5.10325e-06, -0.000657499, 0.0371528, -1.4485, 79.9233], time)

    def getExitPressure(self, time: float):
        return self.getChamberPressure(time) * self.chamberExitRatio

    def getPropUsed(self, time: float):
        return min(self.massFlowRate * time, self.massFlowRate * self.burnTime)

    def getRemainingProp(self, time: float):
        return max(0.0, self.massFlowRate * self.burnTime - self.getPropUsed(time))


class Fluids:
    def __init__(self, Cd: float, A: float, gamma: float, R: float):
        """
        Fluids system
        :param Cd: Discharge Coefficient
        :param A: Throat/orifice area [m2]
        :param gamma: Heat capacity ratio
        :param R: Specific gas const
        """
        self.Cd = Cd
        self.A = A
        self.gamma = gamma
        self.R = R

    def computeMdot(self, P_tank, P_downstream, T_tank):
        """
        Computes mdot based tank pressure, downstream tank pressure, and tank temperature
        :param P_tank:
        :param P_downstream:
        :param T_tank:
        :return: mass flow rate [kg/s]
        """
        # CR = Pt / Pc = (2 / (k + 1)) ** (k / (k - 1))
        critical_ratio = (2 / (self.gamma + 1)) ** (self.gamma / (self.gamma - 1))

        if P_downstream / P_tank > critical_ratio:
            # Unchoked flow -- flow is subsonic at nozzle throat
            pressure_ratio = P_downstream / P_tank
            term1 = 2 * self.gamma / (self.gamma -1)
            term2 = (1 - pressure_ratio ** ((self.gamma - 1) / self.gamma))
            if term2 < 0:
                return 0.0  # no flow, avoids math domain error

            mdot = self.Cd * self.A * P_tank * math.sqrt(term1 * term2 / (self.R * T_tank))

        else:
            # Choked flow -- flow is supersonic
            mdot = self.Cd * self.A * P_tank * math.sqrt(
                self.gamma / (self.R * T_tank) *
                (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (self.gamma - 1))
            )

        return mdot


class CombustionChamber:
    def __init__(self, fuel: RP1, lox: LOX, of_ratio: float, Pc: float, eps: float, At: float,
                 air: AirProfile = None, lox_tank: PropTank = None, fuel_tank: PropTank = None,
                 ullage_tank: UllageGas = None):

        # Indicates if the engine is active, will be switched to false on loss of pressurant
        self.active = True

        # Liquids
        self.lox = lox
        self.fuel = fuel
        self.air = air if air is not None else AirProfile()

        # Tanks
        self.lox_tank = lox_tank if lox_tank is not None else PropTank(volume=0.19, fluid=self.lox)
        self.fuel_tank = fuel_tank if fuel_tank is not None else PropTank(volume=0.1, fluid=self.fuel)
        self.ullage_tank = ullage_tank if ullage_tank is not None else UllageGas()

        # Combustion Chamber Properties
        self.of_ratio = of_ratio
        self.Pc = Pc
        self.Pe = None
        self.Ve = 0
        self.eps = eps
        self.g = 9.08665       # m/s2
        self.At = At

        self._update_gas_properties()

    def _update_gas_properties(self):
        """Calls fuels CEA interface to update combustion properties"""
        props = self.fuel.gasProperties(self.Pc, self.of_ratio, self.eps)
        self.Tc = props["T_c"]
        self.gamma = props["gamma"]
        self.R = props["R_specific"]
        self.isp_vac = props["Isp_vac"]
        self.isp_sea = props["Isp_sea"]
        self.c_star = props["c_star"]
        self.updateExitPressure()
        self.updateExhaustVelocity()

    def updateConditions(self, Pc: float = None, of_ratio: float = None, eps: float = None):
        """
        Updates the conditions that will be used
        :param Pc: Chamber pressure [Pa]
        :param of_ratio: Oxygen Fuel Ratio
        :param eps: Nozzle expansion ratio
        :return:
        """
        if Pc is not None:
            self.Pc = Pc
        if of_ratio is not None:
            self.of_ratio = of_ratio
        if eps is not None:
            self.eps = eps

        self._update_gas_properties()


    def getCurrentIsp(self, vacuum: bool = True) -> float:
        """
        Returns an isp based on vacuum
        :param vacuum: bool
        :return:
        """
        return self.isp_vac if vacuum else self.isp_sea

    def getMassFlowRate(self):
        """
        Uses chamber pressure, throat area, and characteristic velocity to compute mass flow rate
        mdot = Pc * At / c*
        mdot_f = mdot / (1 + o/f)
        mdot_l = mdot - mdot_f
        :return:
        """
        mdot = self.Pc * self.At / self.c_star
        mdot_f = mdot / (1 + self.of_ratio)
        mdot_l = mdot - mdot_f

        return mdot_f, mdot_l

    def updateExitPressure(self):
        """Updates exit pressure"""
        Me = self._solve_exit_mach()
        self.Me = Me
        exp = (1 + (self.gamma - 1) / 2 * Me**2)
        self.Pe = self.Pc / (exp**(self.gamma / (self.gamma - 1)))

    def updateExhaustVelocity(self):
        """Exhaust velocity equation"""
        term1 = 2 * self.gamma * self.R * self.Tc / (self.gamma - 1)
        exp = (self.gamma - 1) / self.gamma
        term2 = (1 - ((self.Pe / self.Pc)**exp))
        self.Ve = math.sqrt(term1 * term2)

    def updateChamberPressure(self):
        """Updates chamber pressure using mdot c* / At"""
        mdot_f, mdot_o = self.getMassFlowRate()
        mdot_total = mdot_f + mdot_o
        self.Pc = mdot_total * self.c_star / self.At

    def getThrust(self) -> np.ndarray:
        """
        Gets raw engine thrust using: F = mdot * ISP * g
        This is the raw thrust without nozzle affects
        :return: thrust in [x,y,z], mass flow rate fuel, mass flow rate lox
        """
        mdot_f, mdot_o = self.getMassFlowRate()
        mdot = mdot_o + mdot_f
        F = mdot * self.Ve
        return np.array([0.0, 0.0, F])

    def burnStep(self, dt: float):
        """
        An iteration of time step, changes the mass in each tank, returns the RAW thrust,
        will return no thrust if tanks are empty
        :param dt: time step [s]
        :return: Thrust vector, mass flow rate fuel, mass flow rate lox
        """
        if not self.active:
            return np.zeros(3), 0, 0

        # STEP 1: Get fuel mass in tanks
        mf_mass = self.fuel_tank.mass
        mo_mass = self.lox_tank.mass

        # STEP 2: Check if tanks have liquid
        if mf_mass <= 0 or mo_mass <= 0:
            self.active = False
            return np.zeros(3), 0, 0

        # STEP 3: Get current flow rates
        mdot_f, mdot_o = self.getMassFlowRate()
        mdot_tot = mdot_f + mdot_o

        # STEP 4: Subtract mass from tanks
        mf_used = mdot_f * dt
        mo_used = mdot_o * dt


        if mf_mass > mf_mass or mo_used > mo_mass:
            # Prevent overdraw if near empty
            dt_actual = min(mf_mass / mdot_f, mo_mass / mdot_o)
            mf_used = mdot_f * dt_actual
            mo_used = mdot_o * dt_actual
            self.active = False
        else:
            dt_actual = dt

        self.fuel_tank.mass -= mf_used
        self.lox_tank.mass -= mo_used

        # STEP 5: Update ullage tank pressure
        self.ullage_tank.gasLeaving(dt=dt_actual, mdot=mdot_tot)

        # STEP 6: Update Chamber Pressure
        self.updateChamberPressure()

        # STEP 7: Update gas properties
        self._update_gas_properties()

        # STEP 8: Update exit pressure, exhaust velocity

        # Get thrust
        thrust = self.getThrust()


        return thrust, mdot_f, mdot_o



    def _solve_exit_mach(self) -> float:
        """Solves for exit Mach number using isentropic area-Mach relation"""
        gamma = self.gamma
        eps = self.eps

        def func(M):
            lhs = eps
            rhs = (1 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M ** 2)) ** ((gamma + 1) / (2 * (gamma - 1)))
            return lhs - rhs

        # Only supersonic exit flows are valid (e.g., M_e > 1)
        return scipy.optimize.brentq(func, 1.01, 10.0)


    def _get_rawthrust_over_time(self, step=1000, dt=1.0):
        """
        Performs a sweep over a set altitude to analyze the change in thrust
        time steps are not accurate and are only for mdot
        :param step: Sweep steps [m]
        :param dt: Time step in seconds
        :return: list: h, thrust
        """
        results = []
        time = 0.0

        while time <= (step / dt):
            thrust_vec, mf_mass, mo_mass = self.burnStep(dt=dt)


            if np.all(thrust_vec == 0):
                break

            results.append([time, thrust_vec, mf_mass, mo_mass])

            time += step
        return results

    def _get_tank_sizes(self, burn_time: float, fuel_temp: float = 288.15, ox_temp: float = 90.0) -> dict:
        """
        Function to compute required fuel and oxidizer tank volumes [m³]
        :param burn_time: burn duration [s]
        :param fuel_temp: fuel temp for density lookup [K]
        :param ox_temp: oxidizer temp [K]
        :return: dict with fuel and oxidizer tank volumes in m3
        """
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
                                   fuel_temp: float = 288.15,
                                   ox_temp: float = 90.0,
                                   ullage_frac: float = 0.05,
                                   pressurant_temp: float = 300.0,
                                   pressurant_R: float = 2077,
                                   pressurant_gamma: float = 1.66,
                                   target_pressure: float = None) -> dict:
        """
        Estimates ullage gas volume and pressure requirements
        :param burntime: burn duration [s]
        :param fuel_temp: fuel temp [K]
        :param ox_temp: oxidizer temp [K]
        :param ullage_frac: initial ullage fraction (e.g. 0.05 for 5%)
        :param pressurant_temp: helium/N2 temp [K]
        :param pressurant_R: specific gas constant of pressurant [J/kg·K]
        :param pressurant_gamma: gamma for pressurant gas
        :param target_pressure: desired initial tank pressure [Pa] (defaults to Pc)
        :return: dict of ullage volumes and pressures
        """
        target_pressure = target_pressure or self.Pc

        tanks = self._get_tank_sizes(burn_time=burntime, fuel_temp=fuel_temp, ox_temp=ox_temp)
        Vf = tanks["fuel_volume"]
        Vo = tanks["oxidizer_volume"]

        ullage_fuel_0 = Vf * ullage_frac
        ullage_ox_0 = Vo * ullage_frac

        ullage_fuel_1 = Vf + ullage_fuel_0
        ullage_ox_1 = Vo + ullage_ox_0

        # Pressurant mass needed -- m = PV / RT
        mf_press = (target_pressure * ullage_fuel_0) / (pressurant_R * pressurant_temp)
        mo_press = (target_pressure * ullage_ox_0) / (pressurant_R * pressurant_temp)

        # Final pressure after full burn (assuming isentropic expansion)
        pf_final = target_pressure * (ullage_fuel_0 / ullage_fuel_1) ** pressurant_gamma
        po_final = target_pressure * (ullage_ox_0 / ullage_ox_1) ** pressurant_gamma

        return {
            "fuel_tank": {
                "initial_ullage_volume": ullage_fuel_0,
                "final_ullage_volume": ullage_fuel_1,
                "pressurant_mass": mf_press,
                "final_pressure": pf_final
            },
            "oxidizer_tank": {
                "initial_ullage_volume": ullage_ox_0,
                "final_ullage_volume": ullage_ox_1,
                "pressurant_mass": mo_press,
                "final_pressure": po_final
            },
            "total_pressurant_mass": mf_press + mo_press
        }



class Nozzle:
    def __init__(self, air: AirProfile = None, combustion_chamber: CombustionChamber = None,
                 Ae: float = 0.0):
        self.Ae = Ae
        self.air = air if air is not None else AirProfile()
        self.combustionChamber = combustion_chamber if combustion_chamber is not None else CombustionChamber()

    def getThrust(self, alt_m: float = 0, dt: float = 0):
        """
        Uses raw thrust (mdot Ve) + dP * Ae
        :param alt_m:
        :param dt:
        :return:
        """
        thrust_raw, _, _ = self.combustionChamber.burnStep(dt=dt)
        pres_atm = self.air.getStaticPressure(alt_m=alt_m)
        pres_exit = self.combustionChamber.Pe
        thrust_total = thrust_raw  + ((pres_exit - pres_atm) * self.Ae)










if __name__ == "__main__":

    chamber = CombustionChamber(
        fuel=RP1(),
        lox=LOX(),
        of_ratio=2.6,
        Pc=7e6,
        eps=40,
        At=0.00825675768,  # m²
    )
    tank_sizes = chamber._get_tank_sizes(burn_time=30)
    print(f"Fuel Tank: {tank_sizes['fuel_volume']:.2f} m3")
    print(f"Oxidizer Tank: {tank_sizes['oxidizer_volume']:.2f} m3")

    press_data = chamber._get_ullage_pressurization(burntime=60)
    print(f"Total Pressurant Mass: {press_data['total_pressurant_mass']:.4f} kg")
    print(f"Final Fuel Tank Pressure: {press_data['fuel_tank']['final_pressure']:.2f} Pa")


    thrust_data = chamber._get_rawthrust_over_time()

    time = [entry[0] for entry in thrust_data]
    thrs = [entry[1] for entry in thrust_data]
    thrust_z = np.array([vec[2] for vec in thrs])
    plt.plot(time, thrust_z)
    plt.xlabel("time")
    plt.ylabel("Thrust")
    plt.show()

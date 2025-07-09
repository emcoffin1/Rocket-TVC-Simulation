import numpy as np
import math
import matplotlib.pyplot as plt
from rocketcea.cea_obj import CEA_Obj
from LiquidModels import LOX, RP1, UllageGas
from RegulatorsTanksModels import DomeReg, PropTank
from models.EnvironmentalModels import AirProfile

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
    def __init__(self, fuel: RP1, lox: LOX, of_ratio: float, Pc: float, eps: float, At: float, Ae: float,
                 air: AirProfile = None, lox_tank: PropTank = None, fuel_tank: PropTank = None):
        self.lox = lox
        self.fuel = fuel
        self.air = air if air is not None else AirProfile()

        self.lox_tank = lox_tank if lox_tank is not None else PropTank(volume=0.19, fluid=self.lox)
        self.fuel_tank = fuel_tank if fuel_tank is not None else PropTank(volume=0.1, fluid=self.fuel)

        self.of_ratio = of_ratio
        self.Pc = Pc
        self.eps = eps
        self.g = 9.08665       # m/s2
        self.At = At
        self.Ae = Ae

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

    def getThrust(self, alt_m: float = 0.0) -> np.ndarray:
        """
        Gets raw engine thrust using: F = mdot * ISP * g = mdot * ISP(vac) * g - (Pamb - Ae)
        :param alt_m: Altitude [m]
        :return: thrust in [x,y,z]
        """
        air_p = self.air.getStaticPressure(alt_m=alt_m)
        isp = self.getCurrentIsp()
        mdot_f, mdot_o = self.getMassFlowRate()
        mdot = mdot_o + mdot_f
        F = mdot * isp * self.g - (air_p * self.Ae)
        return np.array([0.0, 0.0, F])

    def burnStep(self, dt: float, alt_m: float):
        """
        An iteration of time step, changes the mass in each tank, returns the thrust,
        will return no thrust if tanks are empty
        :param dt: time step [s]
        :param alt_m: current altitude [m]
        :return:
        """
        # Subtract mass from tanks
        mf_mass = self.fuel_tank.mass
        mo_mass = self.lox_tank.mass

        # Check if fuel is still in tanks
        if mf_mass <= 0 or mo_mass <= 0:
            return np.zeros(3), mf_mass, mo_mass

        mf_rate, mo_rate = self.getMassFlowRate()
        mf_used = mf_rate * dt
        mo_used = mo_rate * dt
        mf_mass -= mf_used
        mo_mass -= mo_used

        # Get thrust
        thrust = self.getThrust(alt_m=alt_m)

        return thrust, mf_mass, mo_mass

    def _get_thrust_over_alt(self, max_alt_m=100e3, step=1000, dt=1.0):
        """
        Performs a sweep over a set altitude to analyze the change in thrust
        time steps are not accurate and are only for mdot
        :param max_alt_m: Max altitude to sweep to [m]
        :param step: Sweep steps [m]
        :param dt: Time step in seconds
        :return: list: h, thrust
        """
        results = []
        altitude = 0.0

        while altitude <= max_alt_m:
            thrust_vec, mf_mass, mo_mass = self.burnStep(dt=dt, alt_m=altitude)

            if np.all(thrust_vec == 0):
                break

            results.append([altitude, thrust_vec, mf_mass, mo_mass])

            altitude += step
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









if __name__ == "__main__":

    chamber = CombustionChamber(
        fuel=RP1(),
        lox=LOX(),
        of_ratio=2.6,
        Pc=7e6,
        eps=40,
        At=0.00825675768,  # m²
        Ae=0.0159851293
    )
    tank_sizes = chamber._get_tank_sizes(burn_time=30)
    print(f"Fuel Tank: {tank_sizes['fuel_volume']:.2f} m3")
    print(f"Oxidizer Tank: {tank_sizes['oxidizer_volume']:.2f} m3")

    press_data = chamber._get_ullage_pressurization(burntime=60)
    print(f"Total Pressurant Mass: {press_data['total_pressurant_mass']:.4f} kg")
    print(f"Final Fuel Tank Pressure: {press_data['fuel_tank']['final_pressure']:.2f} Pa")


    thrust_data = chamber._get_thrust_over_alt()

    alts = [entry[0] for entry in thrust_data]
    thrs = [entry[1] for entry in thrust_data]
    mf_l = [entry[2] for entry in thrust_data]
    mf_0 = [entry[3] for entry in thrust_data]
    thrust_z = np.array([vec[2] for vec in thrs])
    plt.plot(alts, mf_l)
    plt.xlabel("Alts")
    plt.ylabel("Thrust")
    plt.show()
    # #chamber.summary()
    #
    # # Or just get the mass flow directly:
    # mdot = chamber.getMassFlowRate()
    # print(f"Mass Flow: {mdot:.2f} kg/s")
    # print(f"ISP: {chamber.getCurrentIsp()}")

    # # === Simulation Setup ===
    #
    # # Shared ullage gas
    # ullage = UllageGas(P0=2e6, V0=0.01, T0=300, R=296.8, gamma=1.4, isothermal=True)
    #
    # # Regulators
    # reg_lox = DomeReg(outlet_pressure=2.5e5)  # LOX line pressure
    # reg_fuel = DomeReg(outlet_pressure=2.1e5)  # Fuel line pressure
    #
    # # Tanks
    # tank_lox = PropTank(volume=0.02, initial_mass=10, fluid_density=1141)
    # tank_fuel = PropTank(volume=0.01, initial_mass=5, fluid_density=820)
    #
    # # Flow lines
    # ox_flow = Fluids(Cd=0.8, A=2.5e-5, gamma=1.4, R=296.8)
    # fuel_flow = Fluids(Cd=0.8, A=1.5e-5, gamma=1.4, R=296.8)
    #
    # # Simulation loop
    # dt = 1.0
    # total_time = 2000
    #
    # log = {
    #     'time': [],
    #     'mdot_lox': [],
    #     'mdot_fuel': [],
    #     'of_ratio': [],
    #     'ullage_pressure': [],
    #     'tank_lox_mass': [],
    #     'tank_fuel_mass': []
    # }
    #
    # cutoff_triggered = False
    # min_pressure = 1.1e5
    #
    # for t in range(int(total_time / dt)):
    #     if cutoff_triggered:
    #         print(f"--- CUTOFF at {t*dt:.1f}s ---")
    #         break
    #     P_ull = ullage.getPressure
    #     T_ull = ullage.getTemperature
    #
    #     # Check cutoff flag
    #     if P_ull < min_pressure:
    #         cutoff_triggered = True
    #         continue
    #     if tank_fuel.mass <= 0 or tank_lox.mass <= 0:
    #         cutoff_triggered = True
    #         continue
    #
    #     # LOX
    #     P_lox = reg_lox.regulate(P_ull)
    #     mdot_lox = ox_flow.computeMdot(P_lox, P_downstream=1e5, T_tank=T_ull)
    #     tank_lox.withdraw(mdot_lox, dt)
    #
    #     # Fuel
    #     P_fuel = reg_fuel.regulate(P_ull)
    #     mdot_fuel = fuel_flow.computeMdot(P_fuel, P_downstream=1e5, T_tank=T_ull)
    #     tank_fuel.withdraw(mdot_fuel, dt)
    #
    #     # Update ullage expansion
    #     V_new_total = tank_fuel.getVolumeUllage + tank_lox.getVolumeUllage
    #     ullage.expand(V_new_total - ullage.getVolume)
    #
    #     # Log data
    #     of_ratio = mdot_lox / mdot_fuel if mdot_fuel > 0 else float('inf')
    #
    #
    #     log['time'].append(t * dt)
    #     log['mdot_lox'].append(mdot_lox)
    #     log['mdot_fuel'].append(mdot_fuel)
    #     log['of_ratio'].append(of_ratio)
    #     log['ullage_pressure'].append(P_ull)
    #     log['tank_lox_mass'].append(tank_lox.mass)
    #     log['tank_fuel_mass'].append(tank_fuel.mass)
    #
    #
    #
    # # === Plotting ===
    # fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    #
    # axs[0].plot(log['time'], log['mdot_lox'], label='LOX mdot')
    # axs[0].plot(log['time'], log['mdot_fuel'], label='Fuel mdot')
    # axs[0].set_ylabel("Mass Flow Rate [kg/s]")
    # axs[0].legend()
    #
    # axs[1].plot(log['time'], log['of_ratio'])
    # axs[1].set_ylabel("O/F Ratio")
    #
    # axs[2].plot(log['time'], log['ullage_pressure'])
    # axs[2].set_ylabel("Ullage Pressure [kPa]")
    # axs[2].set_xlabel("Time [s]")
    #
    # plt.tight_layout()
    # plt.show()


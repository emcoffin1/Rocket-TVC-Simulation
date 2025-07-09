from rocketcea.cea_obj import CEA_Obj
from math import exp
from warnings import warn


class RP1:
    """
    Provides basic physical and thermochemical constants for sims
    """
    # Liquid Properties (at STP)
    density_liquid      = 810.0     # kg/m3
    viscosity_liquid    = 170e-3    # Pa*s
    flash_point         = 322.0     # K

    # Gas-phase base constants
    _CEA = CEA_Obj(oxName="LOX", fuelName='RP-1')

    @staticmethod
    def liquidDensity(temperature: float) -> float:
        """
        Returns liquid density as function of temperature [kg/m3]
        Linear approximation: destiny decreases with temperature
        :param temperature: [K]
        :return:
        """
        rho0 = RP1.density_liquid
        T0 = 288.15     # Reference temperature [K]
        k = 0.3         # kg/m3 per K
        return rho0 - k * (temperature - T0)

    @staticmethod
    def liquidViscosity(temperature: float) -> float:
        """
        Returns liquid viscosity [Pa*s] as a function of temp [K]
        Uses three-parameter arrhenius equations
        :param temperature: [K]
        :return:
        """
        # Uses octane coefficients
        A = 0.007889  # [Pa*s]
        B = 1456.2  # [K]
        C = -51.44    # [K]
        mPas = A * exp(B / (temperature - C))
        return mPas * 1e-3

    @classmethod
    def gasProperties(cls, chamber_pressure_pa: float, of_ratio: float, eps: float) -> dict:
        """
        Uses CEA to obtain gas-phase properties at different Pc and O/F
        :param chamber_pressure_pa: Chamber pressure [Pa]
        :param of_ratio: Oxidizer to fuel ratio by mass
        :param eps: Nozzle expansion ratio (Ae/At)
        :return:
        """
        # Convert pressure to bar for CEA
        pc_bar = chamber_pressure_pa / 1e5
        # Initialize dict to return
        props = {}


        # Molecular weight nad specific gas const
        m_wt, gamma = cls._CEA.get_Chamber_MolWt_gamma(pc_bar, of_ratio)
        props["gamma"] = gamma
        props["R_specific"] = 8.314 / m_wt

        # Chamber temp
        T_c = cls._CEA.get_Tcomb(pc_bar, of_ratio)
        props["T_c"] = T_c

        # Characteristic vel c*
        cstar_data = cls._CEA.get_Cstar(pc_bar, of_ratio)
        props["c_star"] = cstar_data

        # Vcuum C*
        Ivac, Cstr, Tc_cea = cls._CEA.get_IvacCstrTc(pc_bar, of_ratio)
        props["Isp_vac"] = Ivac

        # SL ISP with ambient estimate
        Isp_sea = cls._CEA.estimate_Ambient_Isp(Pc=pc_bar, MR=of_ratio ,eps=eps, Pamb=1.01325)
        props["Isp_sea"] = Isp_sea[0]

        return props


class LOX:
    """
    Provides basic physical and thermochemical constants for sims
    """
    # Liquid Properties (at STP)
    density_liquid      = 1141.0        # kg/m3
    viscosity_liquid    = 0.2e-3        # Pa*s
    boiling_point       = 90.2          # K

    # Gas-phase base constants
    _CEA = CEA_Obj(oxName="LOX", fuelName='RP-1')

    @staticmethod
    def liquidDensity(temperature: float) -> float:
        """
        Returns liquid density as function of temperature [kg/m3]
        Linear approximation: destiny decreases with temperature
        :param temperature: [K]
        :return:
        """
        rho0 = LOX.density_liquid
        T0 = LOX.boiling_point     # Reference temperature [K]
        k = 0.5         # kg/m3 per K
        return rho0 - k * (temperature - T0)

    @staticmethod
    def liquidViscosity(temperature: float) -> float:
        """
        Returns liquid viscosity [Pa*s] as a function of temp [K]
        Uses power law approximation
        :param temperature: [K]
        :return:
        """
        # Uses octane coefficients
        mu0 = LOX.viscosity_liquid
        T0 = LOX.boiling_point
        exp = -0.7
        return mu0 * (temperature / T0) ** exp

    @classmethod
    def gasProperties(cls, chamber_pressure_pa: float, of_ratio: float, eps: float) -> dict:
        """
        Uses CEA to obtain gas-phase properties at different Pc and O/F
        :param chamber_pressure_pa: Chamber pressure [Pa]
        :param of_ratio: Oxidizer to fuel ratio by mass
        :param eps: Nozzle expansion ratio (Ae/At)
        :return:
        """
        # Convert pressure to bar for CEA
        pc_bar = chamber_pressure_pa / 1e5
        # Initialize dict to return
        props = {}


        # Molecular weight nad specific gas const
        m_wt, gamma = cls._CEA.get_Chamber_MolWt_gamma(pc_bar, of_ratio)
        props["gamma"] = gamma
        props["R_specific"] = 8.314 / m_wt

        # Chamber temp
        T_c = cls._CEA.get_Tcomb(pc_bar, of_ratio)
        props["T_c"] = T_c

        # Characteristic vel c*
        cstar_data = cls._CEA.get_Cstar(pc_bar, of_ratio)
        props["c_star"] = cstar_data

        # Vcuum C*
        Ivac, Cstr, Tc_cea = cls._CEA.get_IvacCstrTc(pc_bar, of_ratio)
        props["Isp_vac"] = Ivac

        # SL ISP with ambient estimate
        Isp_sea = cls._CEA.estimate_Ambient_Isp(Pc=pc_bar, MR=of_ratio ,eps=eps, Pamb=1.01325)
        props["Isp_sea"] = Isp_sea[0]

        return props


class UllageGas:
    """
    Models gas behavior of the pressurant during blow-down
    Can be isothermal or adiabatic
    """
    def __init__(self, P0: float = 2e6, V0: float = 0.01, T0: float = 300, R: float = 296.8, gamma: float = 1.4,
                 density: float = 1.25,isothermal: float=True):
        """

        :param P0: Initial gas pressure [Pa]
        :param V0: Initial tank volume   [m3]
        :param R: Specific gas const
        :param gamma: Heat cap ratio [cp/cv]
        :param density: Density of Nitrogen [kg/m3]
        :param T0: Initial temp     [K]
        :param isothermal: bool - initially true - if false = adiabatic
        """
        self.P = P0
        self.V = V0
        self.T = T0
        self.R = R
        self.density = density
        self.isothermal = isothermal
        self.m = self.P * self.V / (self.R * self.T)
        self.m_used = 0

    def expand(self, dV: float) -> float:
        """
        Expands gas into extra volume and adjust the P, V, and T
        :param dV: Volume which the gas expands into [m3]
        :return: New gas pressure [Pa]
        """
        # New volume
        V_new = self.V + dV

        if V_new <= 0:
            raise ValueError("Ullage volume < 0!")

        if self.isothermal:
            # P * V constant ( Boyle's law )
            P_new = self.P * self.V / V_new
            T_new = self.T

        else:
            # adiabatic P * V^gamma = const
            P_new = self.P * (self.V / V_new) ** self.gamma
            # ideal gas: T = P V / (mR)
            T_new = P_new * V_new / (self.m * self.R)

        # compute new mass using ideal gas law
        m_new = P_new * V_new / (self.R * T_new)
        dm = self.m - m_new
        self.m_used += dm
        self.m = m_new

        # update state
        self.P, self.V, self.T, = P_new, V_new, T_new
        return P_new

    def gasLeaving(self, dt: float, mdot: float):

        # Get mass leaving using mdot * dt
        dm = mdot * dt

        # Subtract mass
        self.m -= dm

        # Check if it's empty
        if self.m <= 0:
            self.m = 0
            warn("Ullage tank is empty")

        else:
            # Update pressure
            self.P = self.m * self.R * self.T / self.V



    @property
    def getPressure(self) -> float:
        """Returns current gas pressure [Pa]"""
        return self.P

    @property
    def getTemperature(self) -> float:
        """Returns current gas temperature [K]"""
        return self.T

    @property
    def getVolume(self) -> float:
        """Returns current gas volume [m3]"""
        return self.V

    @property
    def get_mass(self) -> float:
        """Returns current mass in ullage tank"""
        return self.m

    @property
    def getMassUsed(self) -> float:
        """Returns the mass of ullage used during firing up to point of call"""
        return self.m_used


if __name__ == "__main__":
    f = RP1().gasProperties(1.005,3.2,1.5)
    l = LOX().gasProperties(1.005,3.2,1.5)
    print(f)
    print(l)
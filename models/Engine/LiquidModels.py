from rocketcea.cea_obj import CEA_Obj
from math import exp


class RP1:
    """
    Provides basic physical and thermochemical constants for sims
    """
    # Liquid Properties (at STP)
    density_liquid      = 810.0     # kg/m3
    viscosity_liquid    = 170e-3    # Pa*s
    flash_point         = 322.0     # K
    storing_temp        = 288       # K

    # Gas-phase base constants
    _CEA = CEA_Obj(oxName="LOX", fuelName='RP-1')

    @staticmethod
    def name():
        return "RP1"

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


class LOX:
    """
    Provides basic physical and thermochemical constants for sims
    """
    # Liquid Properties (at STP)
    density_liquid      = 1141.0        # kg/m3
    viscosity_liquid    = 0.2e-3        # Pa*s
    boiling_point       = 90.2          # K
    storing_temp        = 90            # K

    # Gas-phase base constants
    _CEA = CEA_Obj(oxName="LOX", fuelName='RP-1')

    @staticmethod
    def name():
        return "LOX"

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


class BiProp:
    """Class containing information regarding a mixture of LOX and RP-1"""
    _CEA = CEA_Obj(oxName="LOX", fuelName='RP-1')

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
        pc_psi = chamber_pressure_pa / 6894.76
        # Initialize dict to return
        props = {}


        # Molecular weight nad specific gas const
        m_wt, gamma = cls._CEA.get_Chamber_MolWt_gamma(pc_psi, of_ratio, eps)
        props["gamma"] = gamma
        props["R_specific"] = 8.314 / (m_wt * 0.45359237)

        # Chamber temp
        T_c = cls._CEA.get_Tcomb(pc_psi, of_ratio)
        props["T_c"] = (T_c - 491.67) * (5/9) + 273.15

        # Characteristic vel c*
        cstar_data = cls._CEA.get_Cstar(pc_psi, of_ratio)
        props["c_star"] = cstar_data * 0.3048

        # Vcuum C*
        Ivac, Cstr, Tc_cea = cls._CEA.get_IvacCstrTc(pc_psi, of_ratio)
        props["Isp_vac"] = Ivac * 0.3048

        # SL ISP with ambient estimate
        #Isp_sea = cls._CEA.estimate_Ambient_Isp(Pc=pc_psi, MR=of_ratio, eps=eps)
        #props["Isp_sea"] = Isp_sea[0] * 0.3048

        return props


class UllageGas:
    """
    Models gas behavior of the pressurant during blow-down
    Can be isothermal or adiabatic
    """
    def __init__(self, m0: float = 2e6, V0: float = 0.01, T0: float = 285, R: float = 296.8, mdot: float = 4.2134195):
        """

        :param P0: Initial gas pressure             [Pa]
        :param V0: Initial tank volume              [m3]
        :param R: Specific gas const
        :param T0: Initial temp                     [K]
        :param mdot: Constant mass flow rate target [kg/s]
        """
        self.m = m0
        self.V = V0
        self.T = T0
        self.R = R
        self.P = m0 * R * T0 / V0
        self.m_total = self.m
        self.m_used = 0
        self.gamma = 1.4    # Specific heat
        self.mdot = -mdot

        print(f" Loaded: ULLAGE mass:       {self.m}")
        print(f" Loaded: ULLAGE pressure:   {self.P}")

        self.log_P = []
        self.log_V = []
        self.log_T = []
        self.log_m = []

    def gasLeaving(self, dm: float, dt: float):
        """
        Accounts for a mass change as pressurant flows into other tank
        Computes the pressure and temperature following mass change
        :param dm: mass removed from ullage tank to fill fuel/lox tank [kg]
        :param dt: time step [s]
        :return:
        """

        # dm_dt_const = dm/dt
        # print(f"Gas Leaving ---- {dm}")
        #
        # # Compute change in temperature
        # # dT/dt = -mdot * T/m * gamma+1
        # dT_dt = -dm_dt_const * self.T / self.m * (self.gamma + 1)
        # self.T += dT_dt * dt
        #
        # # Compute change in pressure
        # # dP/dt = -mdot * RT/V * gamma
        # dP_dt = -dm_dt_const * (self.R * self.T / self.V) * self.gamma
        # self.P += dP_dt * dt
        #
        # self.m -= dm
        # self.m_used += dm

        # Clamp dm to avoid unphysical mass removal
        dm = min(dm, self.m)
        if dm <= 0:
            self.log_m.append(self.m)
            self.log_P.append(self.P)
            self.log_T.append(self.T)
            self.log_V.append(self.V)
            return

        dm_dt = dm / dt

        # Save current state before changes for dT calculation
        m_prev = self.m
        T_prev = self.T

        # Update temperature (isentropic assumption)
        # dT/dt = - (gamma - 1) * T / m * dm/dt
        dT_dt = -dm_dt * T_prev / m_prev * (self.gamma - 1)
        self.T += dT_dt * dt

        # Update mass
        self.m -= dm
        self.m_used += dm

        # Update pressure using ideal gas law (derived from current T and m)
        if self.m > 0:
            self.P = self.m * self.R * self.T / self.V
        else:
            self.P = 0.0
            self.T = 0.0  # If you want to zero T when tank is empty

        self.log_m.append(self.m)
        self.log_P.append(self.P)
        self.log_T.append(self.T)
        self.log_V.append(self.V)

        # print(f"ULLAGE - P: {self.P}  -  T: {self.T}  -  Mass: {self.m}  -  Mass used: {self.m_used}")



    @property
    def getPressure(self) -> float:
        """Returns current gas pressure [Pa]"""
        return self.P

    @property
    def get_mass(self) -> float:
        """Returns current mass in ullage tank"""
        return self.m



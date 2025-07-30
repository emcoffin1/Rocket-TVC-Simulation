import datetime
import math
from pymsis import msis
import numpy as np
class WindProfile:
    def __init__(self):
        pass

    def getWindVelocity(self, alt_m):
        return np.zeros(3)



class GravityModel:
    def __init__(self):
        self.R = 6.471e6    # m -- Radius
        self.g0 = 9.80665   # m/s2 -- standard g SL

    def getGravity(self, alt_m):
        g = self.g0 * (self.R / (self.R + alt_m)) ** 2
        return np.array([0, 0, -g])


class CoriolisModel:
    def __init__(self, latrad=np.radians(35.4275)):
        self.launchLatRag = latrad
        self.omega = 7.2921150e-5
        self.omegaE = np.array([
            0,
            self.omega * math.cos(self.launchLatRag),
            self.omega * math.sin(self.launchLatRag)
        ])

    def getCoriolisEffect(self, vel_m_s):
        """
        Returns acceleration as a vector
        :param vel_m_s:
        :return:
        """
        acc = -2 * np.cross(self.omegaE, vel_m_s)
        return acc


class AirProfile:
    def __init__(self):
        self.densityAir = 1.225  # kg/m3
        self.T0 = 288.15         # K
        self.P0 = 101325         # Pa
        self.L = 0.0065          # K/m
        self.g = 9.80665         # m/s2
        self.R = 287.05          # J/(kg*K)
        self.S = 110.4           # K -- sutherlands const
        self.k = 1.432e-4

        self.gamma = 1.4         # for dry air

        self.maxEffectiveAlt = 70000 # m
        self.referenceALt = 11000

        # Pymsis Information, set for expected launch time
        # march 21 2025, 12Pm
        self.start_time = datetime.datetime(2025, 3, 21, 12, 0, 0)
        self.date_time = [self.start_time]

        self.f107 = 150     # Solar flux index (change based on solar activity)
        self.f107a = 150    # 81 day average
        self.ap = 4         # Geomagnetic index (can be daily or 3 hourly)
        self.aps = [self.ap]*7


        self.rho = None
        self.pres = None
        self.temp = None

    def getCurrentAtmosphere(self, altitudes_m: float, lat = 33.9, lon = -118.4, time: float = 0):
        """
        Called every time step to update atmospheric conditions
        Time may not be updated if the step is a repeat for rk4 integration
        https://swxtrec.github.io/pymsis/examples/plot_altitude_profiles.html#sphx-glr-examples-plot-altitude-profiles-py
        This model is reported to be inaccurate below 1000m, so a standard, simplified model will be used until that point
        """

        if altitudes_m < 1000:
            self.temp = self.T0 - self.L * altitudes_m
            self._update_gamma()
            self.pres = self.P0 * (self.temp / self.T0) ** (self.g / (self.R * self.L))
            self.rho = self.pres / (self.R * self.temp)
            return

        self._update_current_time(time)
        alt_km = altitudes_m / 1000
        output = msis.run([self.date_time], [lat], [lon], [alt_km], [self.f107], [self.f107a], [self.aps])
        data = np.squeeze(output)

        self.rho = round(float(data[0]), 4) if not np.isnan(data[0]) else 0.0
        self.temp = round(float(data[10]), 4) if not np.isnan(data[10]) else self.T0
        self._update_gamma()
        self.pres = round((self.rho * self.R * self.temp),4)

    def _update_current_time(self, time):
        """Updates current time domain for accurate models"""
        self.date_time = [self.start_time + datetime.timedelta(seconds=time)]

    def _update_gamma(self):
        """Accounts for Gamma change over various temperatures using values from:
        https://www.engineeringtoolbox.com/specific-heat-ratio-d_602.html and converted into the equation
        gamma = 1.4e^(-6.02e-5 * temp)
        """
        # Convert K to c
        t = self.temp - 273.15
        gamma = 1.4 * math.exp(-6.02e-5 * t)
        self.gamma = round(gamma, 3)

    def getDensity(self):
        """
        Gets density from NRLMSISE00
        :return: density (kg/m3)
        """
        return self.rho

    def getStaticPressure(self):
        """
        Gets static pressure from NRLMSISE00
        :return: [Pa]
        """
        return self.pres


    def getDynamicPressure(self, velocity_mps: float):
        """
        Uses 1/2 rho v^2 to return dynamic pressure
        :param velocity_mps:
        :return:
        """
        velocity_mps = np.linalg.norm(velocity_mps)
        rho = self.getDensity()
        return 0.5 * rho * velocity_mps ** 2

    def getTemperature(self):
        """
        Returns temperature in K from NRLMSISE00
        """
        return self.temp

    def getSpeedOfSound(self):
        """
        Returns speed of sound using sqrt(YRT)
        :return: [m/s]
        """
        a = math.sqrt(self.gamma * self.R * self.temp)
        return a

    def getMachNumber(self, velocity_mps: float):
        """
        Returns mach number as a function of velocity, temperature, gas constant, temperature
        :param velocity_mps:
        :return: mach number
        """
        a = self.getSpeedOfSound()
        if a > 0:
            return velocity_mps / a
        return 0.0

    def getDynamicViscosity(self):
        """Uses sutherlands law to determine kinematic viscosity of air"""
        mu_ref = 1.716e-5
        t_ref = 273.15
        s_ref = 110.4

        term1 = mu_ref * ((self.temp / t_ref) ** (3 / 2))
        term2 = (t_ref + s_ref) / (self.temp + s_ref)

        return term1 * term2

    def getKinematicViscosity(self, alt_m: float):
        """Uses mu / rho to return kinematic viscosity"""
        mu = self.getDynamicViscosity()
        return mu / self.getDensity()

    def getReynoldsNumber(self, velocity_mps: float, characteristic_length: float):
        rho = self.getDensity()
        mu = self.getDynamicViscosity()
        return (rho * velocity_mps * characteristic_length) / mu if mu > 0 else 0

    def getStagnationPressure(self, velocity_mps: float):
        P = self.getStaticPressure()
        M = self.getMachNumber(velocity_mps=velocity_mps)
        if M < 1e-3:
            return P
        return P * (1 + ((self.gamma - 1) / 2) * M ** 2) ** (self.gamma / (self.gamma - 1))

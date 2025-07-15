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
        """
        self._update_current_time(time)
        print(self.date_time)
        output = msis.run([self.date_time], [lat], [lon], [altitudes_m], [self.f107], [self.f107a], [self.aps])
        data = output.data
        self.rho = float(data[0, 0])
        self.temp = float(data[0, 10])
        self.pres = self.rho * self.R * self.temp

        print(f"RHO: {self.rho}   TEMP: {self.temp}    PRES: {self.pres}")


    def _update_current_time(self, time):
        """Updates current time domain for accurate models"""
        self.date_time = [self.start_time + datetime.timedelta(seconds=time)]

    def getDensity(self, alt_m: float):
        """
        :param alt_m:
        :return density (kg/m3):
        """
        # temp_ratio = 1 - (self.L * alt_m / self.T0)
        # #print(temp_ratio)
        # if temp_ratio <= 0:
        #     return 0.0
        # rho = self.densityAir * temp_ratio ** ((self.g / (self.R * self.L)) + 1)
        # return rho
        return self.rho




    def getStaticPressure(self, alt_m:float):
        """Returns static pressure from NRLMSISE00"""
        # t = self.getTemperature(alt_m)
        #
        # if alt_m <= 11000:
        #     p = 101.29 * (t / 288.08)**5.256
        #
        # elif alt_m <= 25000:
        #     p = 22.65 * math.exp(1.73 - 0.000157*alt_m)
        #
        # elif alt_m > 25000:
        #     p = 2.488 * (t / 216.6)**-11.388
        #
        # return p
        return self.pres


    def getDynamicPressure(self, alt_m: float, velocity_mps: float):
        """
        Uses 1/2 rho v^2 to return dynamic pressure
        :param alt_m:
        :param velocity_mps:
        :return:
        """
        velocity_mps = np.linalg.norm(velocity_mps)
        rho = self.getDensity(alt_m)
        return 0.5 * rho * velocity_mps ** 2

    def getTemperature(self, alt_m: float):
        """
        Returns temperature in K (converted from C)
        """
        # alt_km = alt_m / 1000.0
        # t = None
        # if alt_km <= 11:
        #     # t = 2.3 - (0.154 * alt_km)
        #     t = 14.935 - (alt_km / 0.154)
        #
        # elif alt_km <= 20:
        #     t = -56.46
        #
        # elif alt_km <= 50:
        #     t = -90.0765 + (alt_km / 0.583)
        #
        # elif alt_km <= 56:
        #     t = -5
        #
        # elif alt_km <= 80:
        #     t = 204.968 - (alt_km / 0.2667)
        #
        # elif alt_km <= 90:
        #     t = -95
        #
        # elif alt_km <= 150:
        #     t = -545 + (alt_km / 0.2)
        #
        # elif alt_km <= 500:
        #     # Logarithmic rise from 150 km to ~500 km
        #     # Adjusted base temp (continuity) and scale for realism
        #     t = 500.0 + 250.0 * math.log10((alt_km - 149) + 1)  # Smooth rise
        #
        # else:
        #     # Clamp temp above 500 km (exosphere approximation)
        #     t = 1200.0
        #
        # k = t + 273.15
        #
        # return k
        return self.temp

    def getSpeedOfSound(self, alt_m: float):
        T = self.getTemperature(alt_m)
        print(T)
        a = math.sqrt(self.gamma * self.R * T)

        return a

    def getMachNumber(self, alt_m: float, velocity_mps: float):

        a = self.getSpeedOfSound(alt_m)
        if a > 0:
            return velocity_mps / a
        return 0.0

    def getDynamicViscosity(self, alt_m: float):
        T = self.getTemperature(alt_m)
        mu0 = 1.716e-5
        return mu0 * ((T / self.T0) ** 1.5) * (self.T0 + self.S) / (T + self.S)

    def getKinematicViscosity(self, alt_m: float):
        mu = self.getDynamicViscosity(alt_m)
        rho = self.getDensity(alt_m)
        return mu / rho if rho > 0 else 0

    def getReynoldsNumber(self, alt_m: float, velocity_mps: float, characteristic_length: float):
        rho = self.getDensity(alt_m)
        mu = self.getDynamicViscosity(alt_m)
        return (rho * velocity_mps * characteristic_length) / mu if mu > 0 else 0

    def getStagnationPressure(self, alt_m: float, velocity_mps: float):
        P = self.getStaticPressure(alt_m)
        M = self.getMachNumber(alt_m, velocity_mps)
        if M < 1e-3:
            return P
        return P * (1 + ((self.gamma - 1) / 2) * M ** 2) ** (self.gamma / (self.gamma - 1))

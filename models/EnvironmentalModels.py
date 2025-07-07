import numpy as np
import math

class WindProfile:
    def __init__(self, airmodel: object):
        self.air = airmodel
        pass

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

    def getDensity(self, alt_m:float):
        """
        :param alt_m:
        :return density (kg/m3):
        """
        P = self.getStaticPressure(alt_m)
        layer = self._findLayer(alt_m)
        T0 = layer["T_base"]
        L = layer["L"]
        h0 = layer["h_base"]
        T = T0 + L * (alt_m - h0) if L != 0 else T0
        return P / (self.R * T)

    def getStaticPressure(self, alt_m:float):
        if alt_m < self.maxEffectiveAlt:
            return self.P0 * math.exp(-self.k * alt_m)
        return 0.0

    def getDynamicPressure(self, alt_m: float, velocity_mps: float):
        """
        Uses 1/2 rho v^2 to return dynamic pressure
        :param alt_m:
        :param velocity_mps:
        :return:
        """
        rho = self.getDensity(alt_m)
        return 0.5 * rho * velocity_mps ** 2

    def getTemperature(self, alt_m: float):
        layer = self._findLayer(alt_m)
        T0 = layer["T_base"]
        L = layer["L"]
        h0 = layer["h_base"]
        return T0 + L * (alt_m - h0) if L != 0 else T0

    def getSpeedOfSound(self, alt_m: float):
        T = self.getTemperature(alt_m)
        return math.sqrt(self.gamma * self.R * T)

    def getMachNumber(self, alt_m: float, velocity_mps: float):
        a = self.getSpeedOfSound(alt_m)
        return velocity_mps / a

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

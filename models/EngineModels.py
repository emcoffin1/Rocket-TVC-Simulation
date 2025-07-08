import numpy as np
import math
import matplotlib.pyplot as plt

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


class SimpleEngine:
    def __init__(self):
        self.chamberPressure = 7e6
        self.chamberTemperature = 3500
        self.specificHeatRatio = 1.22
        self.R = 8.314 / 0.022
        self.nozzleThroatArea = 0.01
        self.expansionRatio = 30
        self.P0 = 101325
        self.nozzleExitArea = self.expansionRatio * self.nozzleThroatArea

    def getMassFlowRate(self):
        """
        Uses choked mass flow formula
        :return: [kg/s]
        """
        p1 = self.chamberPressure * self.nozzleThroatArea
        p2 = math.sqrt(self.specificHeatRatio / (self.R * self.chamberTemperature))
        pe = -(self.specificHeatRatio + 1) / (2 * (self.specificHeatRatio - 1))
        p3 = ((self.specificHeatRatio + 1) / 2) ** pe
        return p1 * p2 * p3

    def getExitPressure(self):
        """
        Assumes isentropic expansion to expansion ratio Ae / At
        At = Ae / er
        :return: [Pa]
        """
        return self.chamberPressure * (1 / self.expansionRatio) ** self.specificHeatRatio

    def getThrustCoefficient(self):
        """
        Uses momentum (p1-p3) and pressure correction (p4) to determine thrust coefficient
        :return: [Cf]
        """
        # Momentum Term
        p1 = (2 * self.specificHeatRatio**2) / (self.specificHeatRatio - 1)
        p2 = (2 / (self.specificHeatRatio + 1)) ** ((self.specificHeatRatio + 1) / (self.specificHeatRatio - 1))
        p3 = 1 - (self.getExitPressure() / self.chamberPressure) ** ((self.specificHeatRatio - 1) / self.specificHeatRatio)
        momentum = math.sqrt(p1 * p2 * p3)

        # Pressure Term
        pressure = (self.getExitPressure() - self.getAtmosPress()) / self.chamberPressure * self.expansionRatio
        return momentum + pressure

    def getAtmosPress(self, alt=0):
        """
        Uses P0 * e^(-alt / referenceAlt) to deliver current atmospheric pressure
        :param alt: [m] standard = 0
        :return: [Pa]
        """
        return self.P0 * math.exp(-alt / 8500)



    def getThrust(self):
        """
        Uses F = Cf * Pc * At
        :return: [N]
        """
        cf = self.getThrustCoefficient()
        return cf * self.chamberPressure * self.nozzleThroatArea

    def getIsp(self):
        """
        Uses F / (mdot * g) to delivery specific impulse
        :return: [s]
        """
        thrust = self.getThrust()
        mdot = self.getMassFlowRate()
        g = 9.80665
        return thrust * (mdot / g)




if __name__ == "__main__":
    eng = SimpleEngine()
    print(f"mfr:    {eng.getMassFlowRate()}")
    print(f"Pe:     {eng.getExitPressure()}")
    print(f"Cf:     {eng.getThrustCoefficient()}")
    print(f"Thrust: {eng.getThrust()}")

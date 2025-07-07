import numpy as np
import math
from EnvironmentalModels import AirProfile, GravityModel


def unpackStates(state):
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    mass = state[13]
    return pos, vel, quat, omega, mass


class Rocket:
    def __init__(self):
        # -- Constants -- #
        self.grav = GravityModel()       # m/s2

        # -- Environmental Specific -- #
        #self.air = airmodel

        # -- Vehicle Specific -- #
        self.engine = RocketEngine()
        self.structure = RocketStructure(self.engine.burnTime, self.engine.massFlowRate)
        self.aerodynamics = RocketAerodynamics()
        self.tvc = RocketTVC()

        # -- States -- #
        self.state = np.array([
            0.0, 0.0, 0.0,                      # Position (x, y, z)
            0.0, 0.0, 0.0,                      # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0, 1.0,                 # Quat  (xi, yj, zk, 1)
            0.0, 0.0, 0.0,                      # Angular Velocity
            self.structure.getCurrentMass(0.0)  # Mass
        ])

    def getForces(self, time: float, alt: float):

        # -- Constants -- #
        pos, vel, quat, omega, mass = unpackStates(self.state)
        gravity = self.grav.getGravity(alt)






class RocketStructure:
    def __init__(self, bt_s: float, mfr_kg_s: float):
        self.length = 5.4864    # m -- 18 ft
        self.diameter = 0.254   # m -- 10 in

        self.wetMass = 117.8907629  # kg -- 392.182 lbm
        self.dryMass = 77.5144001   # kg -- 170.890 lbm
        self.liquidMass = self.wetMass - self.dryMass

        self.momInertiaYPInitial = 328.69285873  # kg*m2 -- 7800 lb*ft2
        self.momInertiaYPFinal = 206.48653946    # kg*m2 -- 7800 lb*ft2

        self.cgInitial = 0.28448    # m -- 11.2 ft
        self.cgFinal = 0.29718      # m -- 11.7 ft

        self.burnTime = bt_s
        self.massFlowRate = mfr_kg_s

    def getCurrentMass(self, time: float):
        """
        Gives current mass as a function of mass flow rate and time
        :param time:
        :return current mass:
        """
        if time <= self.burnTime:
            return self.wetMass - self.massFlowRate * time
        else:
            return self.dryMass



    def getCurrentCM(self, time: float):
        """
        Gives current center of mass approximation as it changes over flight
        :param time:
        :return:
        """
        if time <= self.burnTime:
            return self.cgInitial + (self.cgFinal - self.cgInitial) * (time / self.burnTime)
        return self.cgFinal

    def getCurrentPYInertia(self, time: float):
        """
        Gives current moment of inertia of pitch/yaw axis as it changes over flight
        :param time:
        :return:
        """
        if time <= self.burnTime:
            return self.momInertiaYPInitial + (self.momInertiaYPFinal - self.momInertiaYPInitial) * (time / self.burnTime)
        return self.momInertiaYPFinal



class RocketAerodynamics:
    def __init__(self):
        self.dragCoeff = 0.35
        self.liftCoeff = 1.2

        self.CP = 13.38

        self.air = AirProfile()

    def getDragCoeff(self, alt, vel, aoa, burntime):
        """
        Uses poly fit to determine current drag coefficient
        :param alt:
        :param vel:
        :param aoa:
        :return dragCoeff:
        """
        pass

    def getLiftCoeff(self, alt: float, vel: float):
        """
        Uses poly fit to determine current lift coefficient
        :param alt:
        :param vel:
        :param aoa:
        :return liftCoeff:
        """
        pass



class RocketEngine:
    def __init__(self):
        self.isp = 300
        self.massFlowRate = 4.2134195
        self.burnTime = 23.7

        self.chamberExitRatio = 1/8

    def getThrust(self, time: float, pressure: float):
        if time < self.burnTime:

            thrust = np.polyval([0.000127581, -0.0164375, 0.92882, -36.2125, 1998.08], time)
            Pc = self.getChamberPressure(time)
            Pe = Pc / 8
            return thrust * 4.448 + (Pe*6894.76 - pressure) * (5.362**2/4*math.pi/39.37**2) # lbf * N/lbf
        else:
            return 0.0

    def getChamberPressure(self, time: float):
        return np.polyval([5.10325e-06, -0.000657499, 0.0371528, -1.4485, 79.9233], time)

    def getExitPressure(self, time: float):
        return self.getChamberPressure(time) * self.chamberExitRatio

    def getPropUsed(self, time: float):
        return min(self.massFlowRate * time, self.massFlowRate * self.burnTime)

    def getRemainingProp(self, time: float):
        return max(0.0, self.massFlowRate * self.burnTime - self.getPropUsed(time))


class RocketTVC:
    def __init__(self):
        self.maxAngle = 0
        self.maxVectorSpeed = 0
        pass
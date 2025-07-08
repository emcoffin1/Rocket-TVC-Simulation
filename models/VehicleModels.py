import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import EnvironmentalModels

def rk4_step(rocket, state, dt):

    k1 = rocket.getDynamics(state)
    k2 = rocket.getDynamics(state + 0.5 * dt * k1)
    k3 = rocket.getDynamics(state + 0.5 * dt * k2)
    k4 = rocket.getDynamics(state + dt * k3)

    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    q = next_state[6:10]
    q /= np.linalg.norm(q)
    next_state[6:10] = q

    return next_state


def unpackStates(state):
    """
    Unpacks the state variables, so it's easier to manipulate them
    :param state:
    :return: pos, vel, quat, omega, mass, time, aoa, beta, force
    """
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    mass = state[13]
    time = state[14]
    aoa = state[15]
    beta = state[16]

    return pos, vel, quat, omega, mass, time, aoa, beta


def quaternionDerivative(quat, omega):
    """
    Computes the quat derivative using:
    0.5 * quat (quat_product) omega
    :param quat:
    :param omega:
    :return:
    """
    qx, qy, qz, qw = quat
    wx, wy, wz = omega

    dqdt = 0.5 * np.array([
        qw * wx + qy * wz - qz * wy,    # dqx/dt
        qw * wy + qz * wx - qx * wz,    # dqy/dt
        qw * wz + qx * wy - qy * wx,    # dqz/dt
        -qx * wx - qy * wy - qz * wz    # dqw/dt
    ])
    return dqdt


class Rocket:
    def __init__(self):
        # -- Constants -- #
        self.grav = EnvironmentalModels.GravityModel()       # m/s2
        self.cor = EnvironmentalModels.CoriolisModel()
        self.air = EnvironmentalModels.AirProfile()
        self.wind = EnvironmentalModels.WindProfile()

        # -- Vehicle Specific -- #
        self.engine = RocketEngine()
        self.structure = RocketStructure(self.engine.burnTime, self.engine.massFlowRate)
        self.aerodynamics = RocketAerodynamics(self.air, self.wind)
        self.tvc = RocketTVC()

        # -- States -- #
        self.state = np.array([
            0.0, 0.0, 0.0,                       # Position (x, y, z)
            0.0, 0.0, 0.0,                       # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0, 1.0,                  # Quat  (xi, yj, zk, 1)
            0.0, 0.0, 0.0,                       # Angular Velocity
            self.structure.getCurrentMass(0.0),  # Mass
            0.0,                                 # Time
            0.0, 0.0,                            # AOA, BETA in reference to wind

        ])

    def getDynamics(self, state):

        a,b,c,d,_,__ = self.getTotalForce(state)
        force = a + b + c + d
        pos, vel, quat, omega, mass, time, aoa, beta = unpackStates(state)
        # Sum of all accelerations --- a = F/m
        acceleration = force / mass

        # print(np.linalg.norm(force))

        # -- Quaternion Derivative -- #
        dqdt = quaternionDerivative(quat, omega)

        # -- Angular Acceleration -- #
        # Currently zero
        domega = np.zeros(3)

        # -- Mass Change -- #
        dmdt = -self.engine.getMassFlowRate(time)

        # -- State Derivative -- #
        # The change in state variables
        dstate = np.concatenate([
            vel,
            acceleration,
            dqdt,
            domega,
            [dmdt],
            [1.0],
            [0.0],
            [0.0]
        ])

        return dstate

    def getTotalForce(self, state):
        # -- Constants -- #
        pos, vel, quat, omega, mass, time, aoa, beta = unpackStates(state)
        alt_m = pos[2]
        rho = self.air.getDensity(alt_m=alt_m)
        # print(rho)
        stat_pres = self.air.getStaticPressure(alt_m=alt_m)

        # -- Velocity -- #
        v_air = vel - self.wind.getWindVelocity(alt_m=pos[2])
        r_world_to_body = R.from_quat(quat).inv()
        v_air_body = r_world_to_body.apply(v_air)

        aoa = np.arctan2(v_air_body[1], v_air_body[2])
        beta = np.arctan2(v_air_body[0], v_air_body[2])

        # -- Accelerations -- #
        gravity = self.grav.getGravity(alt_m=alt_m)
        coriolis_acc = self.cor.getCoriolisEffect(vel_m_s=vel)

        # -- Forces -- #
        # Thrust
        thrust_force_body = self.engine.getThrust(time=time, pressure=stat_pres)
        thrust_force_global = R.from_quat(quat).apply(thrust_force_body)

        # Drag
        drag_force = self.aerodynamics.getDragForce(vel_m_s=v_air, rho=rho, cross_area=self.structure.xyPlaneArea,
                                                    pos=pos, aoa=aoa, beta=beta, burntime=self.engine.burnTime)
        #print(np.linalg.norm(drag_force))

        # -- SUMS -- #
        # Sum of forces -- must be a vector
        # force = (thrust_force_global
        #         + drag_force
        #         + (gravity * mass)
        #         + (coriolis_acc * mass)
        #         )

        # Dynamic Pressure
        q = self.air.getDynamicPressure(alt_m=alt_m, velocity_mps=vel)

        # print(drag_force)
        return thrust_force_global, drag_force, (gravity*mass), (coriolis_acc * mass), rho, q





class RocketStructure:
    def __init__(self, bt_s: float, mfr_kg_s: float):
        self.length = 5.4864    # m -- 18 ft
        self.diameter = 0.254   # m -- 10 in
        self.zxPlaneArea = self.length * self.diameter
        self.xyPlaneArea = math.pi / 4 * self.diameter**2

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
    def __init__(self, airprofile: EnvironmentalModels.AirProfile, windprofile: object):
        self.dragCoeff = 0.35
        self.liftCoeff = 1.2

        self.CP = 13.38

        self.air = airprofile
        self.wind = windprofile

    def getDragCoeff(self, alt: float, mach, aoa: float, beta: float, burntime: float):
        """
        Uses poly fit to determine current drag coefficient
        :param alt:
        :param mach:
        :param aoa: rad
        :param beta:
        :param burntime:
        :return dragCoeff:
        """
        cd = 0.3 + 0.5 * np.sin(aoa)
        return cd

    def getLiftCoeff(self, alt: float, vel):
        """
        Uses poly fit to determine current lift coefficient
        :param alt:
        :param vel:
        :param aoa:
        :return liftCoeff:
        """
        return 1.2

    def getDragForce(self, vel_m_s, pos, rho, cross_area, burntime, aoa, beta):
        """
        Returns the drag force experienced on the entire vehicle
        Uses: 1/2 rho v^2 A Cl
        :param vel_m_s:
        :param pos:
        :param rho:
        :param cross_area:
        :param burntime:
        :param aoa:
        :param beta:
        :return: [x,y,z] in inertial/global frame
        """
        v_air = vel_m_s
        v_mag = np.linalg.norm(v_air)

        if v_mag == 0:
            return np.zeros(3)

        # Direction opposite to motion
        drag_direction = -v_air / v_mag  # explicitly say "opposite"
        mach = self.air.getMachNumber(pos[2], v_mag)
        cd = self.getDragCoeff(pos[2], mach, aoa, beta, burntime)
        drag_magnitude = 0.5 * rho * v_mag ** 2 * cd * cross_area



        drag = drag_magnitude * drag_direction
        return drag


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
            T = thrust * 4.448 + (Pe*6894.76 - pressure) * (5.362**2/4*math.pi/39.37**2) # lbf * N/lbf
            return np.array([0, 0, T])
        else:
            return np.zeros(3)

    def getMassFlowRate(self, time: float):
        if time <= self.burnTime:
            return self.massFlowRate
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
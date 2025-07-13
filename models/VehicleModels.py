import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import EnvironmentalModels
from models.Engine.EngineModels import *
from models.Engine.LiquidModels import *

def rk4_step(rocket, state, dt):

    k1 = rocket.getDynamics(state, dt, side_effect=False)
    k2 = rocket.getDynamics(state + 0.5 * dt * k1, dt, side_effect=False)
    k3 = rocket.getDynamics(state + 0.5 * dt * k2, dt, side_effect=False)
    k4 = rocket.getDynamics(state + dt * k3, dt, side_effect=True)

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
        self.engine = Engine()
        self.structure = RocketStructure(engine=self.engine)
        self.aerodynamics = RocketAerodynamics(self.air, self.wind)
        self.tvc = RocketTVC()

        # -- States -- #
        self.state = np.array([
            0.0, 0.0, 0.0,                       # Position (x, y, z)
            0.0, 0.0, 0.0,                       # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0, 1.0,                  # Quat  (xi, yj, zk, 1)
            0.0, 0.0, 0.0,                       # Angular Velocity
            self.structure.wetMass,              # Mass
            0.0,                                 # Time
            0.0, 0.0,                            # AOA, BETA in reference to wind

        ])

        self.thrust = []
        self.burntime = None

        self._initialize_vehicle()

    def getDynamics(self, state, dt: float, side_effect: bool = True):

        a, b, c, d, _, __, static_pres = self.getTotalForce(state=state, dt=dt, side_effect=side_effect)
        force = a + b + c + d
        # print(f"Drag: {np.linalg.norm(b):.3f}")
        pos, vel, quat, omega, mass, time, aoa, beta = unpackStates(state=state)
        # Sum of all accelerations --- a = F/m
        acceleration = force / mass

        # print(np.linalg.norm(force))

        # -- Quaternion Derivative -- #
        dqdt = quaternionDerivative(quat, omega)

        # -- Angular Acceleration -- #
        # Currently zero
        domega = np.zeros(3)

        # -- Mass Change -- #
        dmdt = -self.structure.getMassChange()

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

    def getTotalForce(self, state, dt: float, side_effect: bool = True):
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
        thrust_mag = self.engine.runBurn(dt=dt, alt_m=alt_m, side_effect=side_effect)
        if thrust_mag == 0 and not self.burntime:
            self.burntime = time
        if side_effect and thrust_mag != 0:
            self.thrust.append(thrust_mag)
        # print(thrust_mag)
        # print(thrust_mag)
        # Thrust_force_body needs to be rotated depending on the TVC angle
        thrust_force_body = np.array([0.0, 0.0, thrust_mag])
        thrust_force_global = R.from_quat(quat).apply(thrust_force_body)

        # Drag
        drag_force = self.aerodynamics.getDragForce(vel_m_s=v_air, rho=rho, cross_area=self.structure.xyPlaneArea,
                                                    pos=pos, aoa=aoa, beta=beta)
        # print(np.linalg.norm(drag_force))

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
        return thrust_force_global, drag_force, (gravity*mass), (coriolis_acc * mass), rho, q, stat_pres

    def _initialize_vehicle(self):
        """A function to immediately display vehicle information on launch"""
        print("=" * 60)
        print("INITIALIZING VEHICLE")
        print(f"Initial Vehicle Mass:   {self.structure.wetMass} [kg]")
        print(f"Initial Fluid Mass:     {self.structure.liquidMass} [kg]")
        print("=" * 60)


class RocketStructure:
    def __init__(self, engine: Engine):
        self.engine = engine

        self.length = 5.4864    # m -- 18 ft
        self.diameter = 0.254   # m -- 10 in
        self.zxPlaneArea = self.length * self.diameter
        self.xyPlaneArea = math.pi / 4 * self.diameter**2

        # self.wetMass = 117.8907629  # kg -- 392.182 lbm
        self.dryMass = 77.5144001   # kg -- 170.890 lbm
        self.liquidMass = self.engine.getFluidMass()
        self.wetMass = self.dryMass + self.liquidMass

        self.current_mass = self.wetMass


        self.momInertiaYPInitial = 328.69285873  # kg*m2 -- 7800 lb*ft2
        self.momInertiaYPFinal = 206.48653946    # kg*m2 -- 7800 lb*ft2

        self.cgInitial = 0.28448    # m -- 11.2 ft
        self.cgFinal = 0.29718      # m -- 11.7 ft

    def getMassChange(self):
        """
        Gives mass change as a function of dry mass and remaining fluid mass.
        :return: mass change
        """
        if self.engine.combustion_chamber.active:
            # Get mass after engine update
            val = self.dryMass + self.engine.getFluidMass()
            # Subtract current mass from new mass
            dm = self.current_mass - val
            # Set new mass as current mass
            self.current_mass = val
            return dm
        else:
            # No method to change mass, dm = 0
            return 0

    def getCurrentCM(self):
        """
        Gives current center of mass approximation as it changes over flight
        Assumed to be uniform shift and is therefor extracted linearly
        :return:
        """
        if self.engine.combustion_chamber.active:
            val = (self.cgInitial + ((self.current_mass - self.wetMass) / (self.dryMass - self.wetMass)) *
                    (self.cgFinal - self.cgInitial))
            # print(val)
            return  val
        return self.cgFinal

    def getCurrentPYInertia(self):
        """
        Gives current moment of inertia of pitch/yaw axis as it changes over flight
        :return:
        """
        if self.engine.combustion_chamber.active:
            return (self.momInertiaYPInitial + ((self.current_mass - self.wetMass) / (self.dryMass - self.wetMass))
                    * (self.momInertiaYPFinal - self.momInertiaYPInitial))
        return self.momInertiaYPFinal






class RocketAerodynamics:
    def __init__(self, airprofile: EnvironmentalModels.AirProfile, windprofile: object):
        self.dragCoeff = 0.35
        self.liftCoeff = 1.2

        self.CP = 4.20624   # m -- 13.38 ft

        self.air = airprofile
        self.wind = windprofile

        self.cd_M_points = np.array([
            0.0166667, 0.125, 0.2583333, 0.3833333, 0.5, 0.6, 0.6833333, 0.775,
            0.85, 0.9333333, 1.0, 1.1083333, 1.1916667, 1.2833333, 1.3333333,
            1.4333333, 1.525, 1.6, 1.6833333, 1.775, 1.8666667, 1.9583333,
            2.0666667, 2.1916667, 2.2916667, 2.4083333, 2.55, 2.7, 2.875,
            3.0666667, 3.3083333, 3.5416667, 3.8, 4.1, 4.525, 5.0166667,
            5.575, 6.1083333, 6.5833333, 7.0166667, 7.5666667, 8.0583333,
            8.6, 9.1083333, 9.775, 10.0416667
        ])

        self.cd_points = np.array([
            0.3000, 0.2830, 0.2696, 0.2643, 0.2607, 0.2688, 0.2804, 0.3009,
            0.3295, 0.3643, 0.4027, 0.4536, 0.4955, 0.5313, 0.5482, 0.5536,
            0.5509, 0.5429, 0.5330, 0.5205, 0.5063, 0.4848, 0.4616, 0.4321,
            0.4054, 0.3813, 0.3536, 0.3268, 0.3009, 0.2750, 0.2500, 0.2321,
            0.2179, 0.2080, 0.2018, 0.1991, 0.2036, 0.2116, 0.2205, 0.2304,
            0.2420, 0.2518, 0.2598, 0.2616, 0.2616, 0.2616
        ])

    def getDragCoeff(self, alt: float, mach, aoa: float, beta: float):
        """
        Uses poly fit to determine current drag coefficient
        :param alt:
        :param mach:
        :param aoa: rad
        :param beta:
        :return dragCoeff:
        """

        Mach_clamped = np.clip(mach, self.cd_M_points[0], self.cd_M_points[-1])
        val = float(np.interp(Mach_clamped, self.cd_M_points, self.cd_points))
        # print(val)
        return val
        # cd = 0.3 + 0.5 * np.sin(aoa)
        # return cd

    def getLiftCoeff(self, alt: float, vel):
        """
        Uses poly fit to determine current lift coefficient
        :param alt:
        :param vel:
        :return liftCoeff:
        """
        return 1.2

    def getDragForce(self, vel_m_s, pos, rho, cross_area, aoa, beta):
        """
        Returns the drag force experienced on the entire vehicle
        Uses: 1/2 rho v^2 A Cl
        :param vel_m_s:
        :param pos:
        :param rho:
        :param cross_area:
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
        # cd = self.getDragCoeff(pos[2], mach, aoa, beta)
        cd = 0.25
        drag_magnitude = 0.5 * rho * v_mag ** 2 * cd * cross_area

        drag = drag_magnitude * drag_direction
        return drag



class RocketTVC:
    def __init__(self):
        self.maxAngle = 0
        self.maxVectorSpeed = 0
        pass
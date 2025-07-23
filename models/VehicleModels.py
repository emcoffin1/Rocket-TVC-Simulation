import numpy as np
from scipy.spatial.transform import Rotation as R
from models.Engine.EngineModels import *
from models.Structural.StructuralModel import StructuralModel
from models.TVC.TVCModel import *
from models.TVC.LqrQuaternionModels import LQR, QuaternionFinder
from models.Aero.AerodynamicsModel import *
from models import EnvironmentalModels

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

        q = np.eye(3) * 5
        r = np.eye(3) * 1
        # q = np.eye(3) * 1.0
        # r = np.eye(3) * 100.0

        # -- Constants -- #
        self.grav = EnvironmentalModels.GravityModel()       # m/s2
        self.cor = EnvironmentalModels.CoriolisModel()
        self.air = EnvironmentalModels.AirProfile()
        self.wind = EnvironmentalModels.WindProfile()
        self.lqr = LQR(q=q, r=r)

        # -- Init Atmosphere -- #
        self.air.getCurrentAtmosphere(altitudes_m=0, time=0)

        # -- Vehicle Specific -- #
        self.engine = Engine()
        self.structure = StructuralModel(engine_class=self.engine, liquid_total_ratio=0.56425)
        self.aerodynamics = Aerodynamics(self.air)
        self.tvc = TVCStructure(lqr=self.lqr, structure_model=self.structure)

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

        # Logging
        self.thrust = []                         # Thrust curve
        self.burntime = None                     # Time burn stops
        self.mach = []
        self.velocity = []
        self.reynolds = []
        self.viscosity = []
        self.mfr = []
        self.pc = []

        # Init print function
        self._initialize_vehicle()

    def getDynamics(self, state, dt: float, side_effect: bool = True):
        a, b, c, d, _, __, static_pres, domega = self.getTotalForce(state=state, dt=dt, side_effect=side_effect)
        force = a + b + c + d
        pos, vel, quat, omega, mass, time, aoa, beta = unpackStates(state=state)

        # Update altitude and time for atmospheric model IF side_effect is enabled
        if side_effect:
            #print(pos[2], time, force, mass)
            self.air.getCurrentAtmosphere(altitudes_m=pos[2], time=time)
            self.structure.updateProperties()


        # Sum of all accelerations --- a = F/m
        acceleration = force / mass

        # -- Quaternion Derivative -- #
        dqdt = quaternionDerivative(quat, omega)
        # print(quat, "|||",omega, "|||", dqdt)
        # -- Angular Acceleration -- #
        # handled directly in forces

        # -- Mass Change -- #
        dmdt = self.structure.dm

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

        rho = self.air.getDensity()
        stat_pres = self.air.getStaticPressure()
        reynolds = self.air.getReynoldsNumber(np.linalg.norm(vel),characteristic_length=self.structure.diameter)
        viscosity = self.air.getDynamicViscosity()

        if side_effect:
            m = round(self.air.getMachNumber(velocity_mps=np.linalg.norm(vel)), 2)
            self.mach.append(m)
            a,b = self.engine.combustion_chamber.getMassFlowRate()
            self.mfr.append(a+b)
            self.pc.append(self.engine.combustion_chamber.Pc)

            self.reynolds.append(reynolds)
            self.viscosity.append(viscosity)
            self.velocity.append(np.linalg.norm(vel))
            # print(round(time,2))

        # -- Velocity -- #
        v_air = vel - self.wind.getWindVelocity(alt_m=pos[2])
        r_world_to_body = R.from_quat(quat).inv()
        v_air_body = r_world_to_body.apply(v_air)

        aoa = np.arctan2(v_air_body[1], v_air_body[2])
        beta = np.arctan2(v_air_body[0], v_air_body[2])

        # -- Accelerations -- #
        gravity = self.grav.getGravity(alt_m=alt_m)
        coriolis_acc = self.cor.getCoriolisEffect(vel_m_s=v_air_body)

        # -- Forces -- #
        # Thrust
        thrust_mag = self.engine.runBurn(dt=dt, alt_m=alt_m, side_effect=side_effect)
        self.tvc.update_variables_(thrust_mag)
        # Pitch (around x) Yaw (around y)
        theta_x, theta_y, w = self.tvc.calculate_theta(dt=dt, rocket_location=pos, rocket_quat=quat)

        # if side_effect:
        #     print(f"{time:.2f}, {pos[2]:.3f}, {np.rad2deg(theta_x):.2f}, {np.rad2deg(theta_y):.2f}")

        limit = np.deg2rad(25)
        theta_x = np.clip(theta_x, -limit, limit)
        theta_y = np.clip(theta_y, -limit, limit)

        # theta_y = 0
        # print(w, theta_x, theta_y)

        L = self.structure.length
        torque_body = np.array([
            L * thrust_mag * np.sin(theta_y),
            -L * thrust_mag * np.sin(theta_x),
            0.0
        ])

        domega = torque_body / self.structure.I
        domega = np.clip(domega, -100.0, 100.0)
        # domega = np.zeros(3)

        # r_tvc = R.from_euler('yx', [-theta_y, -theta_x])
        r_x = R.from_euler('x', -theta_x)
        r_y = R.from_euler('y', -theta_y)
        r_tvc = r_x * r_y  # Apply X rotation, then Y rotation

        if thrust_mag == 0 and not self.burntime:
            self.burntime = time
            # exit()
        if side_effect and thrust_mag != 0:
            self.thrust.append(thrust_mag)



        # Thrust_force_body needs to be rotated depending on the TVC angle
        thrust_force_body = r_tvc.apply([0.0, 0.0, thrust_mag])
        r_body_to_world = R.from_quat(quat)
        thrust_force_global = r_body_to_world.apply(thrust_force_body)

        # if side_effect:
        #     # print(round(time,2), thrust_mag, np.round(w,2),np.round(np.rad2deg(theta_x),2),np.round(np.rad2deg(theta_y),2))
        #     print(round(time,2), thrust_mag, np.round(w,2),np.round((theta_x),2),np.round((theta_y),2))

        # Drag

        drag_force = self.aerodynamics.getDragForce(vel=v_air_body)


        # Dynamic Pressure
        q = self.air.getDynamicPressure(velocity_mps=vel)

        # print(f"[TVC] θx = {np.rad2deg(theta_x):.2f}°, θy = {np.rad2deg(theta_y):.2f}°")
        # print(f"[TVC] Torque = {torque_body}")
        # print(f"[TVC] domega = {domega}")

        return thrust_force_global, drag_force, (gravity*mass), (coriolis_acc * mass), rho, q, stat_pres, domega

    def _initialize_vehicle(self):
        """A function to immediately display vehicle information on launch/initialization"""
        print("=" * 60)
        print("INITIALIZING VEHICLE")
        print(f"Initial Vehicle Mass:   {self.structure.wetMass} [kg]")
        # print(f"Initial Fluid Mass:     {self.structure.liquidMass} [kg]")
        print(f"Initial Fluid Mass:     {self.structure.fluid_mass} [kg]")
        print("=" * 60)




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

    def getDragCoeff(self, mach, aoa: float = 0):
        """
        Uses poly fit to determine current drag coefficient
        :param alt:
        :param mach:
        :param aoa: rad
        :param beta:
        :return dragCoeff:
        """

        # Mach_clamped = np.clip(mach, self.cd_M_points[0], self.cd_M_points[-1])
        # val = float(np.interp(Mach_clamped, self.cd_M_points, self.cd_points))
        x = mach
        val = None
        if x <= 0.8:
            val = 0.103 + -0.0218*x + 0.0174 + x**2
        elif x <= 1.10:
            val = -0.859 + 1.76*x + -0.775*x**2
        elif x <= 2.0:
            val = -0.037*x + 0.183
        else:
            val = 0.097
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
        mach = self.air.getMachNumber(v_mag)
        cd = self.getDragCoeff(mach=mach)
        drag_magnitude = 0.5 * rho * v_mag ** 2 * cd * cross_area

        drag = drag_magnitude * drag_direction
        return drag


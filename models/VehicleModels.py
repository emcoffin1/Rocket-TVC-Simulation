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

        q = np.eye(3) * 2
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
            # print(round(time,2), np.round(quat,2))

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


        theta_x, theta_y, w = self.tvc.calculate_theta(dt=dt, rocket_location=pos, rocket_quat=quat, side_effect=side_effect)

        if not self.engine.combustion_chamber.active:
            theta_x, theta_y = 0, 0


        limit = np.deg2rad(2.5)
        theta_x = np.clip(theta_x, -limit, limit)
        theta_y = np.clip(theta_y, -limit, limit)


        L = self.structure.length
        torque_body = np.array([
            -L * thrust_mag * theta_x,
            -L * thrust_mag * theta_y,
            0.0
        ])

        domega = torque_body / self.structure.I
        # domega = np.zeros(3)

        # r_tvc = R.from_euler('yx', [-theta_y, -theta_x])
        # theta_x = 0
        # theta_y = 0
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


        # Drag

        drag_force = self.aerodynamics.getDragForce(vel=v_air_body)


        # Dynamic Pressure
        q = self.air.getDynamicPressure(velocity_mps=vel)


        return thrust_force_global, drag_force, (gravity*mass), (coriolis_acc * mass), rho, q, stat_pres, domega

    def _initialize_vehicle(self):
        """A function to immediately display vehicle information on launch/initialization"""
        print("=" * 60)
        print("INITIALIZING VEHICLE")
        print(f"Initial Vehicle Mass:   {self.structure.wetMass} [kg]")
        # print(f"Initial Fluid Mass:     {self.structure.liquidMass} [kg]")
        print(f"Initial Fluid Mass:     {self.structure.fluid_mass} [kg]")
        print("=" * 60)



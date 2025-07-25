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

    # Normalize the two quaternions in the state variables
    q = next_state[6:10]
    q /= np.linalg.norm(q)

    g_q = next_state[13:17]
    g_q /= np.linalg.norm(g_q)

    # Update those values
    next_state[6:10] = q
    next_state[13:17] = g_q

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
    tvc_quat = state[13:17]
    mass = state[17]
    time = state[18]
    aoa = state[19]
    beta = state[20]
    dtheta = state[21:23]

    return pos, vel, quat, omega, tvc_quat, mass, time, aoa, beta, dtheta


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

        q = np.eye(3) * 100
        r = np.eye(3) * 0.5

        # -- Constants -- #
        self.grav = EnvironmentalModels.GravityModel()
        self.cor = EnvironmentalModels.CoriolisModel()
        self.air = EnvironmentalModels.AirProfile()
        self.wind = EnvironmentalModels.WindProfile()       # Currently not implemented (monte-carlo style sims)
        self.lqr = LQR(q=q, r=r)

        # -- Init Atmosphere -- #
        self.air.getCurrentAtmosphere(altitudes_m=0, time=0)

        # -- Vehicle Specific -- #
        self.engine = Engine()
        self.structure = StructuralModel(engine_class=self.engine, liquid_total_ratio=0.56425)
        self.aerodynamics = Aerodynamics(self.air)
        self.tvc = TVCStructure(
            lqr_controller=self.lqr,
            structure_model=self.structure,
            aero_model=self.aerodynamics)

        # -- States -- #
        self.state = np.array([
            0.0, 0.0, 0.0,                       # Position (x, y, z)
            0.0, 0.0, 0.0,                       # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0, 1.0,                  # Quat  (xi, yj, zk, 1)
            0.0, 0.0, 0.0,                       # Angular Velocity
            0.0, 0.0, 0.0, 1.0,                  # Current gimbal orientation
            self.structure.wetMass,              # Mass
            0.0,                                 # Time
            0.0, 0.0,                            # AOA, BETA in reference to wind
            0.0, 0.0,                            # Gimbal angles
        ])

        # -- Logging -- #
        self.thrust     = []                     # Thrust curve
        self.burntime   = None                   # Time burn stops
        self.mach       = []                     # Mach list
        self.velocity   = []                     # Velocity list
        self.reynolds   = []                     # Reynolds list
        self.viscosity  = []
        self.mfr        = []
        self.pc         = []

        # Init print function
        self._initialize_vehicle()

    def getDynamics(self, state, dt: float, side_effect: bool = True):

        # Unpack current states
        pos, vel, quat, omega, tvc_quat, mass, time, aoa, beta, dtheta = unpackStates(state=state)

        # ======================== #
        # -- UPDATE ENVIRONMENT -- #
        # ======================== #

        if side_effect:
            self.air.getCurrentAtmosphere(altitudes_m=pos[2], time=time)
            self.structure.updateProperties()

        # ============ #
        # -- FORCES -- #
        # ============ #

        thrust, drag, gravity, coriolis, rho, q_dyn, stat_pres, domega = self.getTotalForce(state=state, dt=dt,
                                                                                            side_effect=side_effect)
        force_total = thrust + drag + gravity + coriolis

        # ============================ #
        # -- TRANSLATION KINEMATICS -- #
        # ============================ #

        # Acceleration using a = F/m
        acceleration = force_total / mass

        # Position derivative = velocity
        dpos = vel

        # Velocity derivative = acceleration
        dvel = acceleration

        # =========================== #
        # -- QUATERNION KINEMATICS -- #
        # =========================== #

        # Change in quaternion
        dqdt = quaternionDerivative(quat, omega)

        # ================= #
        # -- Mass Change -- #
        # ================= #

        dmdt = self.structure.dm

        # ========================= #
        # -- GIMBAL ANGLE CHANGE -- #
        # ========================= #

        dtheta_x = self.tvc.d_theta_x
        dtheta_y = self.tvc.d_theta_y
        dgimbal_q = quaternionDerivative(self.tvc.gimbal_orientation.as_quat(), np.array([dtheta_x, dtheta_y, 0.0]))

        # ====================== #
        # -- State Derivative -- #
        # ====================== #

        # The change in state variables
        dstate = np.concatenate([
            vel,
            acceleration,
            dqdt,
            domega,
            dgimbal_q,
            [dmdt],
            [1.0],
            [0.0],
            [0.0],
            [dtheta_x, dtheta_y]
        ])

        return dstate

    def getTotalForce(self, state, dt: float, side_effect: bool = True):
        # =============== #
        # -- Constants -- #
        # =============== #
        pos, vel, quat, omega, tvc_quat, mass, time, aoa, beta, dtheta = unpackStates(state)
        alt_m = pos[2]

        rho = self.air.getDensity()
        stat_pres = self.air.getStaticPressure()
        reynolds = self.air.getReynoldsNumber(np.linalg.norm(vel),characteristic_length=self.structure.diameter)
        viscosity = self.air.getDynamicViscosity()

        # =================== #
        # -- UPDATE THINGS -- #
        # =================== #

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

        # ============== #
        # -- Velocity -- #
        # ============== #

        # Velocity in world frame
        # velocity rocket - wind velocities
        v_air = vel - self.wind.getWindVelocity(alt_m=pos[2])

        # Determine rocket orientation on inertial frame
        r_world_to_body = R.from_quat(quat).inv()

        # Determine velocity of air on body (for drag calculations)
        v_air_body = r_world_to_body.apply(v_air)

        # Determine angle of vehicle for drag calculations (not currently implemented)
        aoa = np.arctan2(v_air_body[1], v_air_body[2])
        beta = np.arctan2(v_air_body[0], v_air_body[2])

        # =================== #
        # -- ACCELERATIONS -- #
        # =================== #

        gravity = self.grav.getGravity(alt_m=alt_m)
        coriolis_acc = self.cor.getCoriolisEffect(vel_m_s=v_air_body)

        # =================== #
        # -- THRUST FORCES -- #
        # =================== #

        # Get raw thrust (no direction)
        thrust_mag = self.engine.runBurn(dt=dt, alt_m=alt_m, side_effect=side_effect)

        # Save burntime for later display
        if thrust_mag == 0 and not self.burntime:
            self.burntime = time
            # exit()

        # Update tvc thrust values for easier computing
        # !pulls data from other objects, keep as function!
        self.tvc.update_variables_(thrust_mag)

        # Get updated gimbal orientation in body frame
        gimbal_orient: R = self.tvc.compute_gimbal_orientation(
            time= time,
            dt=   dt,
            rocket_pos=   pos,
            rocket_quat=  quat,
            rocket_vel=   vel,
            rocket_acc=   self._compute_acceleration(),  # however you get accel
            side_effect= side_effect
        )

        # Apply gimbal rotation to thrust vector in body frame
        thrust_vector_body = gimbal_orient.apply([0.0, 0.0, thrust_mag])  # original thrust vector straight out

        # Rotate thrust vecotr to inertial frame
        r_body_to_world = R.from_quat(quat)     # Inertial frame rotation
        thrust_force_global = r_body_to_world.apply(thrust_vector_body)     # Application of rotation

        if side_effect and thrust_mag != 0:
            self.thrust.append(thrust_force_global)

        # ================================ #
        # -- CHANGE IN ANGULAR VELOCITY -- #
        # ================================ #

        # Length of vehicle
        l_veh = self.structure.length

        # Find torque using T = L*T*theta
        torque_body = np.array([
            -l_veh * thrust_mag * np.sin(self.tvc.theta_x_prev),
            -l_veh * thrust_mag * np.sin(self.tvc.theta_y_prev),
            0.0
        ])

        # Find change in angular velocity (alpha) using T/I ~~ Torque = inertia * angular accleration
        domega = torque_body / self.structure.I

        # ================= #
        # -- DRAG FORCES -- #
        # ================= #

        drag_force = self.aerodynamics.getDragForce(vel=v_air_body)

        # ====================== #
        # -- DYNAMIC PRESSURE -- #
        # ====================== #
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



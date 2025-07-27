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

    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

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
        qw * wx + qy * wz - qz * wy,  # dqx/dt
        qw * wy + qz * wx - qx * wz,  # dqy/dt
        qw * wz + qx * wy - qy * wx,  # dqz/dt
        -qx * wx - qy * wy - qz * wz  # dqw/dt
    ])
    return dqdt


class Rocket:
    def __init__(self):

        q = np.diag([5, 5, 5, 1, 1, 1])
        r = np.eye(3) * 0.1

        # -- Constants -- #
        self.grav = EnvironmentalModels.GravityModel()
        self.cor = EnvironmentalModels.CoriolisModel()
        self.air = EnvironmentalModels.AirProfile()
        self.wind = EnvironmentalModels.WindProfile()  # Currently not implemented (monte-carlo style sims)

        # -- Init Atmosphere -- #
        self.air.getCurrentAtmosphere(altitudes_m=0, time=0)

        # -- Vehicle Specific -- #
        self.engine = Engine()
        self.structure = StructuralModel(engine_class=self.engine, liquid_total_ratio=0.56425)
        self.aerodynamics = Aerodynamics(self.air)
        self.tvc = TVCStructure(
            structure_model=self.structure,
            aero_model=self.aerodynamics)

        # -- TABS -- #
        z_comp = self.structure.cm_current - self.structure.length
        r_comp = 0.2032
        self.pos_x_tab = FinTab(rocket_position=np.array([r_comp, 0, z_comp]))
        self.neg_x_tab = FinTab(rocket_position=np.array([-r_comp, 0, z_comp]))
        self.pos_y_tab = FinTab(rocket_position=np.array([0, r_comp, z_comp]))
        self.neg_y_tab = FinTab(rocket_position=np.array([0, -r_comp, z_comp]))
        self.fins = [self.pos_x_tab, self.neg_x_tab, self.pos_y_tab, self.neg_y_tab]
        self.roll_control = RollControl(fins=self.fins, struct=self.structure)

        # -- CONTROL -- #
        self.lqr = LQR(q=q, r=r)
        self.quaternion = QuaternionFinder(lqr=self.lqr)

        # -- States -- #
        self.state = np.array([
            0.0, 0.0, 0.0,  # Position (x, y, z)
            0.0, 0.0, 0.0,  # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0, 1.0,  # Quat  (xi, yj, zk, 1)
            0.0, 0.0, 0.0,  # Angular Velocity
            0.0, 0.0, 0.0, 1.0,  # Current gimbal orientation
            self.structure.wetMass,  # Mass
            0.0,  # Time
            0.0, 0.0,  # AOA, BETA in reference to wind
            0.0, 0.0,  # Gimbal angles
        ])

        # -- Logging -- #
        self.acc = 0
        self.thrust = []  # Thrust curve
        self.burntime = None  # Time burn stops
        self.mach = []  # Mach list
        self.velocity = []  # Velocity list
        self.reynolds = []  # Reynolds list
        self.viscosity = []
        self.dynamic_pres = []

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

        thrust_vector_body, drag_force_body, coriolis_force_body, lift_force_body, total_force_global = self.getTotalForce(
            state=state, dt=dt, side_effect=side_effect)

        # ============================ #
        # -- TRANSLATION KINEMATICS -- #
        # ============================ #

        # Acceleration using a = F/m
        acceleration = total_force_global / mass

        # Position derivative = velocity
        dpos = vel

        # Velocity derivative = acceleration
        dvel = acceleration

        # =========================== #
        # -- QUATERNION KINEMATICS -- #
        # =========================== #

        # Change in quaternion
        dqdt = quaternionDerivative(quat, omega)

        # ============================= #
        # -- ANGULAR VELOCITY CHANGE -- #
        # ============================= #

        # Lever arm in body frame
        r_cp = np.array([0.0, 0.0, self.structure.cp_current - self.structure.cm_current])
        r_thrust = np.array([0.0, 0.0, self.structure.length - self.structure.cm_current])

        # Torques T = r (cross) F
        torque_thrust = np.linalg.cross(r_thrust, thrust_vector_body)
        torque_drag = np.linalg.cross(r_cp, drag_force_body)
        torque_coriolis = np.linalg.cross(r_cp, coriolis_force_body)
        torque_roll = np.zeros(3)
        for f, x in zip(lift_force_body, self.roll_control.fins):
            torque_roll += np.linalg.cross(x.location, f)

        torque_body_total = torque_thrust + torque_drag + torque_coriolis + torque_roll

        # Angular acceleration
        # !eventually update I to reflect better physics and mass distribution!
        # domega          = torque_body_total / self.structure.I
        domega = (torque_thrust + torque_roll) / self.structure.I
        # domega = np.zeros(3)

        if side_effect:
            # print(f"THRUST: {np.round(torque_thrust,2)} || DRAG: {np.round(torque_drag,2)} || CORIOLIS: {np.round(torque_coriolis,2)}")
            # print(f"T: {time} || THRUST: {np.round(thrust_vector_body,2)}")
            # print(f"GIMBAL X: {np.rad2deg(self.tvc.theta_y):.2f} || TORQUE: {np.rad2deg(np.arctan(x/z)):.2f} || QUAT: {np.round(quat),3}")
            # print(f"THRUST BODY: {np.round(thrust_vector_body,1)} || TORQUE BODY: {np.round(torque_thrust,2)} || TORQUE EARTH: {np.round(R.from_quat(quat).apply(torque_thrust),2)}")
            # print(f"ACTUAL: {domega[1]*dt}")
            # print(np.round(pos,2))
            print(f"ACTUAL TORQUE: {np.round(torque_thrust+torque_roll, 2)}")
            pass

        # ========================= #
        # -- GIMBAL ANGLE CHANGE -- #
        # ========================= #

        dtheta_x = self.tvc.d_theta_x
        dtheta_y = self.tvc.d_theta_y
        dgimbal_q = quaternionDerivative(self.tvc.gimbal_orientation.as_quat(), np.array([dtheta_x, dtheta_y, 0.0]))

        # ================= #
        # -- Mass Change -- #
        # ================= #

        dmdt = self.structure.dm

        # ====================== #
        # -- State Derivative -- #
        # ====================== #

        # The change in state variables
        dstate = np.concatenate([
            dpos,
            dvel,
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
        """Handles all force related equations"""
        # =============== #
        # -- Constants -- #
        # =============== #
        pos, vel, quat, omega, tvc_quat, mass, time, _, _, _ = unpackStates(state)
        alt_m = pos[2]

        rho = self.air.getDensity()
        stat_pres = self.air.getStaticPressure()
        reynolds = self.air.getReynoldsNumber(np.linalg.norm(vel), characteristic_length=self.structure.diameter)
        viscosity = self.air.getDynamicViscosity()
        mach = self.air.getMachNumber(velocity_mps=np.linalg.norm(vel))

        # ================= #
        # -- UPDATE LOGS -- #
        # ================= #

        if side_effect:
            self.mach.append(mach)
            self.reynolds.append(reynolds)
            self.viscosity.append(viscosity)
            self.velocity.append(np.linalg.norm(vel))
            self.dynamic_pres.append(self.air.getDynamicPressure(vel))

        # ============== #
        # -- Velocity -- #
        # ============== #

        # Velocity in world frame
        # velocity rocket - wind velocities
        v_air_global = vel - self.wind.getWindVelocity(alt_m=pos[2])

        # Determine rocket orientation on inertial frame
        r_world_to_body = R.from_quat(quat).inv()

        # Determine velocity of air on body (for drag calculations)
        v_air_body = r_world_to_body.apply(v_air_global)

        # ==================== #
        # -- NATURAL FORCES -- #
        # ==================== #

        # Gravity [0 0 9.8] function of altitude
        gravity_global = self.grav.getGravity(alt_m=alt_m)
        gravity_force_global = gravity_global * mass

        # Coriolis [x y z] function of velocity
        coriolis_acc_body = self.cor.getCoriolisEffect(vel_m_s=v_air_body)
        coriolis_force_body = coriolis_acc_body * mass
        coriolis_force_body = np.zeros(3)
        coriolis_force_global = R.from_quat(quat).apply(coriolis_force_body)

        # ================= #
        # -- DRAG FORCES -- #
        # ================= #

        # Determine angle of vehicle for drag calculations (not currently implemented)
        # aoa = np.arctan2(v_air_body[1], v_air_body[2])
        # beta = np.arctan2(v_air_body[0], v_air_body[2])

        # Drag force function
        drag_force_body = self.aerodynamics.getDragForce(vel=v_air_body)
        # drag_force_body = np.zeros(3)
        drag_force_global = R.from_quat(quat).apply(drag_force_body)

        # =================== #
        # -- THRUST FORCES -- #
        # =================== #

        # Get raw thrust (no direction) [z]
        thrust_mag = self.engine.runBurn(dt=dt, alt_m=alt_m, side_effect=side_effect)

        # Save burntime for later display
        if thrust_mag == 0 and not self.burntime:
            self.burntime = time

        # Update tvc thrust values for easier computing
        # !pulls data from other objects, keep as function!
        self.tvc.update_variables_(thrust_mag)

        torque_cmd = self.quaternion.compute_command_torque(rocket_pos=pos, rocket_quat=quat,
                                                                             inertia_matrix=self.structure.I,
                                                                             rocket_omega=omega,
                                                                             side_effect=side_effect
                                                                             )

        # Get updated gimbal orientation in body frame
        thrust_dir_body: R = self.tvc.compute_gimbal_orientation_using_torque(
            torque_body_cmd=torque_cmd,
            dt=dt,
            side_effect=side_effect
        )

        # Apply gimbal rotation to thrust vector in body frame
        thrust_vector_body = thrust_dir_body.apply([0, 0, thrust_mag])
        # thrust_vector_body = thrust_dir_body * thrust_mag

        # Rotate thrust vecotr to inertial frame
        thrust_force_global = R.from_quat(quat).apply(thrust_vector_body)  # Application of rotation

        if side_effect and thrust_mag != 0:
            self.thrust.append(thrust_force_global)
            # print(f"THRUST {np.round(thrust_force_global, 2)}")

        # ================== #
        # -- ROLL CONTROL -- #
        # ================== #

        # Get new tab angles (return for debug only, other definitions access objects directly)
        x_theta, x__theta, y_theta, y__theta = self.roll_control.calculate_theta(torque_cmd=torque_cmd, rho=rho,
                                                                                 vel=v_air_body, dt=dt)
        # Returns a list of all tab forces
        lift_force_body = self.aerodynamics.getLiftForce(vel_ms=v_air_body, roll_tabs=self.roll_control)

        # Convert to body force after summing the arrays (should come to 0)
        lift_force_global = R.from_quat(quat).apply(sum(lift_force_body))

        # ======================== #
        # -- TOTAL GLOBAL FORCE -- #
        # ======================== #

        total_force_global = (thrust_force_global + drag_force_global + gravity_force_global +
                              coriolis_force_global + lift_force_global)

        return thrust_vector_body, drag_force_body, coriolis_force_body, lift_force_body, total_force_global

    def _initialize_vehicle(self):
        """A function to immediately display vehicle information on launch/initialization"""
        print("=" * 60)
        print("INITIALIZING VEHICLE")
        print(f"Initial Vehicle Mass:   {self.structure.wetMass} [kg]")
        # print(f"Initial Fluid Mass:     {self.structure.liquidMass} [kg]")
        print(f"Initial Fluid Mass:     {self.structure.fluid_mass} [kg]")
        print("=" * 60)

# thrust_vector_body = gimbal_orient.apply([0.0, 0.0, thrust_mag])  # original thrust vector straight out

import numpy as np

from models import EnvironmentalModels
from models.Aero.AerodynamicsModel import *
from models.Engine.EngineModels import *
from models.Structural.StructuralModel import StructuralModel
from models.TVC.LqrQuaternionModels import LQR, QuaternionFinder
from models.TVC.TVCModel import *


def rk4_step(rocket, state, dt):
    k1 = rocket.getDynamics(state, dt, side_effect=False)
    k2 = rocket.getDynamics(state + 0.5 * dt * k1, dt, side_effect=False)
    k3 = rocket.getDynamics(state + 0.5 * dt * k2, dt, side_effect=False)
    k4 = rocket.getDynamics(state + dt * k3, dt, side_effect=True)

    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Normalize the two quaternions in the state variables
    q = next_state[6:10]
    q /= np.linalg.norm(q)

    # Update those values
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
    dtheta = state[17:20]

    return pos, vel, quat, omega, mass, time, aoa, beta, dtheta


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

        # -- CONTROL THEORY VARIABLES -- #
        q = np.diag([10, 10, 10, 2, 2, 2]) * 200
        r = np.diag([1, 1, 1]) * 40
        k_pos = 1
        k_vel = 25

        # -- MISC OBJECTS -- #
        self.grav = EnvironmentalModels.GravityModel()
        self.cor = EnvironmentalModels.CoriolisModel()
        self.air = EnvironmentalModels.AirProfile()
        self.wind = EnvironmentalModels.WindProfile()  # Currently not implemented (monte-carlo style sims)

        # - Init Atmosphere - #
        self.air.getCurrentAtmosphere(altitudes_m=0, time=0)

        # -- VEHICLE OBJECTS -- #
        self.engine = Engine()
        self.structure = StructuralModel(engine_class=self.engine, liquid_total_ratio=0.56425)
        self.aerodynamics = Aerodynamics(self.air)
        self.tvc = TVCStructure(
            structure_model=self.structure,
            aero_model=self.aerodynamics)

        # - tabs - #
        z_comp = self.structure.cm_current - self.structure.length
        r_comp = 0.2032
        self.pos_x_tab = FinTab(rocket_position=np.array([r_comp, 0, z_comp]), name="PX", positive_torque=-1)
        self.neg_x_tab = FinTab(rocket_position=np.array([-r_comp, 0, z_comp]), name="NX", positive_torque=1)
        self.pos_y_tab = FinTab(rocket_position=np.array([0, r_comp, z_comp]), name="PY", positive_torque=1)
        self.neg_y_tab = FinTab(rocket_position=np.array([0, -r_comp, z_comp]), name="NY", positive_torque=-1)
        self.fins = [self.pos_x_tab, self.neg_x_tab, self.pos_y_tab, self.neg_y_tab]
        self.roll_control = RollControl(fins=self.fins, struct=self.structure)

        # -- CONTROL -- #
        self.lqr = LQR(q=q, r=r, k_pos=k_pos, k_vel=k_vel)
        self.quaternion = QuaternionFinder(lqr=self.lqr)

        # -- STATES -- #
        self.state = np.array([
            0.0, 0.0, 0.0,  # Position (x, y, z)
            0.0, 0.0, 0.0,  # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0, 1.0, # Quaternion
            0.0, 0.0, 0.0,  # Angular Velocity
            self.structure.wetMass,
            0.0,  # Time
            0.0, 0.0,  # AOA, BETA
            0.0, 0.0,  # Gimbal angles
        ])

        # -- LOGS -- #
        self.acceleration   = 0
        self.thrust         = [np.array([0,0,0])]  # Thrust curve
        self.burntime       = None  # Time burn stops
        self.mach           = []  # Mach list
        self.velocity       = []  # Velocity list
        self.reynolds       = []  # Reynolds list
        self.viscosity      = []
        self.dynamic_pres   = []

        self.pitchXZ        = []
        self.yawYZ          = []
        self.velAOA         = []
        self.torque_cmd     = []
        self.torque_act     = []
        self.thrust_magn    = []

        # Init print function
        self._initialize_vehicle()

    def getDynamics(self, state, dt: float, side_effect: bool = True):

        # Unpack current states
        pos, vel, quat, omega, mass, time, aoa, beta, dtheta = unpackStates(state=state)

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
        acceleration    = total_force_global / mass

        # Position derivative = velocity
        dpos            = vel

        # Velocity derivative = acceleration
        dvel            = acceleration

        # =========================== #
        # -- QUATERNION KINEMATICS -- #
        # =========================== #

        # Change in quaternion
        dqdt            = quaternionDerivative(quat, omega)

        # ============================= #
        # -- ANGULAR VELOCITY CHANGE -- #
        # ============================= #

        # Lever arm in body frame
        r_cp            = np.array([0.0, 0.0, self.structure.cp_current - self.structure.cm_current])
        r_thrust        = np.array([0.0, 0.0, self.structure.length - self.structure.cm_current])

        # -- Torques T = r (cross) F -- #

        # Thrust
        torque_thrust   = np.linalg.cross(r_thrust, thrust_vector_body)

        # Drag
        torque_drag     = np.linalg.cross(r_cp, drag_force_body)

        # Coriolis (not active)
        torque_coriolis = np.linalg.cross(r_cp, coriolis_force_body)

        # Roll
        torque_roll = np.zeros(3)
        for f, x in zip(lift_force_body, self.roll_control.fins):
            torque = np.linalg.cross(x.location, f)
            torque_roll += torque

        # -- TOTAL TORQUE -- #
        torque_body_total = torque_thrust + torque_roll + torque_drag

        # -- ANGULAR ACCELERATION -- #
        # !eventually update I to reflect better physics and mass distribution!
        # dw/dt = I^-1 * (Torque - w x (I*w))
        I               = self.structure.I
        omega_cross     = np.linalg.cross(omega, I @ omega)  # gyroscopic term
        domega          = np.linalg.inv(I) @ (torque_body_total - omega_cross)

        if side_effect:
            self.torque_act.append(torque_body_total)
            # print(f"ACTUAL: {np.round(torque_body_total, 4)} || DRAG: {np.round(torque_drag,4)}")
            # print(f"ACTUAL: {torque_body_total}")
            # print(r_cp, drag_force_body, torque_drag)

            print("τ_drag  :", np.round(torque_drag, 4))
            print('T_force :', np.round(drag_force_body,4))
            # print("τ_thrust (from TVC):", np.round(torque_thrust, 2))
            # print("τ_roll (from fins):", np.round(torque_roll, 2))
            # print("τ_total (applied):", np.round(torque_body_total, 2))
            # print("ω_dot:", np.round(domega, 3))
            pass

        # ========================= #
        # -- GIMBAL ANGLE CHANGE -- #
        # ========================= #

        dtheta_x        = self.tvc.d_theta[0]
        dtheta_y        = self.tvc.d_theta[1]

        # ================= #
        # -- Mass Change -- #
        # ================= #

        dmdt            = self.structure.dm

        # ====================== #
        # -- State Derivative -- #
        # ====================== #

        # The change in state variables
        dstate = np.concatenate([
            dpos,
            dvel,
            dqdt,
            domega,
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

        pos, vel, quat, omega, mass, time, _, _, _ = unpackStates(state)
        alt_m = pos[2]

        # -- SAFE QUATERNION CHECK -- #

        quat_norm = np.linalg.norm(quat)
        if not np.isfinite(quat_norm) or quat_norm < 1e-6:
            print(f"[ERROR] Quaternion norm too small or NaN at alt {pos[2]:.2f} m. Resetting to identity.")
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            quat = quat / quat_norm

        rho             = self.air.getDensity()
        stat_pres       = self.air.getStaticPressure()
        reynolds        = self.air.getReynoldsNumber(np.linalg.norm(vel), characteristic_length=self.structure.diameter)
        viscosity       = self.air.getDynamicViscosity()
        mach            = self.air.getMachNumber(velocity_mps=np.linalg.norm(vel))

        # ================= #
        # -- UPDATE LOGS -- #
        # ================= #

        if side_effect:
            self.mach.append(mach)
            self.reynolds.append(reynolds)
            self.viscosity.append(viscosity)
            self.velocity.append(np.linalg.norm(vel))
            self.dynamic_pres.append(self.air.getDynamicPressure(vel))

        # ==================== #
        # -- PITCH YAW ROLL -- #
        # ==================== #

        # Rockets positive z in world frame
        rot             = R.from_quat(quat)
        z_axis_world    = rot.apply([0, 0, 1])

        # Projection into xz plane
        z_proj          = np.array([z_axis_world[0], 0, z_axis_world[2]])
        z_proj          /= np.linalg.norm(z_proj)

        # Pitch angle from vertical (Z up)
        pitch_angle_deg = np.rad2deg(np.arctan2(z_proj[0], z_proj[2]))

        # Projection into  xy plane
        z_proj          = np.array([0, z_axis_world[1], z_axis_world[2]])
        z_proj          /= np.linalg.norm(z_proj)

        # Yaw angle from vertical (Z up)
        yaw_angle_deg   = np.rad2deg(np.arctan2(z_proj[1], z_proj[2]))

        # ============== #
        # -- Velocity -- #
        # ============== #

        # Velocity in world frame
        # velocity rocket - wind velocities
        v_air_global    = vel - self.wind.getWindVelocity(alt_m=pos[2])

        # Determine rocket orientation on inertial frame
        r_world_to_body = R.from_quat(quat).inv()

        # Determine velocity of air on body (for drag calculations)
        v_air_body      = r_world_to_body.apply(v_air_global)

        # AOA -- velocity_vector *(dot) direction of body up (comparison axis)
        v_hat           = v_air_body / (np.linalg.norm(v_air_body) + 1e-6)
        z_body          = np.array([0, 0, 1])
        dot             = np.clip(np.dot(v_hat, z_body), -1.0, 1.0)
        # Take arc-cos
        aoa_deg         = np.rad2deg(np.arccos(dot))

        # ==================== #
        # -- NATURAL FORCES -- #
        # ==================== #

        # Gravity [0 0 9.8] function of altitude
        gravity_global_acc      = self.grav.getGravity(alt_m=alt_m)
        gravity_force_global    = gravity_global_acc * mass

        # Coriolis [x y z] function of velocity
        coriolis_acc_body       = self.cor.getCoriolisEffect(vel_m_s=v_air_body)
        coriolis_force_body     = coriolis_acc_body * mass
        coriolis_force_body     = np.zeros(3)
        coriolis_force_global   = R.from_quat(quat).apply(coriolis_force_body)

        # ================= #
        # -- DRAG FORCES -- #
        # ================= #

        # Drag force function
        drag_force_body     = self.aerodynamics.getDragForce(vel=v_air_body)
        drag_force_global   = R.from_quat(quat).apply(drag_force_body)

        # ========================= #
        # -- TORQUE DISTURBANCES -- #
        # ========================= #

        # Compute the torque disturbances to feed into the lqr
        torque_drag         = np.linalg.cross(np.array([0.0, 0.0, self.structure.cp_current - self.structure.cm_current]), drag_force_body)

        # =================== #
        # -- THRUST FORCES -- #
        # =================== #

        # Get raw thrust (no direction) [z]
        thrust_mag          = self.engine.runBurn(dt=dt, alt_m=alt_m, side_effect=side_effect)

        # Update tvc thrust values for easier computing
        # !pulls data from other objects, keep as function!
        self.tvc.update_variables_(thrust_mag)

        # Sum of expected accelerations to feed to LQR
        accel_base          = drag_force_global/mass + gravity_global_acc

        if thrust_mag != 0:
            torque_cmd      = self.quaternion.compute_command_torque(rocket_pos=pos, rocket_quat=quat, time=time,
                                                                inertia_matrix=self.structure.I, rocket_omega=omega,
                                                                rocket_vel=v_air_global, accel_base=accel_base,
                                                                side_effect=side_effect, dt=dt, drag_torque=torque_drag
                                                                )

        else:
            torque_cmd      = np.zeros(3)
        # Get updated gimbal orientation in body frame

        thrust_dir_body     = self.tvc.compute_gimbal_orientation_using_torque(torque_body_cmd=torque_cmd, dt=dt,
                                                                           side_effect=side_effect, time=time
                                                                           )

        # Apply gimbal rotation to thrust vector in body frame
        thrust_vector_body  = thrust_dir_body * thrust_mag

        # Rotate thrust vecotr to inertial frame
        thrust_force_global = R.from_quat(quat).apply(thrust_vector_body)

        # ================== #
        # -- ROLL CONTROL -- #
        # ================== #

        # Get new tab angles (return for debug only, other definitions access objects directly)
        x_theta, x__theta, y_theta, y__theta = self.roll_control.calculate_theta(torque_cmd=torque_cmd, rho=rho,
                                                                                 vel=v_air_body, dt=dt, time=time,
                                                                                 side_effect=side_effect)
        # Returns a list of all tab forces
        lift_force_body     = self.aerodynamics.getLiftForce(vel_ms=v_air_body, roll_tabs=self.roll_control, time=time,
                                                         side_effect=side_effect)

        # Convert to body force after summing the arrays (should come to 0)
        lift_force_global   = R.from_quat(quat).apply(sum(lift_force_body))

        # ======================== #
        # -- TOTAL GLOBAL FORCE -- #
        # ======================== #

        total_force_global  = (thrust_force_global + drag_force_global + gravity_force_global +
                              coriolis_force_global + lift_force_global)

        # ======================= #
        # -- LOGGING AND DEBUG -- #
        # ======================= #

        # Logging
        if side_effect:
            self.pitchXZ.append(pitch_angle_deg)
            self.yawYZ.append(yaw_angle_deg)
            self.velAOA.append(aoa_deg)
            self.torque_cmd.append(torque_cmd)
            self.thrust_magn.append(thrust_mag)

        # Save burntime for later display
        if thrust_mag == 0 and not self.burntime:
            print(f"[SHUTDOWN] Alt: {pos[2]}")
            print("=" * 100)
            self.burntime = time

        # Debug
        if side_effect:
            # print(f"EXPECTED: {np.round(torque_cmd,2)}")
            # print(f"ALT: {pos[2]:.2f}")
            # print(f"DRAG: {np.round(drag_force_global,4)}")
            # print(f"BODY VEL: {v_air_body}")
            # print(f"GLOBAL VEL: {v_air_global}")
            # print(f"DRAG: {drag_force_body}")
            print(f"GIMBAL: {np.round(np.rad2deg(self.tvc.theta),2)}")
            print(f"PITCH:  {pitch_angle_deg:.2f}")
            pass


        return thrust_vector_body, drag_force_body, coriolis_force_body, lift_force_body, total_force_global

    def _initialize_vehicle(self):
        """A function to immediately display vehicle information on launch/initialization"""
        print("=" * 60)
        print("INITIALIZING VEHICLE")
        print(f"Initial Vehicle Mass:   {self.structure.wetMass} [kg]")
        # print(f"Initial Fluid Mass:     {self.structure.liquidMass} [kg]")
        print(f"Initial Fluid Mass:     {self.structure.fluid_mass} [kg]")
        print("=" * 60)


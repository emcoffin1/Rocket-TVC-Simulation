# By John Nguyen <knightofthealtar64@gmail.com>
# Python 3.10.11 with VPython 7.6.4
# MIT License

import math
import random
import numpy as np
from vpython import *



# UTILITY FUNCTIONS

# azimuth & elevation angle to 3d xyz unit vector
def yp_unit(az, elev):
    return vec(
        cos(az) * cos(elev),  # x
        sin(az) * cos(elev),  # y
        sin(elev)  # z
    )


# Common number string formatting style
def d2(num):
    return "{:.2f}".format(num)


# ENVIRONMENT FUNCTIONS

cg_E = vec(0, 0, -6371000)  # m, position of the center of the Earth

# Specific gas constant of air (R/M)
r_spec = 287.05  # J/(kg * K)


def altitude(pos: vector):
    return mag(cg_E - pos) - mag(cg_E)


# air density function, based on International Standard Atmosphere
def rho(pos: vector):  # kg/m^3 air pressure
    y = altitude(pos)  # Altitude
    if y <= 11000:
        dens = -(1.2985 - 0.3639) / 11610 * (y + 610) + 1.2985
    elif 11000 < y <= 20000:
        dens = -(0.3639 - 0.088) / (20000 - 11000) * (y - 11000) + 0.3639
    elif 20000 < y <= 32000:
        dens = -(0.088 - 0.0132) / (32000 - 20000) * (y - 20000) + 0.088
    elif 32000 < y <= 47000:
        dens = -(0.0132 - 0.002) / (47000 - 32000) * (y - 32000) + 0.0132
    else:
        dens = 0
    return dens


# air temperature function, from ISA
def temp(pos: vector):  # K temperature
    y = altitude(pos)  # Altitude
    if y <= 11000:
        T = 273 + 19 - 6.5 * y / 1000  # -6.5 K per km
    elif 11000 < y <= 20000:
        T = 273 - 56.5 - 0 * y / 1000  # +0.0 K per km
    elif 20000 < y <= 32000:
        T = 273 - 56.5 + 1.0 * y / 1000  # +1.0 K per km
    elif 32000 < y <= 47000:
        T = 273 - 44.5 + 2.8 * y / 1000  # +2.8 K per km
    else:
        T = 273 - 44.5
    return T


# static pressure function
def P(pos: vector):
    return rho(pos) * r_spec * temp(pos)


# speed of sound, interpolated from Engineering Toolbox
def c(pos: vector):
    y = altitude(pos)
    c_points = [344, 340, 330, 320, 310, 300, 295, 295, 300, 300]
    y_points = [-1000, 0, 2500, 5000, 7500, 10000, 11000, 20000, 25000, 100000]
    return np.interp(y, y_points, c_points)


G = 6.674e-11  # Nm^2/kg^2, Universal gravitational constant

g_0 = 9.80665  # m/s^2, Earth standard gravity

M_E = 6e24  # kg, Earth mass


# Gravitational force function
# Returns a vector pointing at the center of the Earth
def g(pos: vector, mass: float):
    return G * M_E * mass / (mag(cg_E - pos)) ** 2 * hat(cg_E - pos)


# wind profile class declaration
class WindProfile:
    def __init__(self, name: str, mu: float, sigma: float, angle_mu: float, angle_sigma: float, step: float,
                 print_debug: bool):
        self.name = name

        # GENERATION PARAMETERS
        self.mu = mu  # m/s, mean windspeed
        self.sigma = sigma  # m/s, std. dev. of windspeed
        self.angle_mu = angle_mu  # rad, mean change in angle per band
        self.angle_sigma = angle_sigma  # rad, std. dev. of change in angle per band
        self.step = step  # altitude band size

        alt = -50
        self.altSet = list()
        while alt < 47000:
            self.altSet.append(alt)
            alt += step
        del alt

        self.speedSet = list()
        self.angleSet = list()
        for y in self.altSet:
            speed = random() * self.sigma + self.mu
            angle = random() * self.angle_sigma + self.angle_mu
            self.speedSet.append(speed)
            self.angleSet.append(angle)

        if print_debug:
            print(f"speed set: {self.speedSet}")
            print(f"angle set: {self.angleSet}")
        # end constructor

    def wind(self, pos: vector):
        alt = altitude(pos)
        index = (alt + 50) / self.step  # normalize to band number
        band_pos = index - floor(index)  # relative position within altitude band
        index = abs(int(index))  # convert to positive whole number for table index
        if index >= len(self.angleSet) - 1:
            return vec(0, 0, 0);
        azimuth = self.angleSet[index]  # grab angle from wind band
        speed = self.speedSet[index]  # grab speed from wind band
        if band_pos <= 0.2:  # if in the lower 20% of wind band, interpolate to the avg. b/w current and previous bands
            azimuth_1 = self.angleSet[index - 1]  # angle from previous wind band
            speed_1 = self.speedSet[index - 1]  # speed from previous wind band
            # avg. of 2 wind bands (y-intercept) + difference b/w current band and avg. point (slope) * rel. pos (x)
            azimuth = (azimuth_1 + azimuth) / 2 + (azimuth - (azimuth_1 + azimuth) / 2) * band_pos / 0.2
            speed = (speed + speed_1) / 2 + (speed - (speed_1 + speed) / 2) * band_pos / 0.2
        if band_pos >= 0.8:  # if in the upper 20% of wind band, interpolate to the avg. b/w current and next bands
            azimuth_1 = self.angleSet[index + 1]  # angle from next wind band
            speed_1 = self.speedSet[index + 1]  # speed from next wind band
            # current wind band (y-intercept) + difference b/w current band and avg. point (slope) * rel. pos (x)
            azimuth = azimuth - (azimuth - (azimuth_1 + azimuth) / 2) * (band_pos - 0.8) / 0.2
            speed = speed - (speed - (speed_1 + speed) / 2) * (band_pos - 0.8) / 0.2

        local_z = hat(pos - cg_E)  # local up direction
        local_x = cross(vec(0, cg_E.mag, 0), local_z).hat  # local east vector
        local_y = cross(local_z, local_x).hat  # local north vector

        return speed * rotate(local_y, -azimuth,
                              local_z)  # Rotates the north vector clockwise around the up vector by the azimuth angle


# Fin set class declaration
# Determines lift properties of fins only.  Drag is handled by the overall rocket cd, A, cd_s, and A_s
class FinSet:
    def __init__(self, num_fins: int, center: vector, pos: vector, planform: float, stall_angle: float, ac_span: float,
                 cl_pass):
        self.num_fins = num_fins  # number of fins.  3 or 4 only supported at this time.
        self.fin_rad_pos = []
        for a in arange(0, 2 * pi, 2 * pi / num_fins):
            self.fin_rad_pos.append(a)
        self.center = center  # position of center of mass relative to parent rocket nose tip
        self.pos = pos  # position of center of lift relative to parent rocket nose tip
        self.planform = planform  # planform wing area of each individual fin
        self.stall_angle = stall_angle  # maximum angle of attack before wing stall
        self.ac_span = ac_span  # radial offset from rocket centerline to center of lift of each fin
        self.cl = cl_pass
        # self.aoa_graph = graph(fast=False, title="Fin AoA", xtitle="t", ytitle="degrees")
        # self.aoa_curves = []
        # for i in self.fin_rad_pos:
        #    self.aoa_curves.append(gcurve(graph=self.aoa_graph, label=f"Fin {d2(i*180/np.pi)} AoA", color=color.blue))
        # self.curve = gcurve(graph=self.aoa_graph, label="fin 0 aoa", color=color.red)

    def ac_pos(self, rot: vector, fin_index: int):
        if fin_index > self.num_fins - 1:
            print(f"Expected fin index b/w 0 and {self.num_fins - 1} got {fin_index}")
            exit(1)
        return yp_unit(self.fin_rad_pos[fin_index] + rot.x, 0) * self.ac_span

    def lift(self, aoa: float, altitude: float, airspeed: float):
        return self.cl(aoa) * rho(altitude) * airspeed ** 2 / 2 * self.planform

    # Returns lift vector in rocket-centric coordinates.
    def lift_vec(self, rot: vector, fin_index: int, aoa: float, altitude: float, airflow: vector, roll: vector):
        L = self.lift(aoa, altitude, airflow.mag) * self.ac_pos(rot, fin_index).cross(roll).hat
        if mag(L + self.flow_perp_fin(rot, airflow, self.ac_pos(rot, fin_index), roll)) < mag(L):
            L *= -1
        return L

    # Returns the component of the airflow vector orthogonal to the fin plane.
    def flow_perp_fin(self, rot: vector, airflow: vector, ac: vector, roll: vector):
        f_II_ac = airflow.dot(ac.hat) * ac.hat  # airflow vector projected onto the aerodynamic center pos vector
        f_II_roll = airflow.dot(roll.hat) * roll.hat  # airflow vector projected onto the roll axis
        f_I_accg = airflow - (
                    f_II_roll + f_II_ac)  # remaining component of airflow vector (orthogonal to both ac pos & roll axis)
        return f_I_accg

    # Returns AoA of each fin.
    def aoa(self, rot: vector, fin_index: int, airflow: vector, roll: vector):
        ac = self.ac_pos(rot, fin_index)  # aerodynamic center position vector
        f_II_ac = airflow.dot(ac.hat) * ac.hat  # airflow vector projected onto the aerodynamic center pos vector
        f_II_roll = airflow.dot(roll.hat) * roll.hat  # airflow vector projected onto the roll axis
        f_I_accg = airflow - (
                    f_II_roll + f_II_ac)  # remaining component of airflow vector (orthogonal to both ac pos & roll axis)
        alpha = atan(
            f_I_accg.mag / f_II_roll.mag)  # right triangle with orthogonal component as opposite and roll component as adjacent
        return alpha

    # Iterates through each fin and plots its angle of attack
    def aoa_plot(self, t: float, rot: vector, airflow: vector, roll: vector):
        for idx, i in enumerate(self.aoa_curves):
            i.plot(t, self.aoa(rot, idx, airflow, roll) * 180 / np.pi)

    def total_lift_vec(self, rot: vector, airflow: vector, roll: vector, altitude: float):
        L_total = vec(0, 0, 0)
        for idx, i in enumerate(self.fin_rad_pos):
            aoa = self.aoa(rot, idx, airflow, roll)
            L = self.lift_vec(rot, idx, aoa, altitude, airflow, roll)
            L_total = L_total + L
        return L_total


# 3-axis RCS system using 8 thrusters at the top of the rocket.  Capable of full yaw/pitch/roll control.
class ReactionControlSystem:
    # set gains to zero by default
    kp = 0
    ki = 0
    kd = 0
    izone = 0
    # set physical properties to zero by default
    fuel_mass = 0
    ct = vec(0, 0, 0)
    cg = vec(0, 0, 0)
    thrust = 0
    throttle = False

    # Constructs fundamental characteristics
    def __init__(self, fuel_mass: float, cg: vector, ct: vector, rcg: vector, radius: float, thrust: float,
                 throttle: bool):
        self.fuel_mass = fuel_mass  # kg, fuel mass
        self.cg = cg  # vec, m, Center of mass
        self.ct = ct  # vec, m, Center of thrust
        self.rcg = rcg  # vec, m, Rocket center of mass
        self.thrust = thrust  # N, Max thrust per thruster
        self.radius = radius  # m, roll thruster radius from center
        self.throttle = throttle  # If true, can proportionally throttle thrusters
        self.omega_acc = vec(0, 0, 0)
        self.last_omega_err = vec(0, 0, 0)

    # Initialize or update the angular rate PID controller.
    def setPID(self, kp: float, ki: float, kd: float, izone: float):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.izone = izone  # Integral min & max

    # Initialize or update the angular rate trapezoid profile parameters.
    def setProfile(self, max_alpha: float, max_omega: float, min_err: float):
        self.max_alpha = max_alpha  # magnitude of maximum angular acceleration
        self.max_omega = max_omega  # magnitude of maximum angular velocity
        self.min_err = min_err  # minimum angular error to apply guidance

    # Set the guidance heading request.
    def setReference(self, roll_setpoint: vector, yaw_setpoint: vector):
        self.roll_setpoint = roll_setpoint.hat  # Desired roll axis
        self.yaw_setpoint = yaw_setpoint.hat  # Desired yaw axis

    # Update the current rocket pose.
    def setPose(self, roll: vector, pitch: vector, yaw: vector, omega: vector):
        self.roll = roll.hat  # vector, rocket roll axis
        self.pitch = pitch.hat  # unit vector, rocket pitch axis
        self.yaw = yaw.hat  # unit vector, rocket yaw axis
        self.omega = omega  # vector, rad/s, rocket angular velocity

    # Update the moments of inertia of the entire rocket
    def setI(self, I_O: vector):
        self.I_O = I_O  # kg m^2, mass moments of inertia about x, y, z axes

    def setRCg(self, rcg: vector):
        self.rcg = rcg  # vec, m, center of mass of rocket in local coords

    # Retrieve the current total thrust vector exerted by RCS.
    def getThrust(self):
        theta_err = diff_angle(self.roll_setpoint, self.roll) * hat(cross(self.roll, self.roll_setpoint))
        if theta_err.mag > self.min_err:
            # Desired angular velocity to reach the correct nose angle
            omega_d = theta_err
            # Position error
            omega_err = omega_d - self.omega
            # Angular velocity error integral
            self.omega_acc.x = max(min(omega_err.x + self.omega_acc.x, self.izone), -self.izone)
            self.omega_acc.y = max(min(omega_err.y + self.omega_acc.y, self.izone), -self.izone)
            self.omega_acc.z = max(min(omega_err.z + self.omega_acc.z, self.izone), -self.izone)
            # Angular velocity error derivative
            d_omega_err = (omega_err - self.last_omega_err)
            self.last_omega_err = omega_err
            # Angular acceleration from velocity PID
            alpha = omega_err * self.kp + self.omega_acc * self.ki + d_omega_err * self.kd
            # Transform alpha into local coordinates
            alpha_local = vec(alpha.dot(self.pitch), alpha.dot(self.yaw), alpha.dot(self.roll))
            # Calculate required moment about each axis to achieve alpha_local
            moment = vec(alpha.x * self.I_O.x, alpha.y * self.I_O.y, alpha.z * self.I_O.z)
            # Calculate thrust for pitch acceleration
            pitch_radius = (self.ct - self.rcg).z
            thrust_pitch = -moment.x / pitch_radius * self.yaw
            if thrust_pitch.mag > self.thrust * 2:
                thrust_pitch = self.thrust * 2 * thrust_pitch.hat
            # Calculate thrust for yaw acceleration
            yaw_radius = (self.ct - self.rcg).z
            thrust_yaw = moment.y / yaw_radius * self.pitch
            if thrust_yaw.mag > self.thrust * 2:
                thrust_yaw = self.thrust * 2 * thrust_yaw.hat
            # Calculate thrust for roll acceleration
            thrust_roll = moment.z / self.radius
            if abs(thrust_roll) > self.thrust * 4:
                thrust_roll = sign(thrust_roll) * self.thrust * 4
            # Blend yaw and pitch thrust vectors together
            thrust = thrust_pitch + thrust_yaw
            return {'thrust': thrust, 'thrust_roll': thrust_roll}
        else:
            return {'thrust': vec(0, 0, 0), 'thrust_roll': 0}


# Physics object class declaration
class FreeRocket:
    # Graph switches
    heading_enable = True
    position_graph_enable = True
    rotation_graph_enable = False
    velocity_graph_enable = True
    rotation_rate_graph_enable = True
    acceleration_graph_enable = False
    moment_graph_enable = True
    side_profile_enable = False
    top_profile_enable = False
    aoa_graph_enable = False
    force_graph_enable = True
    mass_graph_enable = True
    fin_aoa_graph_enable = False
    rcs_graph_enable = False

    fast_graphing = False
    graph_enable = True

    # Rocket centric coordinate system definition:
    # Origin point is tip of the nose cone, to physically ground all dimensions.
    # Y axis is the yaw axis, with ventral being positive and dorsal being negative.
    # X axis is the pitch axis, with port being positive and starboard being negative.
    # Z axis is the roll axis, with fore being positive and aft being negative.

    # Global coordinate system definition:
    # Origin point is the base of the launcher, just above ground level.
    # X axis runs due east, tangent to Earth surface
    # Y axis runs due north, tangent to Earth surface
    # Z axis runs straight up, co-linear with the radius from the center of the Earth to the origin

    # constructor
    def __init__(self, name: str, pos: vector, roll_axis: vector, yaw_axis: vector, v: vector, I_0: vector, cg: vector,
                 cp: vector, cd, A: float, cd_s: float, A_s: float, main_deploy_alt: float, chute_cd: float,
                 chute_A: float, drogue_cd: float, drogue_A: float, dry_mass: float, fuel_mass: float, thrust,
                 t0: float, wind: WindProfile, initDebug: bool, fin: FinSet, rcs: ReactionControlSystem):
        self.name = name
        # FUNDAMENTAL VECTORS
        self.pos = pos  # m, 3D cartesian position
        self.v = v  # vector, m/s, initial velocity
        self.I_0 = I_0  # kg*m^2, Mass moments of inertia
        # COORDINATE SYSTEM
        self.yaw_axis = yaw_axis.hat
        self.roll_axis = roll_axis.hat
        self.pitch_axis = cross(yaw_axis, roll_axis).hat
        # AERODYNAMIC PROPERTIES
        self.cp = cp  # m, center of pressure position vector (see coordinate system definition above)
        self.cd = cd  # frontal drag coefficient FUNCTION
        self.A = A  # m^2, frontal reference area
        self.cd_s = cd_s  # side drag coefficient (airflow parallel to yaw/pitch axis)
        self.A_s = A_s  # side reference area " "
        self.main_deploy_alt = main_deploy_alt  # m, main parachute deployment altitude
        self.chute_cd = chute_cd  # parachute drag coefficient
        self.chute_A = chute_A  # m^2, parachute reference area
        self.drogue_cd = drogue_cd  # drogue chute cd
        self.drogue_A = drogue_A  # m^2, drogue chute ref. area
        self.wind = wind  # Wind profile
        self.fin = fin  # Fin set
        # MASS PROPERTIES
        self.cg = cg  # m, center of mass position vector (see coordinate system definition above)
        self.dry_mass = dry_mass  # kg
        self.fuel_mass = fuel_mass  # kg
        self.mass = self.dry_mass + self.fuel_mass + rcs.fuel_mass  # kg, total initial mass

        # PROPULSION PROPERTIES

        self.thrust = thrust  # thrust function of time
        t = 0  # s, thrust time variable
        dt = 0.01  # s, thrust time step
        self.J = 0  # total impulse
        while self.thrust(t, 101325) > 0:  # integration of thrust at sea level
            self.J += self.thrust(t, 101325) * dt
            t += dt
        self.bt = t  # s, motor burn time
        self.t0 = t0  # s, ignition time
        self.t1 = self.t0 + self.bt  # s, burnout time

        # MOMENTUM PROPERTIES
        self.p = self.v * self.mass  # kgm/s, Translational momentum
        self.v = self.p / self.mass  # m/s, Translational velocity
        self.drot = vec(0, 0, 0)  # rad/s, pitch/yaw/roll rate
        self.L = vec(  # Angular momentum, pitch/yaw/roll
            self.I_0.x * self.drot.x,
            self.I_0.y * self.drot.y,
            self.I_0.z * self.drot.z
        )

        # GUIDANCE SYSTEM
        self.rcs = rcs
        self.rcs.setRCg(self.cg)
        self.rcs.setI(self.I_0)
        self.rcs.setPose(self.roll_axis, self.pitch_axis, self.yaw_axis, self.drot)

        # State variables for drogue and main parachute deployment, NOT deployment toggles.
        self.drogue = False
        self.main_chute = False
        self.main_time = 0  # Used in sim loop to track time after main deploy, for area ramp.  This prevents an unrealistically hard opening shock.

        # Debug prints
        if initDebug:
            print(f"{self.name} initial momentum: {self.p}kgm/s")
            print(
                f"{self.name} position: {self.pos}m orientation: {self.roll_axis} moments of inertia (PYR): {self.I_0}kg-m^2")
            print(f"{self.name} total impulse at sea level: {d2(self.J)}Ns")

        # Graph object declaration

        if FreeRocket.rcs_graph_enable:
            self.rcs_graph = graph(title=f"RCS Thrust of {self.name}", xtitle="Time (s)", ytitle="Thrust (N)",
                                   fast=FreeRocket.fast_graphing)
            self.rcs_x = gcurve(graph=self.rcs_graph, color=color.red, label="x")
            self.rcs_y = gcurve(graph=self.rcs_graph, color=color.green, label="y")
            self.rcs_z = gcurve(graph=self.rcs_graph, color=color.blue, label="z")

        if FreeRocket.heading_enable:
            self.heading_graph = graph(title=f"Roll-axis Vector of {self.name}", xtitle="Time (s)", ytitle="Component",
                                       fast=FreeRocket.fast_graphing)
            self.heading_x = gcurve(graph=self.heading_graph, color=color.red, label="x")
            self.heading_y = gcurve(graph=self.heading_graph, color=color.green, label="y")
            self.heading_z = gcurve(graph=self.heading_graph, color=color.blue, label="z")

        if FreeRocket.side_profile_enable:
            self.flight_side_graph = graph(title=f"Side Profile of {self.name}", xtitle="Downrange Distance (ft)",
                                           ytitle="Altitude (ft)", fast=FreeRocket.fast_graphing)
            self.flight_side = gcurve(graph=self.flight_side_graph, color=color.green, label="Side-on Track")

        if FreeRocket.top_profile_enable:
            self.flight_top_graph = graph(title="Top Profile of " + self.name, xtitle="x (ft)",
                                          ytitle="z (ft)", fast=FreeRocket.fast_graphing)
            self.flight_top = gcurve(graph=self.flight_top_graph, color=color.blue, label="Top-Down Track")

        if FreeRocket.acceleration_graph_enable:
            self.acceleration_graph = graph(title=f"Acceleration of {self.name}", xtitle="t", ytitle="G",
                                            fast=FreeRocket.fast_graphing)
            self.acceleration_x = gcurve(graph=self.acceleration_graph, color=color.red,
                                         label="a<sub>x</sub>")  # G, x-axis acceleration
            self.acceleration_y = gcurve(graph=self.acceleration_graph, color=color.green,
                                         label="a<sub>y</sub>")  # G, y-axis acceleration
            self.acceleration_z = gcurve(graph=self.acceleration_graph, color=color.blue,
                                         label="a<sub>z</sub>")  # G, z-axis acceleration
            self.acceleration_total = gcurve(graph=self.acceleration_graph, color=color.black,
                                             label="a")  # G, total acceleration
            self.acceleration_g = gcurve(graph=self.acceleration_graph, color=color.green,
                                         label="a<sub>g</sub>")  # G, gravitational acceleration at this altitude

        if FreeRocket.position_graph_enable:
            self.position_graph = graph(title=f"Position of {self.name}", xtitle="seconds", ytitle="ft",
                                        fast=FreeRocket.fast_graphing)
            self.position_x = gcurve(graph=self.position_graph, color=color.red, label="x")  # ft, x position
            self.position_y = gcurve(graph=self.position_graph, color=color.green, label="y")  # ft, y position
            self.position_z = gcurve(graph=self.position_graph, color=color.blue, label="z")  # ft, z position
            self.position_alt = gcurve(graph=self.position_graph, color=color.magenta, label="alt")  # ft, altitude
            self.position_total = gcurve(graph=self.position_graph, color=color.black,
                                         label="Downrange distance")  # ft, downrange distance

        if FreeRocket.rotation_rate_graph_enable:
            self.rotation_rate_graph = graph(title=f"Rotation rate of {self.name}", xtitle="seconds",
                                             ytitle="degrees/s", fast=FreeRocket.fast_graphing)
            self.rotation_rate_yaw = gcurve(graph=self.rotation_rate_graph, color=color.red, label="Yaw rate")
            self.rotation_rate_pitch = gcurve(graph=self.rotation_rate_graph, color=color.green, label="Pitch rate")
            self.rotation_rate_roll = gcurve(graph=self.rotation_rate_graph, color=color.blue, label="Roll rate")
            self.rotation_rate_total = gcurve(graph=self.rotation_rate_graph, color=color.black, label="Total rate")

        if FreeRocket.aoa_graph_enable:
            self.aoa_graph = graph(title=f"AoA of {self.name}", xtitle="seconds", ytitle="alpha, degrees",
                                   fast=FreeRocket.fast_graphing)
            self.aoa = gcurve(graph=self.aoa_graph, color=color.red, label="alpha")  # degrees, Angle of attack

        if FreeRocket.moment_graph_enable:
            self.moment_graph = graph(title=f"Moments on {self.name}", xtitle="seconds", ytitle="lb-ft",
                                      fast=FreeRocket.fast_graphing)
            self.moment_yaw = gcurve(graph=self.moment_graph, color=color.red, label="Yaw moment")
            self.moment_pitch = gcurve(graph=self.moment_graph, color=color.green, label="Pitch moment")
            self.moment_roll = gcurve(graph=self.moment_graph, color=color.blue, label="Roll moment")
            self.moment_total = gcurve(graph=self.moment_graph, color=color.black, label="Total moment")

        if FreeRocket.velocity_graph_enable:
            self.velocity_graph = graph(title=f"Velocity of {self.name}", xtitle="seconds", ytitle="ft/s",
                                        fast=FreeRocket.fast_graphing)
            self.velocity_x = gcurve(graph=self.velocity_graph, color=color.red, label="v<sub>x</sub>")
            self.velocity_y = gcurve(graph=self.velocity_graph, color=color.green, label="v<sub>y</sub>")
            self.velocity_z = gcurve(graph=self.velocity_graph, color=color.blue, label="v<sub>z</sub>")
            self.velocity_total = gcurve(graph=self.velocity_graph, color=color.black, label="|v|")

        if FreeRocket.force_graph_enable:
            self.force_graph = graph(title=f"Forces on {self.name}", xtitle="seconds", ytitle="lbf",
                                     fast=FreeRocket.fast_graphing)
            self.drag_plot = gcurve(graph=self.force_graph, color=color.blue, label="F<sub>d</sub>")
            self.thrust_plot = gcurve(graph=self.force_graph, color=color.red, label="F<sub>t</sub>")
            self.gravity_plot = gcurve(graph=self.force_graph, color=color.green, label="F<sub>g</sub>")

        if FreeRocket.mass_graph_enable:
            self.mass_graph = graph(title=f"Mass of {self.name}", xtitle="seconds", ytitle="lbm",
                                    fast=FreeRocket.fast_graphing)
            self.mass_plot = gcurve(graph=self.mass_graph, color=color.red, label="m")

        self.v_max = 0  # m/s, maximum absolute velocity
        self.v_max_time = 0  # s, max velocity time
        self.z_max = 0  # m, maximum altitude
        self.z_max_time = 0  # s, apogee time
        self.a_max = 0  # m/s^2, maximum acceleration
        self.g_max = 0  # G, maximum acceleration
        self.a_max_time = 0  # s, max accel time
        self.v_ground_hit = 0  # m/s, ground hit velocity
        self.duration = 0  # s, flight duration
        self.q_max = 0  # Pa, maximum dynamic pressure
        self.q_max_time = 0  # s
        self.q_max_speed = 0  # m/s
        self.q_max_accel = 0  # m/s^2
        self.mach_max = 0  # M, maximum Mach number
        self.mach_max_time = 0  # s
        self.mach_max_speed = 0  # m/s
        self.mach_max_altitude = 0  # m
        self.drag_loss = 0  # m/s
        self.grav_loss = 0  # m/s
        self.cosine_loss = 0  # m/s
        # end of constructor

    # Reference area estimator
    def A_alpha(self, alpha):  # Sine interpolation between frontal area and side area.
        return abs(sin(alpha) * self.A_s + cos(alpha) * self.A)

    # Simulation
    def simulate(self, t, dt):
        # FORCE & MOMENT COMPONENTS

        # BODY DRAG
        heading = self.roll_axis  # 3d linear unit vector of vehicle orientation
        airflow = self.v + self.wind.wind(self.pos)  # 3d linear vector of oncoming airstream (reversed)
        alpha = math.acos(heading.dot(airflow.hat))  # rad, angle of attack
        cd = self.cd(self.v.mag / c(self.pos))  # drag coefficient
        A = self.A_alpha(alpha)  # m^2, reference area
        f_drag = airflow.mag ** 2 * rho(self.pos) * cd * A / 2 * -airflow.hat  # N, body drag force

        M_drag = f_drag.cross(self.roll_axis * mag(self.cp - self.cg))  # Moment generated by drag force

        # FIN LIFT
        # M_lift = self.fin.total_lift_vec(self.rot, airflow, self.cg, self.pos.z).cross(self.fin.center - self.cg)

        # DROGUE PARACHUTE DRAG
        if self.drogue:
            f_drogue = airflow.mag ** 2 * rho(self.pos) * self.drogue_cd * self.drogue_A / 2 * -airflow.hat
        else:
            f_drogue = vec(0, 0, 0)

        # MAIN PARACHUTE DRAG
        if self.main_chute:
            opening_time = 3
            ramp = min(t - self.main_time, opening_time) / opening_time  # Ramps parachute opening area over time
            f_chute = airflow.mag ** 2 * rho(self.pos) * self.chute_cd * self.chute_A * ramp / 2 * -airflow.hat
        else:
            f_chute = vec(0, 0, 0)

        # GRAVITY FORCE
        f_grav = g(self.pos, self.mass)

        # THRUST VECTOR
        f_thrust = vec(0, 0, 0)
        if self.t0 <= t <= self.t1:
            f_thrust = self.thrust(t - self.t0, P(self.pos)) * self.roll_axis

        # REACTION CONTROL SYSTEM
        # This method doesnt update the RCS system so you can apply the updates to the guidance inputs at any desired frequency
        rcs_out = self.rcs.getThrust()
        f_rcs = rcs_out['thrust']
        f_rcs_roll = rcs_out['thrust_roll']

        # Moment due to yaw & pitch thrusters
        M_rcs = cross(self.roll_axis * (self.rcs.ct - self.cg).mag, f_rcs)
        # Moment due to roll thrusters
        M_rcs_roll = self.rcs.radius * f_rcs_roll * self.roll_axis

        # TOTAL NET FORCE
        f_net = f_grav + f_thrust + f_drag + f_chute + f_drogue + f_rcs

        # TOTAL NET MOMENT
        M_net = M_drag + M_rcs + M_rcs_roll

        self.p = self.p + f_net * dt  # incrementing linear momentum
        self.v = self.p / self.mass  # calculating velocity
        self.pos = self.pos + self.v * dt  # incrementing position

        self.L += M_net * dt  # incrementing angular momentum
        L_local = vec(self.L.dot(self.pitch_axis), self.L.dot(self.yaw_axis),
                      self.L.dot(self.roll_axis))  # Transforming angular momentum into local coordinate system
        # Angular velocity in local coordinates
        self.drot = vec(L_local.x / self.I_0.x, L_local.y / self.I_0.y,
                        L_local.z / self.I_0.z)  # calculating rotation rate using 3-axis moments of inertia
        # Angular velocity transformed to global coordinates
        self.drot = self.drot.x * self.pitch_axis + self.drot.y * self.yaw_axis + self.drot.z * self.roll_axis
        # Rotating local coordinate base vectors
        self.roll_axis = rotate(self.roll_axis, self.drot.mag * dt, self.drot)
        self.pitch_axis = rotate(self.pitch_axis, self.drot.mag * dt, self.drot)
        self.yaw_axis = rotate(self.yaw_axis, self.drot.mag * dt, self.drot)

        # End of movement during this iteration

        # Remove burned fuel from vehicle mass
        # Assumes fuel mass flow rate is proportional to thrust
        self.mass = self.mass - f_thrust.mag * dt / self.J * self.fuel_mass

        # Parachute deployment checks

        if self.v.dot(self.pos - cg_E) < 0 and not self.drogue:
            self.drogue = True
            print(
                f"{self.name} Drogue deployment at T+{d2(t)}s at Altitude:{d2(altitude(self.pos) * 39.4 / 12)}ft & Speed:{d2(self.v.mag * 39.4 / 12)}ft/s")

        if altitude(self.pos) < self.main_deploy_alt and self.drogue and not self.main_chute:
            self.main_chute = True
            self.main_time = t
            print(
                f"{self.name} Main chute deployment at T+{d2(t)}s at Altitude:{d2(altitude(self.pos) * 39.4 / 12)}ft & Speed:{d2(self.v.mag * 39.4 / 12)}ft/s")

        mach_n = airflow.mag / c(self.pos)
        # Graph switched variables
        if self.graph_enable:
            if FreeRocket.heading_enable:
                self.heading_x.plot(t, self.roll_axis.x)
                self.heading_y.plot(t, self.roll_axis.y)
                self.heading_z.plot(t, self.roll_axis.z)
            if FreeRocket.side_profile_enable:
                self.flight_side.plot(sqrt((self.pos.x * 39.4 / 12) ** 2 + (self.pos.y * 39.4 / 12) ** 2),
                                      self.pos.z * 39.4 / 12)
            if FreeRocket.top_profile_enable:
                self.flight_top.plot(self.pos.x * 39.4 / 12, self.pos.y * 39.4 / 12)
            if FreeRocket.moment_graph_enable:
                self.moment_yaw.plot(t, M_net.y * 39.4 / 12 / 4.448)
                self.moment_pitch.plot(t, M_net.x * 39.4 / 12 / 4.448)
                self.moment_roll.plot(t, M_net.z * 39.4 / 12 / 4.448)
                self.moment_total.plot(t, M_net.mag * 39.4 / 12 / 4.448)
            if FreeRocket.velocity_graph_enable:
                self.velocity_x.plot(t, self.v.x * 39.4 / 12)
                self.velocity_y.plot(t, self.v.y * 39.4 / 12)
                self.velocity_z.plot(t, self.v.z * 39.4 / 12)
                self.velocity_total.plot(t, self.v.mag * 39.4 / 12)
            if FreeRocket.position_graph_enable:
                self.position_x.plot(t, self.pos.x * 39.4 / 12)
                self.position_y.plot(t, self.pos.y * 39.4 / 12)
                self.position_z.plot(t, self.pos.z * 39.4 / 12)
                self.position_alt.plot(t, altitude(self.pos) * 39.4 / 12)
                self.position_total.plot(t, sqrt((self.pos.x * 39.4 / 12) ** 2 + (self.pos.y * 39.4 / 12) ** 2))
            if FreeRocket.acceleration_graph_enable:
                self.acceleration_x.plot(t, f_net.x / self.mass * 39.4 / 12 / 32.174)
                self.acceleration_y.plot(t, f_net.y / self.mass * 39.4 / 12 / 32.174)
                self.acceleration_z.plot(t, f_net.z / self.mass * 39.4 / 12 / 32.174)
                self.acceleration_total.plot(t, f_net.mag / self.mass * 39.4 / 12 / 32.174)
                self.acceleration_g.plot(t, f_grav.mag / self.mass * 39.4 / 12 / 32.174)
            if FreeRocket.rotation_rate_graph_enable:
                self.rotation_rate_yaw.plot(t, self.drot.dot(self.yaw_axis) * 180 / np.pi)
                self.rotation_rate_pitch.plot(t, self.drot.dot(self.pitch_axis) * 180 / np.pi)
                self.rotation_rate_roll.plot(t, self.drot.dot(self.roll_axis) * 180 / np.pi)
            if FreeRocket.aoa_graph_enable:
                self.aoa.plot(t, alpha / np.pi * 180)
            if FreeRocket.force_graph_enable:
                self.drag_plot.plot(t, f_drag.mag / 4.448)
                self.thrust_plot.plot(t, f_thrust.mag / 4.448)
                self.gravity_plot.plot(t, f_grav.mag / 4.448)
            if FreeRocket.mass_graph_enable:
                self.mass_plot.plot(t, self.mass * 2.204)
            if FreeRocket.fin_aoa_graph_enable:
                self.fin.aoa_plot(t, self.rot, airflow, self.cg)
            if FreeRocket.rcs_graph_enable:
                self.rcs_x.plot(t, f_rcs.dot(self.pitch_axis))
                self.rcs_y.plot(t, f_rcs.dot(self.yaw_axis))
                self.rcs_z.plot(t, f_rcs.dot(self.roll_axis))

        # Update flight report variables
        self.duration = t
        if f_thrust.mag > 0:
            self.drag_loss += f_drag.mag / self.mass * dt
            self.grav_loss += f_grav.mag / self.mass * dt
            self.cosine_loss += (f_thrust.mag - f_thrust.dot(vec(0, 0, 1))) / self.mass * dt
        if altitude(self.pos) > self.z_max:
            self.z_max = altitude(self.pos)
            self.z_max_time = t
        if self.v.mag > self.v_max:
            self.v_max = self.v.mag
            self.v_max_time = t
        a = f_net.mag / self.mass
        if a > self.a_max:
            self.a_max = a
            self.g_max = a / abs(g_0)
            self.a_max_time = t
        if altitude(self.pos) <= 10:
            self.duration = t
            self.v_ground_hit = self.v.mag
        q = 1 / 2 * rho(self.pos) * self.v.mag ** 2
        if self.q_max < q:
            self.q_max = q
            self.q_max_time = t
            self.q_max_speed = self.v.mag
            self.q_max_alt = altitude(self.pos)
        if self.mach_max < mach_n:
            self.mach_max = mach_n
            self.mach_max_time = t
            self.mach_max_speed = self.v.mag
            self.mach_max_altitude = altitude(self.pos)
        # end of simulation method

    def flight_report(self):
        print(f"\n{self.name} Flight Report -------------\n")
        print(f"Apogee: {d2(self.z_max * 39.37 / 12)}ft at T+{d2(self.z_max_time)}s")
        print(f"Maximum speed: {d2(self.v_max * 39.37 / 12 / 5280 * 3600)}mph at T+{d2(self.v_max_time)}s")
        print(f"Maximum acceleration: {d2(self.g_max)}G at T+{d2(self.a_max_time)}s")
        print(f"Flight Duration: {d2(self.duration)}s")
        if self.pos.z < 10:
            print(
                f"Ground hit velocity: {d2(self.v_ground_hit * 39.37 / 12)}ft/s, {d2(self.pos.mag * 39.4 / 12)}ft downrange")
        print(
            f"Maximum dynamic pressure: {d2(self.q_max * 4.448 / 39.37 ** 2)}psig, at {d2(self.q_max_alt * 39.37 / 12)}ft, at {d2(self.q_max_speed * 39.4 / 12)}ft/s, at T+{d2(self.q_max_time)}s")
        print(
            f"Maximum Mach number: {d2(self.mach_max)}M at T+{d2(self.mach_max_time)}s at {d2(self.mach_max_speed * 39.4 / 12)}ft/s at {d2(self.mach_max_altitude * 39.4 / 12)}ft altitude\n")
        print(
            f"Drag losses: {d2(self.drag_loss)} m/s, Gravity losses: {d2(self.grav_loss)} m/s, Cosine losses: {d2(self.cosine_loss)} m/s")

        # end of flight report method

    def inherit(self, parent):
        self.v = parent.v
        self.p = self.v * self.mass
        self.drot = parent.drot
        self.L = vector(
            self.I_0.x * self.drot.x,
            self.I_0.y * self.drot.y,
            self.I_0.z * self.drot.z
        )
        self.pos = parent.pos
        self.roll_axis = parent.roll_axis
        self.yaw_axis = parent.yaw_axis
        self.pitch_axis = parent.pitch_axis

        # end of staging inheritance method

    # end of class def


# Thrust curve data for a static fire test of the flight motor of author's rocket, Alpha Phoenix
I435_time_points = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.65, 1.8, 1.95, 2.1, 2.25, 2.4]
I435_thrust_points = [409.4694, 447.336, 451.1619, 450.9657, 369.0522, 93.0969, 76.8123, 69.7491, 55.2303, 36.5913,
                      27.6642, 21.4839, 17.9523, 13.8321, 5.4936, 2.7468, 2.1582]


# Rocket thrust = mass flow rate * exhaust velocity + (exhaust pressure - ambient pressure) * exit area

# thrust interpolation for 38mm 68-16-16 AP-Al motor test fired 3/16/2023
def I435(t, P):
    if t <= 2.4:
        return np.interp(t, I435_time_points, I435_thrust_points) + (101325 - P) * (2 ** 2 / 4 * pi / 39.37 ** 2)
    else:
        return 0


# thrust function approximation for I240 motor
def I240(t, P):
    return -77 * (t - 0.2) ** 2 + 300 + (101325 - P) * (2 ** 2 / 4 * pi / 39.37 ** 2)


# thrust function for LR-101 pressure-fed kerolox engine

def LR101(t, P):
    if t < 8:
        return 1000 * 4.448 + (101325 - P) * (2.85 ** 2 / 4 * pi / 39.37 ** 2)  # lbf * N/lbf
    else:
        return 0


# thrust function for PEAR sustainer engine (kerolox)
def PEARSustainerEngine(t, P):
    if t < 41.754:
        thrust = np.polyval([0.000127581, -0.0164375, 0.92882, -36.2125, 1998.08], t)
        Pc = np.polyval([5.10325e-06, -0.000657499, 0.0371528, -1.4485, 79.9233], t)
        Pe = Pc / 8
        return thrust * 4.448 + (Pe * 6894.76 - P) * (5.362 ** 2 / 4 * pi / 39.37 ** 2)  # lbf * N/lbf
    else:
        return 0


# thrust function for Sharkshot pressure fed LOX/Kerosene engine
def Hammerhead(t, P):
    if t < 19.4:
        return 1
        # return 4131.6 * 4.448 # N, thrust
    else:
        return 0


def Stationkeeping(t, P):
    if t < 1:
        return 1
    else:
        return 0

    # alpha phoenix fin coefficient of lift function


def cl(aoa: float):
    return 2 * pi * aoa  # using NASA approx for thin subsonic airfoils


# Drag coefficient vs. Reynold's Number for Atlas missile (used as approximation)
cd_M_points = np.array(
    [0.016666666666666666, 0.125, 0.2583333333333333, 0.3833333333333333, 0.5, 0.6, 0.6833333333333333, 0.775, 0.85,
     0.9333333333333333, 1, 1.1083333333333334, 1.1916666666666667, 1.2833333333333332, 1.3333333333333333,
     1.4333333333333333, 1.525, 1.6, 1.6833333333333333, 1.775, 1.8666666666666667, 1.9583333333333333,
     2.0666666666666664, 2.1916666666666664, 2.2916666666666665, 2.408333333333333, 2.55, 2.7, 2.875,
     3.0666666666666664, 3.308333333333333, 3.5416666666666665, 3.8, 4.1, 4.525, 5.016666666666667, 5.575,
     6.108333333333333, 6.583333333333333, 7.016666666666667, 7.566666666666666, 8.058333333333334, 8.6,
     9.108333333333333, 9.775, 10.041666666666666])
cd_points = np.array(
    [0.30000000000000004, 0.28303571428571433, 0.2696428571428572, 0.26428571428571435, 0.26071428571428573,
     0.26875000000000004, 0.28035714285714286, 0.3008928571428572, 0.32946428571428577, 0.3642857142857143,
     0.40267857142857155, 0.4535714285714286, 0.4955357142857144, 0.5312500000000001, 0.5482142857142859,
     0.5535714285714287, 0.5508928571428573, 0.542857142857143, 0.5330357142857144, 0.5205357142857144,
     0.5062500000000001, 0.4848214285714286, 0.461607142857143, 0.4321428571428573, 0.40535714285714297,
     0.3812500000000001, 0.35357142857142865, 0.32678571428571435, 0.3008928571428572, 0.275, 0.25, 0.23214285714285715,
     0.21785714285714286, 0.2080357142857143, 0.2017857142857143, 0.19910714285714287, 0.20357142857142857,
     0.21160714285714288, 0.2205357142857143, 0.23035714285714287, 0.24196428571428574, 0.2517857142857143,
     0.2598214285714286, 0.26160714285714287, 0.26160714285714287, 0.26160714285714287])


def cd_atlas(M: float):
    return np.interp(M, cd_M_points, cd_points)


DummyRCS = dict(fuel_mass=0, cg=vec(0, 0, -1), ct=vec(0, 0, -1), rcg=(0, 0, -1), radius=1, thrust=1, throttle=False)

AlphaPhoenixFins = dict(num_fins=4, center=vec(0, -0.152, 0), pos=vec(0, -0.162, 0), planform=0.005,
                        stall_angle=10 * pi / 180, ac_span=0.05, cl_pass=cl)

FAR_wind = dict(name="FAR", mu=2, sigma=2, angle_mu=pi / 4, angle_sigma=pi / 8, step=100, print_debug=False)
wind_1 = WindProfile(**FAR_wind)

# Alpha Phoenix on I-300
AlphaPhoenix = dict(name="Alpha Phoenix", pos=vec(0, 1, 0), yaw=0, pitch=90 * pi / 180, roll=0, v_0=5, ymi=0.0715,
                    pmi=0.0715, rmi=0.0012, cp=vec(0, -0.152, 0), cd=cd_atlas, A=(2.4 / 2 / 39.4) ** 2 * np.pi,
                    cd_s=1.5, A_s=0.05, main_deploy_alt=150, chute_cd=0.8, chute_A=(32 / 39.4 / 2) ** 2 * np.pi,
                    drogue_cd=0.8, drogue_A=(8 / 39.4 / 2) ** 2 * np.pi, cg=vec(0, -0.142, 0), dry_mass=1.1,
                    fuel_mass=0.220, thrust=I435, t0=0, wind=wind_1, initDebug=True, fin=FinSet(**AlphaPhoenixFins))

# Theseus on LR-101
TheseusFins = dict(num_fins=4, center=vec(0, 0, -5), pos=vec(0, 0, -5.2), planform=0.0258, stall_angle=10 * pi / 180,
                   ac_span=0.165, cl_pass=cl)

launcher = vec(0, 0, 1)
launcher = rotate(launcher, 2 * pi / 180, vec(1, 0, 0))
launcher = rotate(launcher, 2 * pi / 180, vec(0, -1, 0))
Theseus = dict(name="Theseus", pos=vec(0, 1, 0), roll_axis=launcher, yaw_axis=rotate(launcher, pi / 2, vec(1, 0, 0)),
               v=vec(0, 0, 5), I_0=vec(2970, 2970, 10), cg=vec(0, -237 / 39.4, 0), cp=vec(0, -250 / 39.4, 0),
               cd=cd_atlas, A=(8 / 2 / 39.4) ** 2 * np.pi, cd_s=1, A_s=0.5, main_deploy_alt=350, chute_cd=0.8,
               chute_A=(120 / 39.4 / 2) ** 2 * np.pi, drogue_cd=0.8, drogue_A=(60 / 39.4 / 2) ** 2 * np.pi * 2,
               dry_mass=(160) / 2.204, fuel_mass=40.5 / 2.204, thrust=LR101, t0=0, wind=wind_1, initDebug=False,
               fin=FinSet(**TheseusFins), rcs=ReactionControlSystem(**DummyRCS))
# SharkShot on Hammerhead
SharkShotFins = dict(num_fins=4, center=vec(0, -163 / 39.4, 0), pos=vec(0, 0, -163 / 39.4), planform=0.0258,
                     stall_angle=10 * pi / 180, ac_span=0.25, cl_pass=cl)

SharkShotRCS = dict(fuel_mass=5, cg=vec(0, 0, -50 / 39.4), ct=vec(0, 0, -50 / 39.4), rcg=(0, 0, -181 / 39.37),
                    radius=5 / 39.4, thrust=100, throttle=False)

SharkShot = dict(name="SharkShot", pos=vec(0, 0, 1), roll_axis=vec(0, 0, 1), yaw_axis=vec(0, 1, 0), v=vec(0, 0, 5),
                 I_0=vec(5.63e6 / 2.2 / 39.37 ** 2, 5.63e6 / 2.2 / 39.37 ** 2, 5.6e4 / 2.2 / 39.37 ** 2),
                 cg=vec(0, 0, -181.3), cp=vec(0, 0, -240), cd=cd_atlas, A=(12 / 2 / 39.4) ** 2 * np.pi, cd_s=1, A_s=2,
                 main_deploy_alt=500, chute_cd=0.8, chute_A=(300 / 39.4 / 2) ** 2 * np.pi, drogue_cd=0.8, drogue_A=0.5,
                 dry_mass=288.6 / 2.204, fuel_mass=400 / 2.204, thrust=Hammerhead, t0=0, wind=wind_1, initDebug=True,
                 fin=FinSet(**SharkShotFins), rcs=ReactionControlSystem(**SharkShotRCS))

# International Space Station
ISS = dict(name="ISS", pos=vec(0, 0, 4.13e5), roll_axis=vec(1, 0, 0), yaw_axis=vec(0, 0, 1), v=vec(7.67e3, 0, 0),
           I_0=vec(1e4, 1e4, 1e4), cg=vec(0, 0, -5), cp=vec(0, 0, 0), cd=cd_atlas, A=50, cd_s=1, A_s=50,
           main_deploy_alt=0, chute_cd=0, chute_A=0, drogue_cd=0, drogue_A=0, dry_mass=1e1, fuel_mass=1,
           thrust=Stationkeeping, t0=0, wind=wind_1, initDebug=False, fin=FinSet(**SharkShotFins),
           rcs=ReactionControlSystem(**SharkShotRCS))

# Beginning of actual program execution

# Defining vehicles and their properties
# This is a slightly modified Theseus which is used as a booster for space shot upper stage.
Theseus.update(name="PEARBooster", dry_mass=325.1 / 2.204, fuel_mass=40 / 2.204, initDebug=True)
# this is the space shot upper stage.
Theseus.update(name="PEARSustainer", I_0=vec(4410, 4410, 30), A=(10 / 2 / 39.4) ** 2 * np.pi,
               cg=vec(0, 0, -234.7 / 39.4), cp=vec(0, 0, -250 / 39.4), dry_mass=234 / 2.204, fuel_mass=220 / 2.204,
               thrust=PEARSustainerEngine, initDebug=True)

paused = False


def pause():
    global paused
    paused = not paused


pause = button(bind=pause, text="Pause")

time = 0
dtime = 1 / 50


def timestep():
    global dtime
    old = dtime
    dtime = float(input("Time step (s): "))
    if dtime == None:
        dtime = old


dtimebutton = button(bind=timestep, text="Change dt")

apogee_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

i = 0
while i <= 9:
    time = 0
    dtime = 1 / 50
    wind_1 = WindProfile(**FAR_wind)
    sustainer = FreeRocket(**Theseus)
    sustainer.rcs.setReference(vec(0, 0, 1), vec(0, 1, 0))
    sustainer.rcs.setPID(0, 0, 0, 0)
    sustainer.rcs.setProfile(1, 1, np.pi / 180)
    sustainer.wind = wind_1
    if i == 0:
        sustainer.graph_enable = True
    else:
        sustainer.graph_enable = False
    print(f"Time step: {dtime:.3f}")
    print("----BEGIN SIMULATION----")
    # While still on launch rail
    while altitude(sustainer.pos) < 21:
        sustainer.simulate(time, dtime)
        sustainer.roll_axis = launcher
        sustainer.drot = vec(0, 0, 0)
        sustainer.p = sustainer.p.mag * launcher.hat
        time += dtime
        while paused:
            sleep(1)

    dtime = 1 / 800
    print(
        f"Rocket cleared launch rail at {sustainer.v.mag * 39.37 / 12:.2f} ft/s at T+{time:.2f}s.  Time step changed to {dtime:.3f}s")
    n = 1

    while time < sustainer.t1:
        sustainer.simulate(time, dtime)
        time += dtime
        if i == 0 and n % 100 == 0:
            sustainer.graph_enable = True
        else:
            sustainer.graph_enable = False
        n += 1
        while paused:
            sleep(1)

    dtime = 0.05
    print(f"Engine burnout at {sustainer.v.mag * 39.37 / 12:.2f} ft/s at T+{time:.2f}s.  dt adjusted to {dtime:.3f}s")
    while altitude(sustainer.pos) > 0:
        sustainer.simulate(time, dtime)
        time += dtime
        if i == 0 and n % 10 == 0:
            sustainer.graph_enable = True
        else:
            sustainer.graph_enable = False
        n += 1
        while paused:
            sleep(1)

    apogee_set[i] = ceil(sustainer.z_max * 39.37 / 12)
    sustainer.flight_report()
    print(f"Apogees: {apogee_set}")
    i += 1

print("-----END MONTE CARLO SIMULATION-----\n")
print(f"Apogees: {apogee_set}")
print(f"Mean apogee: {np.average(apogee_set)}")
print(f"Stdev apogee: {np.std(apogee_set)}")
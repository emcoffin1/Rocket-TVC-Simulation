import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from windProfiles import WindForce
from ControlSystem import Controller
class Rocket:
    def __init__(self):
        # Environment Constants
        self.gravity = 32.172   # ft/s2
        self.refAlt = 27887     # ft
        self.density = 0.0765

        # Vehicle Constants
        self.length = 18        # ft
        self.diameter = 0.83    # ft = 10 in
        self.wetMass = 392.182  # lbm
        self.dryMass = 170.890  # lbm
        self.liquidMass = self.wetMass - self.dryMass
        self.dragCoeff = 0.35
        self.liftCoeff = 1.2
        self.momInteriaInit = 7800 # lbm-ft2 , assumption
        self.momInteriaFinal = 4900
        self.momInteriaRoll = 1190
        self.cgI = 11.2
        self.cgF = 11.7
        self.cp = 13.38

        # Engine Constants
        self.burnTime = 23.7
        self.isp = 300          # second, estimate
        self.thrust = 2000      # lbf
        self.massFlowRate = 9.289 # lbm/s

    def getInertialMatrix(self, time):
        if time < self.burnTime:
            ratio = time / self.burnTime

            py_I = (ratio - 1) * self.momInteriaInit + ratio * self.momInteriaInit
            return np.diag([self.momInteriaRoll, py_I, py_I])
        else:
            return np.diag([self.momInteriaRoll, self.momInteriaFinal, self.momInteriaFinal])


    def getDensity(self, alt):
        return self.density * math.exp(-alt / self.refAlt)

    def getMass(self, time: float):
        if (time < self.burnTime):
            return self.wetMass - self.massFlowRate * time
        else:
            return self.dryMass

    def getMassFlowRate(self, time: float):
        if time < self.burnTime:
            return self.massFlowRate
        else:
            return 0.0

    def getThrust(self, time: float, alt:float):
        if time < self.burnTime:
            pres = self.ambient_pressure_lbf_ft2(alt)

            thrust = np.polyval([0.000127581, -0.0164375, 0.92882, -36.2125, 1998.08], time)
            Pc = np.polyval([5.10325e-06, -0.000667499, 0.0371528, -1.4485, 79.9233], time)
            Pe = Pc / 8
            return thrust + ((Pe * 6894.76 - pres) * (5.362 ** 2 / 4 * np.pi / 39.37 ** 2)) / 4.448  # output in lbf
        else:
            return 0

    def getCPOffset(self, time):
        if time < self.burnTime:
            ratio = time / self.burnTime

            cg = (1 - ratio) * self.cgI + ratio * self.cgF
            offset = cg - self.cp

            return offset
        else:
            return self.cgF - self.cp

    def ambient_pressure_lbf_ft2(self, alt_ft: float):
        P0 = 2116.22  # lbf/ft²
        T0 = 518.67  # °R
        L = 0.003566  # °R/ft
        if alt_ft < 36089:  # troposphere model
            return P0 * (1 - L * alt_ft / T0) ** 5.25588
        else:
            # Stratosphere (~11km to ~20km) approximation
            # Isothermal layer (T ≈ 390.0 °R)
            P1 = 472.68  # pressure at 36089 ft in lbf/ft²
            return P1 * np.exp(-0.00004806 * (alt_ft - 36089))


class FlightSim:
    def __init__(self, vm, dm, alt_start):
        self.rocket = Rocket()
        self.windSim = WindForce(vm=vm, dm=dm, alt_start=alt_start)

        self.controller = Controller(kp=20.0, kd=20.0)

        self.state = np.array([
            0.0, 0.0, 0.0,      # pos
            0.0, 0.0, 0.0,      # vel
            0.0, 0.0, 0.0, 1.0, # quats
            0.0, 0.0, 0.0,      # ang vel
            self.rocket.wetMass # mass
        ])

        self.runSim()

    def unpackedState(self, state):
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]
        mass = state[13]
        return pos, vel, quat, omega, mass

    def rocketForces(self, time, pos, vel, quat, mass):
        grav = self.rocket.gravity
        alt = pos[2]
        rho = self.rocket.getDensity(alt)



        # -- Engine Thrust -- #
        thrust = np.array([0.0, 0.0, self.rocket.getThrust(time, alt)])
        thrust_inert = R.from_quat(quat).apply(thrust)

        # -- Drag Force -- #
        v_mag = np.linalg.norm(vel)
        if v_mag > 0:
            A = math.pi * (self.rocket.diameter / 2) ** 2
            drag = -0.5 * rho * v_mag**2 * self.rocket.dragCoeff * A * (vel / v_mag)
        else:
            drag = np.zeros(3)

        # -- Gravity Force -- #
        gravity = np.array([0.0, 0.0, -grav])  # gravity pulls down

        # -- Wind Thrust -- #
        wind = self.windSim.checkWind(alt, 0.83*18, self.rocket.liftCoeff, rho)

        # -- Resultant Force -- #
        force = thrust_inert + drag + gravity # + wind

        # -- Resultant Acceleration -- #
        accel = force / mass

        return accel, wind

    def quaternionDerivative(self, quat, omega):
        qx, qy, qz, qw = quat
        wx, wy, wz = omega

        dqdt = 0.5 * np.array([
            qw * wx + qy * wz - qz * wy,
            qw * wy + qz * wx - qx * wz,
            qw * wz + qx * wy - qy * wx,
            -qx * wx - qy * wy - qz * wz,
        ])
        return dqdt

    def angularAcceleration(self, time, quat, omega, wind_force):
        # Get full moment of inertia matrix, function of time
        I = self.rocket.getInertialMatrix(time)

        # Get CP offset, function of time
        cp_offset = self.rocket.getCPOffset(time)
        # print(cp_offset)

        # Rotate wind force into body frame
        wind_body = R.from_quat(quat).inv().apply(wind_force)

        # if np.linalg.norm(wind_body) != 0:
        #     print(wind_body)

        # Assume wind acts at cp
        cp_vector = np.array([cp_offset, 0.0, 0.0])
        torque_body = np.cross(cp_vector, wind_body)


        # Eulers rotational equation
        I_inv = np.linalg.inv(I)
        gyro_term = np.cross(omega, I @ omega)
        #alpha = I_inv @ (torque_body - gyro_term)

        tvc_torque = self.controller.control_torque(quat=quat, omega=omega, target_quat=[0,0,0,1])

        total_torque = torque_body + tvc_torque
        alpha = I_inv @ (total_torque - np.cross(omega, I @ omega))

        # if np.linalg.norm(alpha) != 0:
        #     print(alpha)

        return alpha

    def massChangeRate(self, time):
        return self.rocket.getMassFlowRate(time)

    def derivatives(self, time, state):
        pos, vel, quat, omega, mass = self.unpackedState(state)

        accel,wind = self.rocketForces(time, pos, vel, quat, mass)
        dquat = self.quaternionDerivative(quat, omega)
        domega = self.angularAcceleration(time, quat, omega, wind)
        dmass = self.massChangeRate(time)

        return np.concatenate([vel, accel, dquat, domega, [dmass]])


    def rk4(self, derivs, state, t, dt):
        k1 = derivs(t, state)
        k2 = derivs(t + dt/2, state + dt/2 * k1)
        k3 = derivs(t + dt/2, state + dt/2 * k2)
        k4 = derivs(t + dt, state + dt * k3)
        return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def runSim(self):
        dt = 0.1
        t = 0.0
        state = self.state.copy()

        times = [t]
        states = [state.copy()]


        while state[2] >= 0:
            state = self.rk4(self.derivatives, state, t, dt)

            # Normalize quaternion
            state[6:10] /= np.linalg.norm(state[6:10])

            t += dt
            times.append(t)
            states.append(state.copy())


            if state[2] <= 0 and t > 5.0:  # prevent early false-positive on ground
                break

        return np.array(times), np.array(states)



import numpy as np
import VehicleModels
import EnvironmentalModels
import matplotlib.pyplot as plt

if __name__ == "__main__":

    rocket = VehicleModels.Rocket()
    state = rocket.state.copy()

    dt = 0.1  # timestep in seconds
    t_final = 30.0
    steps = int(t_final / dt)

    # Logging Containers
    time_log = []
    pos_log = []
    vel_log = []
    quat_log = []
    aoa_log = []
    beta_log = []
    mass_log = []
    force_log = []

    for _ in range(steps):
        # Unpack state
        a,b,c,d = rocket.getTotalForce(state)
        pos, vel, quat, omega, mass, time, aoa, beta = VehicleModels.unpackStates(state)

        force_log.append([a, b, c, d])
        # Log data
        time_log.append(time)
        pos_log.append(pos)
        vel_log.append(vel)
        quat_log.append(quat)
        aoa_log.append(aoa)
        beta_log.append(beta)
        mass_log.append(mass)

        # RK4 integration
        state = VehicleModels.rk4_step(rocket, state, dt)

    time_log = np.array(time_log)
    pos_log = np.array(pos_log)
    vel_log = np.array(vel_log)
    quat_log = np.array(quat_log)
    aoa_log = np.array(aoa_log)
    beta_log = np.array(beta_log)
    mass_log = np.array(mass_log)
    force_log = np.array(force_log)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(time_log, force_log[:, 0])
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(time_log, force_log[:, 1])
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(time_log, vel_log[:, 2])
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# eng = VehicleModels.RocketEngine()
# air = EnvironmentalModels.AirProfile()
# h = []
# g = []
# t = 0
# thrust = []
# times = []
# dt = 0.1
# vals = 30 / dt
# for x in range(int(vals)):
#     p = air.getStaticPressure(x)
#     T = eng.getThrust(t, p)
#     thrust.append(np.linalg.norm(T))
#     h.append(x)
#     times.append(t)
#
#     t += dt
#
#
# plt.plot(times,thrust)
# plt.show()


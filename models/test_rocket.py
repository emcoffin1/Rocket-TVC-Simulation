import numpy as np
import VehicleModels
import EnvironmentalModels
import matplotlib.pyplot as plt

if __name__ == "__main__":

    rocket = VehicleModels.Rocket()
    state = rocket.state.copy()

    dt = 0.05  # timestep in seconds
    t_final = 3000.0
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
    density_log = []
    dynamicpres_log = []

    for _ in range(steps):
        # Unpack state
        a, b, c, d, rho, q = rocket.getTotalForce(state)
        pos, vel, quat, omega, mass, time, aoa, beta = VehicleModels.unpackStates(state)

        force_log.append(a+b+c+d)
        dynamicpres_log.append(q)
        # Log data
        time_log.append(time)
        pos_log.append(pos)
        vel_log.append(vel)
        quat_log.append(quat)
        aoa_log.append(aoa)
        beta_log.append(beta)
        mass_log.append(mass)
        density_log.append(rho)

        if pos[2] < 0 and time > 5:
            break

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
    density_log = np.array(density_log)
    dynamicpres_log = np.array(dynamicpres_log)

    max_q = 0
    max_q_index = 0

    # Assumes you have logs of altitude and dynamic pressure over time
    for i in range(1, len(pos_log[:, 2])):
        if pos_log[i, 2] < pos_log[i - 1, 2]:  # Descent started
            break
        if dynamicpres_log[i] > max_q:
            max_q = dynamicpres_log[i]
            max_q_index = i

    print(
        f"Max Q (ascent only): {max_q:.2f} Pa at t = {time_log[max_q_index]:.2f} s, alt = {pos_log[max_q_index, 2]:.2f} m")

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(time_log, force_log)
    plt.xlabel("Forces")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time_log, pos_log)
    plt.xlabel("Alt")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time_log, vel_log)
    plt.xlabel("Vel")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time_log, dynamicpres_log)
    plt.xlabel("Density")
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


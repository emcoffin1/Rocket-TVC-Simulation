import numpy as np
import VehicleModels
import EnvironmentalModels
import matplotlib.pyplot as plt


def print_results(time, pos, vel, q, rocket):

    print("=" * 60)
    print("FLIGHT ENDED")

    # Burntime
    max_th_i = np.argmax(rocket.thrust)
    print(f"Burn Time:      {rocket.burntime}s")
    print(f"Average Thrust: {np.average(rocket.thrust):.2f}N")
    print(f"Max Thrust:     {np.max(rocket.thrust):.2f}N at Time: {time[max_th_i]:.2f}s")


    # Altitude
    max_alt_i = np.argmax(pos[:, 2])  # correct array passed in
    max_alt = pos[max_alt_i, 2]
    print(f"Max Altitude:   {max_alt:.2f} m at Time: {time[max_alt_i]:.2f}s")

    # Velocity
    vel_norm = np.linalg.norm(vel, axis=1)
    max_vel = np.max(vel_norm)
    max_vel_time_i = np.argmax(vel_norm)
    print(f"Max Velocity:   {max_vel:.2f} m/s at Time: {time[max_vel_time_i]:.2f}s")

    # Determine max pressure
    max_q = 0
    max_q_index = 0
    for i in range(1, len(pos[:, 2])):
        if pos[i, 2] < pos[i - 1, 2]:  # Descent started
            break
        if q[i] > max_q:
            max_q = q[i]
            max_q_index = i

    print(f"Max Pressure:   {max_q:.2f} Pa at Time: {time[max_q_index]:.2f} s at Altitude: {pos[max_q_index, 2]:.2f} m")



if __name__ == "__main__":

    rocket = VehicleModels.Rocket()
    state = rocket.state.copy()

    dt = 0.05  # timestep in seconds
    t_final = 6000.0
    steps = int(t_final / dt)

    # Logging Containers
    time_log = []
    pos_log = []
    vel_log = []
    quat_log = []
    aoa_log = []
    beta_log = []
    mass_log = []
    thrust_log = []
    drag_log = []
    density_log = []
    dynamicpress_log = []

    for _ in range(steps):
        # Unpack state
        a, b, c, d, rho, q, _ = rocket.getTotalForce(state, dt, side_effect=False)
        pos, vel, quat, omega, mass, time, aoa, beta = VehicleModels.unpackStates(state)
        # print(time, ", ", a[2])
        thrust_log.append(b)
        drag_log.append(b)
        dynamicpress_log.append(q)
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
    thrust_log = np.array(thrust_log)
    drag_log = np.array(drag_log)
    density_log = np.array(density_log)
    dynamicpress_log = np.array(dynamicpress_log)

    print_results(time=time_log, pos=pos_log, vel=vel_log, q=dynamicpress_log, rocket=rocket)

    print("--- EXTRA LOG ---")
    # for i,t in zip(rocket.thrust, time_log):
    #     print(t, i)
    #
    # print(f"max th: {np.max(rocket.thrust)}")
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(time_log, thrust_log, label="Thrust")
    # # plt.plot(rocket.drags, label="Drag")
    # plt.xlabel("Forces (N)")
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(time_log, pos_log*3)
    # plt.xlabel("Alt (m)")
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(time_log, vel_log)
    # plt.xlabel("Velocity (m/s)")
    # plt.grid(True)

    # plt.subplot(2, 2, 4)

    plt.plot(rocket.alt, rocket.mach)
    # plt.xlabel("Mass Change")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


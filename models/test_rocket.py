import numpy as np
import VehicleModels
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R


def print_results(time, pos, vel, q, rocket):

    print("=" * 60)
    print("FLIGHT ENDED")

    # Burntime
    try:
        max_th = max(rocket.thrust_magn)
        max_th_i = rocket.thrust_magn.index(max_th)
        print(f"Burn Time:      {rocket.burntime:.2f}s")
        print(f"Average Thrust: {(sum(rocket.thrust_magn) / len(rocket.thrust_magn)):.2f}N")
        print(f"Max Thrust:     {max_th_i:.2f}N at Time: {time[max_th_i]:.2f}s")
    except Exception:
        print("[FAULT] Thrust values faulty")

    # Altitude
    try:
        max_alt_i = np.argmax(pos[:, 2])  # correct array passed in
        max_alt = pos[max_alt_i, 2]
        print(f"Max Altitude:   {max_alt:.2f} m at Time: {time[max_alt_i]:.2f}s")
    except Exception:
        print("[FAULT] Attitude values faulty")

    # Velocity
    try:
        vel_norm = np.linalg.norm(vel, axis=1)
        max_vel = np.max(vel_norm)
        max_vel_time_i = np.argmax(vel_norm)

        max_mach = np.max(np.array(rocket.mach))
        max_mach_i = np.argmax(np.argmax(np.array(rocket.mach)))
        print(f"Max Velocity:   {max_vel:.2f} m/s at Time: {time[max_vel_time_i]:.2f}s")
        print(f"Max Mach:       {max_mach:.2f}    at Time: {time[max_mach_i]:.2f}s")
    except Exception:
        print("[FAULT] Velocity values faulty")

    # Determine max pressure
    try:
        max_q = 0
        max_q_index = 0
        for i in range(1, len(pos[:, 2])):
            if pos[i, 2] < pos[i - 1, 2]:  # Descent started
                break
            if q[i] > max_q:
                max_q = q[i]
                max_q_index = i

        print(f"Max Pressure:   {max_q:.2f} Pa at Time: {time[max_q_index]:.2f} s at Altitude: {pos[max_q_index, 2]:.2f} m")
    except Exception:
        print("[FAULT] MAX Q Values faulty")

    print("=" * 60)


def attitude_polar_plot(pitch, yaw, gimbals):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})

    # -- First Plot: Pitch vs Yaw --
    axs[0].plot(yaw, pitch, label="Orientation")
    axs[0].set_theta_zero_location("N")
    axs[0].set_theta_direction(-1)
    axs[0].set_rlabel_position(90)
    axs[0].set_title("Rocket Pitch vs Yaw")
    axs[0].legend()

    axs[1].plot(gimbals[:, 0], gimbals[:, 1], label="Gimbal", color='Red')
    axs[1].set_theta_zero_location("N")
    axs[1].set_theta_direction(-1)
    axs[1].set_rlabel_position(90)
    axs[1].set_title("Gimbal Orientation")
    axs[1].legend()

    plt.tight_layout()
    plt.show()



def plot_3d(pos_log):
    # Load once
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(PROJECT_ROOT, "TVC/cubic_sweep_profile.csv")
    df = pd.read_csv(filename)

    # Build interpolators
    interp_x = interp1d(df['z'], df['x'], kind='linear', fill_value='extrapolate')
    interp_y = interp1d(df['z'], df['y'], kind='linear', fill_value='extrapolate')

    # Extract x, y, z
    x_vals = pos_log[:, 0]
    y_vals = pos_log[:, 1]
    z_vals = pos_log[:, 2]

    exp = []
    for z in z_vals:
        exp.append([interp_x(z), interp_y(z), z])
    exp_log = np.array(exp)

    # Extract x, y, z
    x_exp = exp_log[:, 0]
    y_exp = exp_log[:, 1]
    z_exp = exp_log[:, 2]

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals, y_vals, z_vals, label='Trajectory Path')
    ax.plot(x_exp, y_exp, z_exp, label="Expected Path")
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z (Altitude) [m]')
    ax.set_title('3D Trajectory from np.array')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    # Ensure equal scaling
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    # Find global min and max range
    all_min = min(x_min, y_min)
    all_max = max(x_max, y_max)

    # Force equal aspect on X and Y by syncing their limits
    ax.set_xlim(all_min, all_max)
    ax.set_ylim(all_min, all_max)

    plt.show()

if __name__ == "__main__":

    rocket              = VehicleModels.Rocket()
    state               = rocket.state.copy()

    dt                  = 0.01  # timestep in seconds
    t_final             = 6000.0
    steps               = int(t_final / dt)

    # Logging Containers
    time_log            = []
    pos_log             = []
    vel_log             = []
    omega_log           = []
    quat_log            = []
    aoa_log             = []
    beta_log            = []
    mass_log            = []
    thrust_log          = []
    drag_log            = []
    density_log         = []
    dynamic_press_log   = []
    actual_pitch_log    = []
    target_pitch_log    = []
    pitch_error_log     = []
    trajectory_log      = []

    for _ in range(steps):
        # Unpack state
        thrust_body, drag_body, coriolis_body, lift_body,total_global = rocket.getTotalForce(state, dt, side_effect=False)
        pos, vel, quat, omega, mass, time, aoa, beta, gimbals = VehicleModels.unpackStates(state)


        if not rocket.engine.combustion_chamber.active:
            print(f"ALT: {pos[2]}")
            break

        if pos[2] < 1e-5 and time > 5:
            print("ROCKET IMPACT")
            print(f"TIME: {time:.2f}")
            break

        # RK4 integration
        state = VehicleModels.rk4_step(rocket, state, dt)

        time_log.append(time)
        pos_log.append(pos)
        vel_log.append(vel)
        omega_log.append(omega)
        quat_log.append(quat)
        dynamic_press_log.append(rocket.dynamic_pres[-1])

    # -- UPDATE LOGS TO ARRAYS -- #
    pos_log             = np.array(pos_log)
    vel_log             = np.array(vel_log)
    quat_log            = np.array(quat_log)
    omega_log           = np.array(omega_log)
    pos_error           = np.array(rocket.quaternion.pos_error)
    torque_command      = np.array(rocket.torque_cmd)
    torque_actual       = np.array(rocket.torque_act)
    gimbals             = np.array(rocket.tvc.gimbal_log)

    # -- RANDOM LOGS -- #
    pitch               = rocket.pitchXZ
    yaw                 = rocket.yawYZ
    mach                = rocket.mach
    reynolds            = rocket.reynolds
    viscosity           = rocket.viscosity
    dynamic_press_log   = rocket.dynamic_pres



    print_results(time=time_log, pos=pos_log, vel=vel_log, q=dynamic_press_log, rocket=rocket)

    plt.plot(time_log, torque_actual, label="ACT")
    plt.plot(time_log, torque_command, label="CMD")
    plt.title("TORQUE")
    plt.legend()
    plt.grid(True)
    plt.show()

    attitude_polar_plot(pitch=pitch, yaw=yaw, gimbals=gimbals)
    plot_3d(pos_log=pos_log)


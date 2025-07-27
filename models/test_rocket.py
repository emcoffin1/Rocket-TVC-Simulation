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
    # max_th_i = np.argmax(rocket.thrust)
    print(rocket.burntime)
    print(f"Burn Time:      {rocket.burntime:.2f}s")
    print(f"Average Thrust: {np.average(rocket.thrust):.2f}N")
    # print(f"Max Thrust:     {np.max(rocket.thrust):.2f}N at Time: {time[max_th_i]:.2f}s")

    # Altitude
    max_alt_i = np.argmax(pos[:, 2])  # correct array passed in
    max_alt = pos[max_alt_i, 2]
    print(f"Max Altitude:   {max_alt:.2f} m at Time: {time[max_alt_i]:.2f}s")

    # Velocity
    vel_norm = np.linalg.norm(vel, axis=1)
    max_vel = np.max(vel_norm)
    max_vel_time_i = np.argmax(vel_norm)

    max_mach = np.max(np.array(rocket.mach))
    max_mach_i = np.argmax(np.argmax(np.array(rocket.mach)))
    print(f"Max Velocity:   {max_vel:.2f} m/s at Time: {time[max_vel_time_i]:.2f}s")
    print(f"Max Mach:       {max_mach:.2f}    at Time: {time[max_mach_i]:.2f}s")

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


def body_pitch_from_quat(quat: np.ndarray) -> float:
    """
    Return the signed pitch (deg) about body‑Y, without 180° flips.
    """
    qx, qy, qz, qw = quat
    # Force qw ≥ 0 to avoid sign ambiguity
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    # Rotate body‐forward [0,0,1] into world
    fwd = R.from_quat([qx, qy, qz, qw]).apply([0.0, 0.0, 1.0])
    # Pitch is angle between fwd and world‐Z in the X–Z plane:
    pitch_rad = np.arctan2(fwd[0], fwd[2])
    return np.degrees(pitch_rad)

if __name__ == "__main__":

    rocket = VehicleModels.Rocket()
    state = rocket.state.copy()

    dt = 0.01  # timestep in seconds
    t_final = 6000.0
    steps = int(t_final / dt)

    # Logging Containers
    time_log = []
    pos_log = []
    vel_log = []
    omega_log = []
    quat_log = []
    aoa_log = []
    beta_log = []
    mass_log = []
    thrust_log = []
    drag_log = []
    density_log = []
    dynamicpress_log = []
    actual_pitch_log = []
    target_pitch_log = []
    pitch_error_log = []
    trajectory_log = []

    for _ in range(steps):
        # Unpack state
        thrust_body, drag_body, coriolis_body, lift_body,total_global = rocket.getTotalForce(state, dt, side_effect=False)
        pos, vel, quat, omega, tvc_quat, mass, time, aoa, beta, gimbals = VehicleModels.unpackStates(state)


        if not rocket.engine.combustion_chamber.active:
            print(f"ALT: {pos[2]}")
            break

        if pos[2] < 0 and time > 5:
            print("ROCKET IMPACT")
            break

        # RK4 integration
        state = VehicleModels.rk4_step(rocket, state, dt)

        time_log.append(time)
        pos_log.append(pos)
        vel_log.append(vel)
        omega_log.append(omega)
        quat_log.append(quat)
        dynamicpress_log.append(rocket.dynamic_pres[-1])

    # -- UPDATE LOGS TO ARRAYS -- #
    time_log = np.array(time_log)
    pos_log = np.array(pos_log)
    vel_log = np.array(vel_log)
    quat_log = np.array(quat_log)
    omega_log = np.array(omega_log)
    dynamicpress_log = np.array(dynamicpress_log)



    print_results(time=time_log, pos=pos_log, vel=vel_log, q=dynamicpress_log, rocket=rocket)

    #
    # print("--- EXTRA LOG ---")
    # plt.subplot(3,1,1)
    # plt.plot(time_log[:-1],rocket.velocity)
    # plt.xlabel("Vel")
    # plt.grid(True)
    #
    # plt.subplot(3,1,2)
    # plt.plot(time_log[:-1], rocket.viscosity)
    # plt.xlabel("Viscosity")
    # plt.grid(True)
    #
    # plt.subplot(3,1,3)
    # plt.plot(time_log[:-1], rocket.reynolds)
    # plt.xlabel("Reynolds")
    # plt.grid(True)
    # q_e = []
    # for x in rocket.tvc.quaternion.error:
    #     # q_e.append([np.rad2deg(x[0]), np.rad2deg(x[1]), np.rad2deg(x[2])])
    #     q_e.append(x)
    # q_e = np.array(q_e)

    # q_e = np.array(rocket.tvc.quaternion.error)
    #
    # plt.subplot(3,1,1)
    # # plt.plot(q_e[:,1], q_e[:,0])
    # plt.plot(time_log, pitch_error_log, label="ERROR")
    # # plt.plot(time_log, target_pitch_log, label="TARGET")
    # # plt.plot(time_log, actual_pitch_log, label="ACTUAL")
    # plt.xlabel("Time")
    # plt.ylabel("Pitch")
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(3,1,2)
    # plt.plot(time_log, omega_log, label="ACTUAL w")
    # plt.plot(time_log[:-2], omega_com, label="COMMANDED w")
    # plt.xlabel("Time")
    # plt.ylabel("Omega")
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(3,1,3)
    # plt.plot(time_log, trajectory_log[:,0], label="X")
    # plt.plot(time_log, trajectory_log[:,1], label="Y")
    # plt.xlabel("Time")
    # plt.ylabel("Trajectory Error")
    # plt.legend()
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()

    # plt.subplot(3,1,1)
    # plt.plot(time_log, vel_log)
    # plt.grid(True)
    #
    # plt.subplot(3,1,2)
    # plt.plot(time_log, quat_log[:,:2])
    # plt.grid(True)
    #
    # plt.subplot(3,1,3)
    # plt.plot(time_log, pos_log)
    # plt.grid(True)


    # rocket.mfr.append(rocket.mfr[-1]+ 0.00000001)
    gimb = np.array(rocket.tvc.gimbal_log)
    quat_er = np.array([q.as_quat() for q in rocket.quaternion.quat_error])
    pos_er = np.array(rocket.quaternion.pos_error)

    plt.subplot(5, 1, 1)
    plt.plot(time_log, gimb)
    plt.ylabel("Deg")
    plt.grid(True)

    plt.subplot(5,1,2)
    plt.plot(time_log, quat_er[:, 0], label="qx")
    plt.plot(time_log, quat_er[:, 1], label="qy")
    plt.plot(time_log, quat_er[:, 2], label="qz")
    plt.plot(time_log, quat_er[:, 3], label="qw")
    plt.ylabel("Quat Error")
    plt.grid(True)

    plt.subplot(5,1,3)
    plt.plot(time_log, pos_er)
    plt.ylabel("Pos Error")
    plt.grid(True)

    plt.subplot(5,1,4)
    plt.plot(time_log, quat_log[:,-1])
    plt.ylabel("QUAT")
    plt.grid(True)

    plt.subplot(5,1,5)
    plt.plot(time_log, omega_log)
    plt.ylabel("OMEGA")
    plt.xlabel("Time [s]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()



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
    plt.show()
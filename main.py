from windProfiles import Graph, WindProfile
import matplotlib.pyplot as plt
from rocketProfile import FlightSim
from ControlSystem import BodePlot, ControlBodePlot
from models.VehicleModels import Rocket


def simTest():
    rocket = Rocket()


def windSims():
    test_profiles = [
        WindProfile(16, 100, 5280), WindProfile(32, 165, 5280), WindProfile(50, 260, 5280),
        WindProfile(32, 165, 13200), WindProfile(50, 200, 13200), WindProfile(65, 100, 13200),
        WindProfile(50, 200, 32736), WindProfile(65, 260, 32736), WindProfile(100, 100, 32736)
    ]

    titles = [
        "Low - Case 1", "Low - Case 2", "Low - Case 3",
        "Med - Case 1", "Med - Case 2", "Med - Case 3",
        "High - Case 1", "High - Case 2", "High - Case 3"
    ]

    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    axs = axs.flatten()

    for i in range(9):
        Graph(test_profiles[i].getDistance(),
              test_profiles[i].getForce(),
              "Dist (ft)",
              "Force (lbf)",
              titles[i],
              axs[i])

    plt.tight_layout()
    plt.show()


def flightSims():
    flight = FlightSim(vm=32, dm=165, alt_start=3000)
    time, state = flight.runSim()
    vel = state[:, 3:6]

    # Get absolute velocity

    # # Graph(time, vel, "Time", "Vel", "Time vs Velocity", 0, True)
    # fig, axs = plt.subplots(1,2)
    # Graph(time, vel[:, 2], "Time", "Vel", "Time v Vel", axs[0])
    # Graph(time, state[:, 2], "Time", "Alt", "Time v Alt", axs[1])

    plt.plot(time, state[:, 8])

    for x in state[:,8]:
        print(x)
    # plt.plot(time, vels)

    plt.tight_layout()
    plt.show()


def bodePlot():
    # bode = BodePlot(kp=1, ki=2, kd=0.5, inertia=7800)1
    # bode.run()

    ControlBodePlot(kp=100000, ki=5000, kd=10000, tau_d=0.01, inertia=7800)


if __name__ == "__main__":
    # -- Unblock different codes to analyze various items -- #
    # windSims()
    flightSims()
    # bodePlot()


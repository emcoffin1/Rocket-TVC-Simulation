import numpy as np

from VehicleModels import Rocket
import matplotlib.pyplot as plt
from EnvironmentalModels import AirProfile

# Setup
t = 0
dt = 0.01
alt = 0
vel = 0

# Logging
times = []
thrust = []
accel = []
alts = []

air = AirProfile()
rocket = Rocket()
burn_steps = int(round(rocket.engine.burnTime / dt))


for _ in range(burn_steps):
    # Get current thrust and mass
    pres = air.getStaticPressure(alt/1000)
    T = rocket.engine.getThrust(t, pres)
    m = rocket.structure.getCurrentMass(t)
    a = T / m #- rocket.gravity  # subtract gravity (assumed g = 9.81 or variable)

    # Integrate motion
    vel += a * dt
    alt += vel * dt

    print(t, pres,T, m, a, vel , alt)

    # Log data
    t += dt
    # print(m)
    times.append(t)
    thrust.append(T)
    accel.append(a)
    alts.append(alt)

# Plotting
plt.figure()
plt.plot(times, alts, label='Altitude')
plt.plot(times, accel, label='Acceleration')
plt.plot(times, thrust, label='Thrust')
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend()
plt.title("Rocket Burn Profile")
plt.show()

# air = AirProfile()
# alt = np.linspace(0, 100, 100000)
# pres = []
# for x in alt:
#     pres.append(air.getStaticPressure(x))
#     #rint(alt, pres)
#
# for h in [10.990, 11.000, 11.010]:
#     print(f"{h} m → {air.getStaticPressure(h)} Pa")
#
# plt.plot(alt, pres)
# plt.grid(True)
# plt.show()
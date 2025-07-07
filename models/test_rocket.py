import numpy as np

from VehicleModels import Rocket
import matplotlib.pyplot as plt
from EnvironmentalModels import AirProfile

# Setup
t = 0
flight_time = 500
dt = 0.01
alt = 0
vel = 0

# Logging
times = []
thrust = []
accel = []
alts = []
density = []
temps = []

air = AirProfile()
rocket = Rocket()


for _ in range(int(round(flight_time / dt))):
    # Get current thrust and mass
    pres = air.getStaticPressure(alt)
    dens = air.getDensity(alt)
    temp = air.getTemperature(alt)
    T = rocket.engine.getThrust(t, pres)
    m = rocket.structure.getCurrentMass(t)
    a = T / m - rocket.gravity  # subtract gravity (assumed g = 9.81 or variable)

    # Integrate motion
    vel += a * dt
    alt += vel * dt

    #print(t, pres,T, m, a, vel , alt)

    # Log data
    t += dt
    # print(m)
    times.append(t)
    thrust.append(T)
    accel.append(a)
    alts.append(alt)
    density.append(dens)
    temps.append(temp)

# Plotting
plt.subplot(2, 1, 1)
plt.title("Rocket Burn Profile")
plt.plot(times, alts, label='Altitude')
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(times, thrust, label='Temp')
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

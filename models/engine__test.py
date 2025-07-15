import numpy as np
import matplotlib.pyplot as plt
from EnvironmentalModels import AirProfile

# alt = np.arange(0, 1000001, 1)
#
air = AirProfile()
#
# pres = []
# for a in alt:
#     air.getCurrentAtmosphere(altitudes_m=a, time=0)
#     pres.append(air.getStaticPressure(a))

# Plot
air.getCurrentAtmosphere(altitudes_m=5000, time=0)

# plt.figure()
# plt.plot(alt / 1000, pres)
# plt.xlabel("Altitude [km]")
# plt.ylabel("Pressure [Pa]")
# plt.title("Pressure vs Altitude from MSIS")
# plt.grid(True)
# plt.show()

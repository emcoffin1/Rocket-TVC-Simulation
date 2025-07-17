"""
Will be used to provide a high fidelity aerodynamics model for a TVC simulation
Drag forces will be computed using lookup tables that use angle of attack and mach computed in Ansys Fluent
These values, if applicable, will be fitted to a spline for easy Cd determination

Empirical model for a rocket drag
https://kth.diva-portal.org/smash/get/diva2:1881325/FULLTEXT01.pdf

Free flight of Orion Crew Vehicle
https://ntrs.nasa.gov/api/citations/20220007178/downloads/Free_flight_of_MPCV.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
class Aerodynamics:
    def __init__(self):
        # -- Initialize other models -- #
        #self.air = air_profile
        pass
    def getDragCoeff(self, mach_v):
        # Get mach value
        x = mach_v
        if x <= 0.6:
            cd = 0.488 + (-0.423 * x) + (0.429 * x ** 2)
        elif x <= 0.8:
            cd = 0.84 + (-1.65 * x) + (1.5 * x ** 2)
        elif x <= 1.2:
            cd = -1.4 + (3.55 * x) + (-1.5 * x ** 2)
        elif x <= 2.0:
            cd = 0.66 + (0.113 * x) + (-0.0667 * x ** 2)
        elif x <= 5.0:
            cd = (-0.0221 * x) + 0.659
        return cd



air = Aerodynamics()
mach = np.arange(0, 5, 0.01)
cd = []
for x in mach:
    cd.append(air.getDragCoeff(x))

plt.plot(mach, cd)
plt.show()
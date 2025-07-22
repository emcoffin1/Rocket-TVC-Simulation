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
    aoa0 = None
    aoa1 = None
    aoa2 = None
    aoa3 = None
    aoa4 = None
    aoa5 = None
    aoa6 = None
    aoa7 = None
    aoa8 = None
    aoa9 = None
    aoa10 = None
    def __init__(self):
        # -- Initialize other models -- #
        #self.air = air_profile
        pass
    def getDragCoeff(self, mach_v):
        # Get mach value
        x = mach_v
        cd = None
        if x <= 0.9:
            cd = 0.425 + -0.149*x + 0.129*x**2
        elif x <= 1.1:
            cd = 0.86*x + -0.378
        else:
            cd = -0.149*x + 0.733
        return cd

#
air = Aerodynamics()
mach = np.arange(0, 5, 0.01)
cd = []
for x in mach:
    cd.append(air.getDragCoeff(x))

plt.plot(mach, cd)
plt.grid(True)
plt.show()

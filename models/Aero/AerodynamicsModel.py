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
        # if x <= 0.6:
        #     cd = 0.488 + (-0.423 * x) + (0.429 * x ** 2)
        # elif x <= 0.8:
        #     cd = 0.84 + (-1.65 * x) + (1.5 * x ** 2)
        # elif x <= 1.2:
        #     cd = -1.4 + (3.55 * x) + (-1.5 * x ** 2)
        # elif x <= 2.0:
        #     cd = 0.66 + (0.113 * x) + (-0.0667 * x ** 2)
        # elif x <= 5.0:
        #     cd = (-0.0221 * x) + 0.659
        # return cd

        if x <= 1.0:
            cd = 0.13 + -0.134*x + 0.617*x**2
        elif x <= 4.5:
            cd = 0.884 + -0.312*x + 0.04*x**2

        return cd

#
# air = Aerodynamics()
# mach = np.arange(0, 5, 0.01)
# cd = []
# for x in mach:
#     cd.append(air.getDragCoeff(x))


cd_M_points = np.array([
            0.0166667, 0.125, 0.2583333, 0.3833333, 0.5, 0.6, 0.6833333, 0.775,
            0.85, 0.9333333, 1.0, 1.1083333, 1.1916667, 1.2833333, 1.3333333,
            1.4333333, 1.525, 1.6, 1.6833333, 1.775, 1.8666667, 1.9583333,
            2.0666667, 2.1916667, 2.2916667, 2.4083333, 2.55, 2.7, 2.875,
            3.0666667, 3.3083333, 3.5416667, 3.8, 4.1, 4.525, 5.0166667,
            5.575, 6.1083333, 6.5833333, 7.0166667, 7.5666667, 8.0583333,
            8.6, 9.1083333, 9.775, 10.0416667
        ])

cd_points = np.array([
            0.3000, 0.2830, 0.2696, 0.2643, 0.2607, 0.2688, 0.2804, 0.3009,
            0.3295, 0.3643, 0.4027, 0.4536, 0.4955, 0.5313, 0.5482, 0.5536,
            0.5509, 0.5429, 0.5330, 0.5205, 0.5063, 0.4848, 0.4616, 0.4321,
            0.4054, 0.3813, 0.3536, 0.3268, 0.3009, 0.2750, 0.2500, 0.2321,
            0.2179, 0.2080, 0.2018, 0.1991, 0.2036, 0.2116, 0.2205, 0.2304,
            0.2420, 0.2518, 0.2598, 0.2616, 0.2616, 0.2616
        ])

plt.plot(cd_M_points, cd_points)
plt.show()
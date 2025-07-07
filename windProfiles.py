import matplotlib.pyplot as plt
import math
import numpy as np

class Graph:
    def __init__(self, x, y, xlabel, ylabel, title, ax, single: bool = False):
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        if single:
            self.makeGraph()
        else:
            self.makeGraphs(ax)


    def makeGraphs(self, ax):
        ax.plot(self.x, self.y)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        ax.grid(True)

        max_val = max(self.y)

        ax.text(0.1, 0.9, f"Max: {max_val:.1f}", transform=ax.transAxes, fontsize=13)

    def makeGraph(self):
        plt.plot(self.x, self.y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.grid(True)

        max_val = max(self.y)

        plt.text(0.1, 0.9, f"Max: {max_val:.1f} lbf", fontsize=13)

        plt.show()
class WindProfile:
    def __init__(self, vm, dm, alt):
        self.vm = vm
        self.dm = dm
        self.alt = alt
        self.area = 0.83 * 18 # cross-sectional area in ft^2
        self.vProfile = []
        self.fProfile = []
        self.distance = []

        self.refAlt = 27887
        self.density = 0.0765
        self.cl = 1.2

        self.createVelocities()

    def getDensity(self):
        return self.density * math.exp(-self.alt / self.refAlt)

    def createVelocities(self):
        for x in range(0, 2*self.dm):
            val = (self.vm / 2) * (1 - math.cos((math.pi * x) / self.dm))
            self.vProfile.append(val)
            self.distance.append(x)

        self.convertForce()

    def convertForce(self):
        for x in range(len(self.vProfile)):
            v = self.vProfile[x]
            f = (1/2) * self.getDensity() * v**2 * self.area * self.cl
            self.fProfile.append(f)

    def getForce(self):
        return self.fProfile

    def getDistance(self):
        return self.distance

    def getVelocity(self, x):
        return self.vProfile[x]
    def getForceVal(self, x):
        return self.fProfile[x]


class WindForce:
    def __init__(self, vm, dm, alt_start):
        self.vm = vm
        self.dm = dm

        self.LR = alt_start
        self.UR = self.LR + (2 * self.dm)

        self.refAlt = 27887
        self.density = 0.0765

    def checkWind(self, alt, area, cl, rho):
        if self.LR <= alt <= self.UR:
            alt = alt - self.LR
            return self.getForce(alt, area, cl, rho)
        else:
            return np.zeros(3)

    def getVel(self, alt):
        vel = (self.vm / 2) * (1 - math.cos((math.pi * alt) / self.dm))
        return vel

    def getForce(self, alt, area, cl, rho):
        vel = self.getVel(alt)
        force = (1 / 2) * rho * vel ** 2 * area * cl
        f = np.array([0.0, force, 0.0])
        return f







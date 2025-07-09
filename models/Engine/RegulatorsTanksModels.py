from warnings import warn
from LiquidModels import UllageGas, RP1, LOX
import matplotlib.pyplot as plt

class DomeReg:
    """
    Simulates the pressure regulation between the ullage and propellant tanks
    """
    def __init__(self, outlet_pressure: float):
        self.outletPressure = outlet_pressure

    def regulate(self, inlet_pressure):
        """
        Only allows a set pressure out unless that pressure falls below
        :param inlet_pressure: The pressure pushing from the tank [Pa]
        :return: pressure to tanks [Pa]
        """
        # returns the smaller of the two, outletPressure regulated
        # or tank pressure
        return min(inlet_pressure, self.outletPressure)


class PropTank:
    """
    Model to contain the lox or fuel
    """
    def __init__(self, volume: float, fluid: object, pressure: float):
        self.volume = volume
        self.fluid = fluid
        self.mass = volume * self.fluid.density_liquid
        self.R = 8.026
        self.P = pressure

    @property
    def getVolumeFilled(self) -> float:
        """Return volume using m/rho"""
        return self.mass / self.fluid.density_liquid




class InjectorPlate:
    def __init__(self, injector_holes: int = 1, injection_area: float = 0.01, mdot_fuel: float = 0.0, mdot_lox: float = 0.0):
        self.injector_holes = injector_holes
        self.injection_area = injection_area    # m2
        self.mdot_fuel = mdot_fuel              # kg/s
        self.mdot_lox = mdot_lox                # kg/s

    @property
    def injectMass(self, dt: float = 0.1):
        return (self.mdot_lox + self.mdot_lox) * dt



if __name__ == "__main__":

    # Initial Setup
    dt = 1.0  # seconds per time step
    total_time = 100  # seconds

    ullage = UllageGas(
        P0=2e6, V0=0.01, T0=300,
        R=296.8, gamma=1.4,
        isothermal=True
    )

    reg = DomeReg(outlet_pressure=2e5)  # 200 kPa regulated line

    tank = PropTank(
        volume=0.02,  # 20 L
        initial_mass=10,  # kg of propellant
        fluid_density=1000  # water-like
    )

    flow = 0.5  # 0.5 kg/s mass flow

    # Simulation Loop
    for t in range(int(total_time / dt)):
        # 1. Get ullage pressure and regulate it
        P_gas = ullage.getPressure
        P_line = reg.regulate(P_gas)

        # 2. Withdraw propellant
        mdot = flow
        tank.withdraw(mdot=mdot, dt=dt)

        # 3. Expand ullage gas into new volume (since propellant left)
        ullage.expand(tank.getVolumeUllage - ullage.getVolume)

        # 4. Logging
        print(f"Time {t * dt:.1f}s | Tank Pressure: {P_line / 1000:.1f} kPa | "
              f"Propellant: {tank.mass:.2f} kg | Ullage P: {P_gas / 1000:.1f} kPa | "
              f"Ullage Vol: {ullage.getVolume * 1000:.2f} L")



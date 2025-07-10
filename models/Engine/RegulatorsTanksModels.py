

class DomeReg:
    """
    Simulates the pressure regulation between the ullage and propellant tanks
    """
    def __init__(self, outlet_pressure: float):
        self.outletPressure = outlet_pressure


class PropTank:
    """
    Model to contain the lox or fuel
    """
    def __init__(self, volume: float, fluid: object):
        self.volume = volume
        self.fluid = fluid
        self.mass = volume * self.fluid.density_liquid
        print(f" Loaded: {self.fluid.name()} mass: {self.mass} kg")
        print(f" Loaded: {self.fluid.name()} volume: {self.volume} m2")


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


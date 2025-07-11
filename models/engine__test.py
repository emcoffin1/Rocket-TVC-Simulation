import numpy as np
import matplotlib.pyplot as plt
import Engine.LiquidModels

class UllageTank:
    def __init__(self, P0, V0, R, gamma, T0=288.15):
        # P0: initial pressurant pressure [Pa]
        # V0: initial ullage volume [m³]
        self.P0, self.V0 = P0, V0
        self.P, self.V      = P0, V0
        self.R, self.gamma  = R, gamma
        self.T              = T0

    def step(self, mass_used, dV):
        # ullage volume increases as propellant leaves
        self.V += dV
        # isentropic expansion: P·V^γ = P0·V0^γ
        self.P = self.P0 * (self.V0 / self.V)**self.gamma
        # (optionally update T if you need it)

class DomeRegulator:
    def __init__(self, tank: UllageTank, p_set):
        # holds feed pressure at p_set until ullage falls below
        self.tank  = tank
        self.p_set = p_set

    @property
    def feed_pressure(self):
        # if tank.P >= p_set → p_set; else → tank.P
        return min(self.tank.P, self.p_set)

class PropellantTank:
    def __init__(self, V0, rho, blowdown_gamma):
        # V0: initial liquid volume [m³]
        # rho: liquid density [kg/m³]
        # blowdown_gamma: exponent in P≈P0·(V/V0)^(-γ)
        self.V0, self.V = V0, V0
        self.rho         = rho
        self.P0          = None  # set in init_pressure()
        self.gamma       = blowdown_gamma

    def init_pressure(self, P0):
        # call once at start to record initial tank pressure
        self.P0 = P0
        self.P  = P0

    def draw(self, mdot, dt):
        m = mdot * dt
        dV = m / self.rho
        self.V = max(self.V - dV, 0.0)
        # blowdown law
        self.P = self.P0 * (self.V / self.V0)**(-self.gamma)
        return m, dV

class CombustionChamber:
    def __init__(self, Pc, alpha_lox, alpha_fuel, Tc):
        # Pc: desired chamber pressure [Pa]
        # alpha_i: [kg/s/√Pa], Tc: chamber temperature [K]
        self.Pc          = Pc
        self.alpha_lox   = alpha_lox
        self.alpha_fuel  = alpha_fuel
        self.Tc          = Tc

    def get_mass_flows(self, p_feed_lox, p_feed_fuel):
        Δp_lox  = max(p_feed_lox  - self.Pc, 0.0)
        Δp_fuel = max(p_feed_fuel - self.Pc, 0.0)
        mdot_lox  = self.alpha_lox  * np.sqrt(Δp_lox)
        mdot_fuel = self.alpha_fuel * np.sqrt(Δp_fuel)
        return mdot_lox, mdot_fuel

class Nozzle:
    def __init__(self, At, Ae, gamma, R):
        # At, Ae: throat and exit areas [m²]
        self.At, self.Ae = At, Ae
        self.gamma       = gamma
        self.R           = R

    def thrust(self, mdot, Pc, Tc, pa):
        # ideal isentropic:
        # exit pressure from area ratio
        pe = Pc * (self.Ae/self.At)**(-self.gamma)
        # momentum term:
        term1 = mdot * np.sqrt(self.gamma * self.R * Tc)
        # pressure thrust:
        term2 = (pe - pa) * self.Ae
        return term1 + term2

class BlowdownEngine:
    def __init__(self, cfg):
        # 1) ullage
        self.ullage   = UllageTank(**cfg["ullage"])

        # 2) propellant tanks
        self.lox_tank  = PropellantTank(**cfg["lox_tank"])
        self.fuel_tank = PropellantTank(**cfg["fuel_tank"])
        # initialize their P0 to ullage.P0 so they start at same pressure
        self.lox_tank .init_pressure(self.ullage.P0)
        self.fuel_tank.init_pressure(self.ullage.P0)

        # 3) regulators
        self.lox_reg  = DomeRegulator(self.ullage, cfg["reg"]["p_lox"])
        self.fuel_reg = DomeRegulator(self.ullage, cfg["reg"]["p_fuel"])

        # 4) chamber & injector model
        self.chamber = CombustionChamber(**cfg["chamber"])

        # 5) nozzle
        self.nozzle  = Nozzle(**cfg["nozzle"])

    def step(self, dt, pa):
        # feed pressures
        p_feed_lox  = self.lox_reg .feed_pressure
        p_feed_fuel = self.fuel_reg.feed_pressure

        # injector flows
        mdot_lox,  mdot_fuel  = self.chamber.get_mass_flows(p_feed_lox,
                                                            p_feed_fuel)
        mdot_total = mdot_lox + mdot_fuel

        # draw from propellant tanks, expand ullage
        m_lox,  dV_lox  = self.lox_tank .draw(mdot_lox,  dt)
        m_fuel, dV_fuel = self.fuel_tank.draw(mdot_fuel, dt)
        self.ullage.step(mass_used=(m_lox + m_fuel),
                         dV=(dV_lox + dV_fuel))

        # compute thrust
        return self.nozzle.thrust(mdot_total,
                                  self.chamber.Pc,
                                  self.chamber.Tc,
                                  pa)

    def run(self, t_final, dt, pa=101325):
        times = np.arange(0, t_final+dt, dt)
        thrust = np.empty_like(times)
        for i, t in enumerate(times):
            thrust[i] = self.step(dt, pa)
        return times, thrust

if __name__ == "__main__":
    # ---------- example metric config ----------
    # convert your fitted alpha’s:
    # α_lox ≈2.709kg/s / sqrt(1.142e6Pa) ≈0.00254
    # α_fuel≈1.505kg/s / sqrt(1.231e6Pa) ≈0.00136
    config = {
      "ullage": {
        "P0":      5e6,    # Pa
        "V0":      0.02,   # m³
        "R":       287.0,  # J/kg·K
        "gamma":   1.4
      },
      "lox_tank": {
        "V0":           0.05,  # m³
        "rho":        1140.0,  # kg/m³
        "blowdown_gamma": 1.2
      },
      "fuel_tank": {
        "V0":           0.03,  # m³
        "rho":         810.0,  # kg/m³
        "blowdown_gamma": 1.2
      },
      "reg": {
        "p_lox":  3e6,   # Pa
        "p_fuel": 3e6    # Pa
      },
      "chamber": {
        "Pc":         3e6,       # Pa
        "alpha_lox":  0.00254,   # kg/s/√Pa
        "alpha_fuel": 0.00136,   # kg/s/√Pa
        "Tc":         3500.0     # K (approx)
      },
      "nozzle": {
        "At":    0.0082568, # m²
        "Ae":    0.0159851, # m²
        "gamma": 1.22,
        "R":     355.0      # J/kg·K
      }
    }

    engine = BlowdownEngine(config)
    t, F = engine.run(t_final=5.0, dt=0.01, pa=101325)

    plt.plot(t, F)
    plt.title("Blowdown Engine Thrust (SI units)")
    plt.xlabel("Time [s]")
    plt.ylabel("Thrust [N]")
    plt.grid(True)
    plt.show()

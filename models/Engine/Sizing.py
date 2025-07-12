from rocketcea.cea_obj import CEA_Obj
from math import sqrt, exp
from scipy.optimize import brentq

"""
This files contains all the definitions and functions required to size the initial engine
As there are many ways to do accomplish this, multiple methods may be used to acquire various values
Many values obtained from : https://www.grc.nasa.gov/www/k-12/airplane/rktthsum.html#
"""


def getLiquidCharacteristics(chamber_pressure: float, of_ratio: float, eps: float):
    """Uses CEA to determine various characteristics of
    LOX and RP1 burn properties
    :param chamber_pressure: Chamber pressure [Pa]
    :param of_ratio: Oxidizer to fuel ratio    []
    :param eps: Epsilon: Nozzle expansion ratio [Ae/At] []
    :returns: tuple - (molar weight, specific heat ratio, gas constant, chamber temp, c*)
              tuple - (kg/mol, --, J/kg*K, K, m/s)
    """
    _CEA = CEA_Obj(oxName="LOX", fuelName="RP-1")

    # Convert Pa to psi
    pc_psi = chamber_pressure / 6894.76

    print("=" * 60)
    print("GET LIQUID CHARACTERISTICS")

    # Molar weight and specific gas constant
    m_wt, gamma = _CEA.get_Chamber_MolWt_gamma(Pc=pc_psi, MR=of_ratio, eps=eps)
    m_wt = 8.314 / (m_wt * 0.45359237)
    r = 8314.4621 / m_wt
    print(f"Molar Mass:                     {m_wt:.4f} [kg/mol]")
    print(f"Specific Heat Ratio (gamma):    {gamma:.4f} []")
    print(f"Gas Constant (R):               {r:.4f} [J/kg*K]")

    # Chamber Temp
    ct = _CEA.get_Tcomb(Pc=pc_psi, MR=of_ratio)
    ct = (ct - 491.67) * (5/9) + 273.15
    print(f"Chamber Temperature:            {ct:.4f} [K]")

    # Characteristic velocity c*
    cstar = _CEA.get_Cstar(Pc=pc_psi, MR=of_ratio)
    cstar = cstar * 0.3048
    print(f"Characteristic Velocity (c*):   {cstar:.4f} [m/s]")

    print("=" * 60)
    return m_wt, gamma, r, ct, cstar


def getMassFlowRate(throat_area: float, chamber_pressure: float, gamma: float, chamber_temp: float):
    """
    Uses throat area, chamber pressure, specific heat ratio, and gas constant to compute mass flow rate
    :param throat_area: Area of chamber throat [m2]
    :param chamber_pressure: Chamber pressure [Pa]
    :param of_ratio: Ratio of oxidizer and fuel
    :param eps: Specific heat ratio
    :return:
    """
    term1 = throat_area * chamber_pressure / sqrt(chamber_temp)
    term2 = sqrt(gamma / r)
    exp   = -(gamma + 1) / (2 * (gamma - 1))
    term3 = ((gamma + 1) / 2) ** exp
    mdot = term1 * term2 * term3
    print("=" * 60)
    print("GET MASS FLOW RATE")
    print(f"Mass Flow Rate:                 {mdot:.4f} [kg/s]")
    print("=" * 60)

    return mdot




def getExitMach(eps: float, gamma: float):
    """
    Solves for the exit Mach number using the isentropic area–Mach relation.
    Automatically determines if subsonic or supersonic based on expansion ratio.
    :param chamber_pressure: [Pa]
    :param of_ratio: []
    :param eps: Expansion ratio (Ae / At)
    :return: Exit Mach number
    """

    def area_mach_residual(M):
        rhs = (1 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M ** 2)) ** ((gamma + 1) / (2 * (gamma - 1)))
        return eps - rhs

    # Compute the minimum expansion ratio (Ae/At) at Mach = 1
    min_eps = (1 / 1.0) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * 1.0 ** 2)) ** ((gamma + 1) / (2 * (gamma - 1)))

    # Decide subsonic vs supersonic based on expansion ratio
    if eps > min_eps:
        bracket = [1.01, 10.0]
        flow_type = "SUP"
    else:
        bracket = [0.01, 0.99]
        flow_type = "SUB"

    # Solve using Brent’s method
    try:
        Me_solution = brentq(area_mach_residual, *bracket)
        print("=" * 60)
        print("GET EXIT MACH NUMBER")
        print(f"Expansion Ratio (eps):          {eps:.4f}")
        print(f"Minimum eps (Mach 1):           {min_eps:.4f}")
        print(f"Flow regime:                    {flow_type}")
        print(f"Exit Mach:                      {Me_solution:.4f} [M]")
        print("=" * 60)
        return Me_solution
    except ValueError as e:
        raise ValueError(f"Mach number solver failed: {e}")


def getExitTemperature(gamma: float, exit_mach, combustion_temp: float):
    term1 = (1 + (gamma - 1) * (exit_mach**2) / 2)**-1
    exit_temp = term1 * combustion_temp
    print("=" * 60)
    print("GET EXIT TEMPERATURE")
    print(f"Specific Heat Ratio (gamma):    {gamma} []")
    print(f"Expansion Ratio (eps):          {exit_mach:.4f} []")
    print(f"Exit Temperature:               {exit_temp:.4f} [K]")
    print("=" * 60)

    return exit_temp

def getExitPressure(gamma: float, exit_mach, chamber_pressure: float):
    term1 = (1 + (gamma - 1) * (exit_mach**2) / 2)**-(gamma / (gamma - 1))
    exit_press = term1 * chamber_pressure
    print("=" * 60)
    print("GET EXIT PRESSURE")
    print(f"Specific Heat Ratio (gamma):    {gamma:.4f} []")
    print(f"Expansion Ratio (eps):          {exit_mach:.4f} []")
    print(f"Exit Pressure:                  {exit_press:.4f} [Pa]")

    if exit_press < 101325:
        print(f"Flow is over-expanded at sealevel")
    else:
        print(f"Flow is under-expanded at sealevel")

    print("=" * 60)

    return exit_temp


def getExitVelocity(exit_mach, gamma: float, spec_gas_const: float, exit_temperature: float):

    exit_v = exit_mach * sqrt(gamma * spec_gas_const * exit_temperature)

    print("=" * 60)
    print("GET EXIT VELOCITY")
    print(f"Exit Velocity:                  {exit_v:.4f} [m/s]")
    print("=" * 60)

    return exit_v

def getMassFlow_Thrust(desired_thrust: float, exit_pressure: float, exit_area: float, exit_velocity:float):
    mdot = (desired_thrust - ((exit_pressure - 101325) * exit_area)) / exit_velocity

    print("=" * 60)
    print("GET MASS FLOW FROM THRUST")
    print(f"Mass Flow Rate:                 {mdot:.4f} [kg/s]")
    print(f"Thrust:                         {desired_thrust:.4f} [N]")
    print(f"Exit Area:                      {exit_area:.4f} [m2]")
    print("=" * 60)

    return mdot


def getMinimumDeltaVelocity(altitude: float):
    term1 = sqrt(2 * 9.8 * altitude)
    print("=" * 60)
    print("GET MINIMUM DELTA VELOCITY")
    print(f"Altitude:                       {altitude:.4f} [m]")
    print(f"Delta v:                        {term1:.4f} [m/s]")
    print("=" * 60)
    return term1

def getMassFraction_dv(delta_v: float, exhaust_velocity: float):
    frac = exp(delta_v / exhaust_velocity)
    print("=" * 60)
    print("GET MASS FRACTION FROM dv")
    print(f"Delta v:                        {delta_v:.4f} [m/s]")
    print(f"Exhaust Velocity:               {exhaust_velocity:.4f} [m/s]")
    print(f"Mass Fraction:                  {frac:.4f} [w0 / wf]")
    print("=" * 60)
    return frac



chamber_pressure = 551853
eps = 1.927
of = 1.8
Ae = 0.01598
molar_weight, gamma, r, chamber_temp, cstar = getLiquidCharacteristics(chamber_pressure=chamber_pressure, of_ratio=of, eps=eps)

Me = getExitMach(eps=1.927, gamma=gamma)
mdot = getMassFlowRate(throat_area=0.00825675768,chamber_temp=chamber_temp, chamber_pressure=551583, gamma=gamma)
exit_temp = getExitTemperature(gamma=gamma, combustion_temp=chamber_temp, exit_mach=Me)
exit_press = getExitPressure(gamma=gamma, chamber_pressure=chamber_pressure, exit_mach=Me)
exit_velocity = getExitVelocity(gamma=gamma, exit_mach=Me, spec_gas_const=r, exit_temperature=chamber_temp)


# No return
getMassFlow_Thrust(desired_thrust=8000, exit_pressure=exit_press, exit_area=Ae, exit_velocity=exit_velocity)
dv = getMinimumDeltaVelocity(altitude=100000)
getMassFraction_dv(delta_v=dv, exhaust_velocity=exit_velocity)










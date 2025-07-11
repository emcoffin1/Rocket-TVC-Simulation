



def _get_tank_size(self, mdot_target: float, of_ratio_target: float, burntime:float,
                   pressure_target: float, lox_temp: float, fuel_temp: float) -> dict:

    mdot_f = mdot_target / (1 + of_ratio_target)
    mdot_l = mdot_target - mdot_f

    rho_fuel = self.fuel.liquidDensity(fuel_temp)
    rho_ox = self.lox.liquidDensity(lox_temp)

    m_f = mdot_f * burntime
    m_o = mdot_l * burntime

    v_f = m_f / rho_fuel
    v_o = m_o / rho_ox

    return {
        "fuel_mass": m_f,
        "fuel_volume": v_f,
        "lox_mass": m_o,
        "lox_volume": v_o
    }

def _get_ullage_tank_pressure(self, ullage_volume: float, mdot_target: float, of_ratio_target: float, burntime:float,
                   pressure_target: float, lox_temp: float, fuel_temp: float) -> dict:

    tank_sizing = self._get_tank_size(mdot_target=mdot_target, of_ratio_target=of_ratio_target, burntime=burntime,
                   pressure_target=pressure_target, lox_temp=lox_temp, fuel_temp=fuel_temp)

    m_f = tank_sizing["fuel_mass"]
    v_f = tank_sizing["fuel_volume"]
    m_o = tank_sizing["lox_mass"]
    v_o = tank_sizing["lox_volume"]

    m_ullage_f = pressure_target * v_f / (self.ullage.R * self.ullage.T)
    m_ullage_o = pressure_target * v_o / (self.ullage.R * self.ullage.T)

    total_ullage_mass = m_ullage_o + m_ullage_f

    ullage_pressure = total_ullage_mass * self.ullage.R * self.ullage.T / ullage_volume

    return {
        "fuel_mass": m_f,
        "fuel_volume": v_f,
        "lox_mass": m_o,
        "lox_volume": v_o,
        "nitrogen_mass": total_ullage_mass,
        "nitrogen_pressure": ullage_pressure
    }

def get_fluid_setup(self, ullage_volume: float, mdot_target: float, of_ratio_target: float, burntime:float,
                   pressure_target: float, lox_temp: float, fuel_temp: float):

    info = self._get_ullage_tank_pressure(ullage_volume=ullage_volume, mdot_target=mdot_target,
                                          of_ratio_target=of_ratio_target, burntime=burntime,
                                          pressure_target=pressure_target, lox_temp=lox_temp, fuel_temp=fuel_temp)

    print("=" * 15)
    print(f"RECOMMENDED -- FUEL MASS:         {info['fuel_mass']:.4f} kg")
    print(f"RECOMMENDED -- FUEL VOLUME:       {info['fuel_volume']:.4f} m3")
    print(f"RECOMMENDED -- LOX MASS:          {info['lox_mass']:.4f} kg")
    print(f"RECOMMENDED -- LOX VOLUME:        {info['lox_volume']:.4f} m3")
    print(f"RECOMMENDED -- NITROGEN MASS:     {info['nitrogen_mass']:.4f} kg")
    print(f"RECOMMENDED -- NITROGEN PRESSURE: {info['nitrogen_pressure']:.4f} Pa")
    print("=" * 15)

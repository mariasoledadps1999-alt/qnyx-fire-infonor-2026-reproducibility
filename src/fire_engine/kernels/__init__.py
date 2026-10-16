"""
src/fire_engine/kernels/__init__.py
"""

from .base import SpreadKernel, SpreadResult
from .rothermel import (
    RothermelKernel,
    RothermelInputs,
    RothermelResult,
    calculate_bulk_density,
    calculate_packing_ratio,
    calculate_characteristic_sav,
    calculate_optimum_packing_ratio,
    calculate_max_reaction_velocity,
    calculate_reaction_exponent,
    calculate_optimum_reaction_velocity,
    calculate_moisture_damping,
    calculate_mineral_damping,
    calculate_net_fuel_load,
    calculate_propagating_flux_ratio,
    calculate_effective_heating_number,
    calculate_heat_of_preignition,
    calculate_wind_coefficients,
    calculate_wind_factor,
    calculate_slope_factor_rothermel,
    calculate_wind_limit,
    calculate_live_fuel_extinction_moisture
)

__all__ = [
    "SpreadKernel",
    "SpreadResult",
    "RothermelKernel",
    "RothermelInputs",
    "RothermelResult",
    "calculate_bulk_density",
    "calculate_packing_ratio",
    "calculate_characteristic_sav",
    "calculate_optimum_packing_ratio",
    "calculate_max_reaction_velocity",
    "calculate_reaction_exponent",
    "calculate_optimum_reaction_velocity",
    "calculate_moisture_damping",
    "calculate_mineral_damping",
    "calculate_net_fuel_load",
    "calculate_propagating_flux_ratio",
    "calculate_effective_heating_number",
    "calculate_heat_of_preignition",
    "calculate_wind_coefficients",
    "calculate_wind_factor",
    "calculate_slope_factor_rothermel",
    "calculate_wind_limit",
    "calculate_live_fuel_extinction_moisture"
]

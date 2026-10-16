"""
src/fire_engine/fuels/__init__.py
"""

from .manager import FuelManager
from .models import (
    FuelParticleClass,
    RothermelFuelBed,
    FuelMoisture,
    get_standard_fuel_model,
    STANDARD_FUEL_MODELS,
    ORIGINAL_13_FUEL_MODELS
)

from .moisture import (
    FuelMoistureState,
    FuelMoistureProvider,
    ConstantFuelMoistureProvider,
    DynamicFuelMoistureProvider,
    DynamicHourlyFuelMoistureProvider,
    ExternalFuelMoistureProvider,
    ConditioningReport,
    FuelMoistureError,
    InvalidFuelMoistureValueError,
    UnsupportedFuelMoistureModeError,
    create_fuel_moisture_provider,
    calculate_simard_emc,
    integrate_timelag_step,
    run_fuel_moisture_conditioning,
    DEFAULT_RESIDUAL_MEMORY_WARNING_THRESHOLD,
)

from .radial_diffusion import (
    RadialDiffusionConfig,
    RadialDiffusionSolver,
    RadialDiffusionError,
    solve_tridiagonal,
    analytical_bessel_cylinder_diffusion
)

from .nelson import (
    NelsonPhysicalModel,
    NelsonFuelMoistureProvider,
    NelsonFuelMoistureTimeSeries,
    NelsonCheckpointState,
    NelsonConditioningReport,
    NelsonError,
    NelsonRequiredWeatherInputMissingError,
    ExtinctionGuardEvaluator,
)

__all__ = [
    "FuelManager",
    "FuelParticleClass",
    "RothermelFuelBed",
    "FuelMoisture",
    "get_standard_fuel_model",
    "STANDARD_FUEL_MODELS",
    "ORIGINAL_13_FUEL_MODELS",
    "FuelMoistureState",
    "FuelMoistureProvider",
    "ConstantFuelMoistureProvider",
    "DynamicFuelMoistureProvider",
    "DynamicHourlyFuelMoistureProvider",
    "ExternalFuelMoistureProvider",
    "ConditioningReport",
    "FuelMoistureError",
    "InvalidFuelMoistureValueError",
    "UnsupportedFuelMoistureModeError",
    "create_fuel_moisture_provider",
    "calculate_simard_emc",
    "integrate_timelag_step",
    "run_fuel_moisture_conditioning",
    "DEFAULT_RESIDUAL_MEMORY_WARNING_THRESHOLD",
    "RadialDiffusionConfig",
    "RadialDiffusionSolver",
    "RadialDiffusionError",
    "solve_tridiagonal",
    "analytical_bessel_cylinder_diffusion",
    "NelsonPhysicalModel",
    "NelsonFuelMoistureProvider",
    "NelsonFuelMoistureTimeSeries",
    "NelsonCheckpointState",
    "NelsonConditioningReport",
    "NelsonError",
    "NelsonRequiredWeatherInputMissingError",
    "ExtinctionGuardEvaluator",
]

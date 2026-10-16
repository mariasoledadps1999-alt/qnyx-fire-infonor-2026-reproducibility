"""
tests/test_smoke.py

Fast smoke test suite runnable in any clean environment in < 2 seconds.
Verifies:
1. All scientific core modules import without errors.
2. Rothermel 1972 forward calculation produces valid positive ROS.
3. Albini / Anderson directional ellipse calculation works.
4. Nelson physical dead fuel moisture solver computes valid states.
5. Key configuration files and directories exist.
"""

import math
import pytest
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_core_imports():
    """Verify all core scientific modules can be imported without missing dependencies."""
    from fire_engine.kernels.rothermel import (
        RothermelKernel,
        RothermelInputs,
        calculate_wind_factor,
        calculate_slope_factor_rothermel
    )
    from fire_engine.directional.spread import (
        DirectionalSpreadInputs,
        DirectionalSpreadResult,
        calculate_directional_rothermel_spread
    )
    from fire_engine.directional.ellipse import (
        calculate_length_to_breadth_anderson_farsite,
        calculate_ellipse_eccentricity,
        calculate_focus_radial_factor_elliptic
    )
    from fire_engine.directional.resultant import combine_wind_and_slope_vectors
    from fire_engine.fuels.models import (
        get_standard_fuel_model,
        load_scott_burgan_catalog,
        FuelMoisture,
        ORIGINAL_13_FUEL_MODELS
    )
    from fire_engine.fuels.nelson import (
        NelsonPhysicalModel,
        NelsonModelConfig,
        NelsonDFMState,
        NelsonFuelMoistureProvider
    )
    from fire_engine.spread import run_minimum_arrival_time
    from fire_engine.weather import WeatherManager


def test_rothermel_smoke_run():
    """Run minimal synthetic Rothermel calculation for FM1."""
    from fire_engine import units
    from fire_engine.fuels.models import get_standard_fuel_model, FuelMoisture
    from fire_engine.directional.spread import (
        DirectionalSpreadInputs,
        calculate_directional_rothermel_spread
    )

    bed = get_standard_fuel_model(1)
    moisture = FuelMoisture(dead_1h=0.08, dead_10h=0.09, dead_100h=0.10, live_herbaceous=1.0, live_woody=1.0)
    inputs = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moisture,
        midflame_wind_ms=2.0,
        wind_from_direction_deg=270.0,  # Wind from West -> blows East
        slope_degrees=0.0,
        aspect_downslope_degrees=0.0
    )
    res = calculate_directional_rothermel_spread(inputs)
    assert res.r_head_m_min > 0.0, "Head ROS must be strictly positive"
    r_head = res.ros_at_heading(res.heading_resultant_deg)
    r_flank = res.ros_at_heading((res.heading_resultant_deg + 90.0) % 360.0)
    r_back = res.ros_at_heading((res.heading_resultant_deg + 180.0) % 360.0)
    assert r_head >= r_flank, "Head ROS must exceed or equal Flank ROS"
    assert r_flank >= r_back, "Flank ROS must exceed or equal Back ROS"
    assert res.reaction_intensity_kw_m2 > 0.0, "Reaction intensity must be positive"


def test_directional_ellipse_smoke():
    """Verify Albini/Anderson ellipse eccentricity."""
    from fire_engine import units
    from fire_engine.directional.ellipse import (
        calculate_length_to_breadth_anderson_farsite,
        calculate_ellipse_eccentricity
    )
    lb = calculate_length_to_breadth_anderson_farsite(5.0 * units.MPH_TO_M_S)
    assert lb > 1.0, "Length-to-breadth ratio must exceed 1.0 under wind"
    ecc = calculate_ellipse_eccentricity(lb)
    assert 0.0 < ecc < 1.0, "Eccentricity must be bounded in (0, 1)"


def test_nelson_physical_model_smoke():
    """Verify 1-step Nelson physical dead moisture solver."""
    from fire_engine.fuels.nelson import (
        NelsonPhysicalModel,
        NelsonModelConfig,
        NelsonDFMState
    )
    model = NelsonPhysicalModel(NelsonModelConfig.for_10h())
    model.initialize(init_moisture=0.15, init_temp_c=20.0)
    wx = {
        "temperature_c": 30.0,
        "relative_humidity_pct": 25.0,
        "solar_radiation_w_m2": 600.0,
        "precipitation_mm": 0.0,
    }
    model.update(weather_current=wx, dt_hours=1.0)
    state = model.get_state()
    assert 0.0 < state.mean_weighted_moisture < 0.50, f"Expected physical dead fuel moisture ratio, got {state.mean_weighted_moisture}"


def test_repository_structure_exists():
    """Verify clean repository layout contracts."""
    assert (REPO_ROOT / "src" / "fire_engine").is_dir()
    assert (REPO_ROOT / "configs").is_dir()
    assert (REPO_ROOT / "scripts").is_dir()
    assert (REPO_ROOT / "benchmarks").is_dir()
    assert (REPO_ROOT / "manifests").is_dir()
    assert (REPO_ROOT / "results").is_dir()
    assert (REPO_ROOT / "configs" / "fuel_models_scott_burgan_qnyx.json").is_file()
    assert (REPO_ROOT / "configs" / "scenarios" / "talami_2017.yaml").is_file()
    assert (REPO_ROOT / "scripts" / "reproduce_all.py").is_file()
    assert (REPO_ROOT / "scripts" / "run_talami_simulation.py").is_file()
    assert (REPO_ROOT / "scripts" / "validate_talami_conaf.py").is_file()
    assert (REPO_ROOT / "scripts" / "run_bench_006_flammap.py").is_file()
    assert (REPO_ROOT / "scripts" / "compare_bench_006.py").is_file()

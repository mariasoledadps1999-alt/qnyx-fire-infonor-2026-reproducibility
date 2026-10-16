"""
tests/test_rothermel_kernel.py

Suite de pruebas automatizadas en 3 niveles para el kernel físico aislado de Rothermel (1972 INT-115).
Taxonomía:
- Level 1: Unit Equations & Constants (tolerancia flotante < 1e-6)
- Level 2: Analytical Internal Benchmarks, Limits & Monotonicity (tolerancia < 1e-4)
- Level 3: External Golden Benchmarks (Scott 2007 RMRS-GTR-192 / NWCG PMS 436-3 / BehavePlus, tolerancia < 2.0 %)
"""

import math
import pytest

from fire_engine import units
from fire_engine.fuels.models import (
    FuelParticleClass,
    RothermelFuelBed,
    FuelMoisture,
    get_standard_fuel_model,
    STANDARD_FUEL_MODELS
)
from fire_engine.kernels.rothermel import (
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
    calculate_rothermel_spread
)


# =============================================================================
# LEVEL 1: UNIT EQUATIONS & CONSTANTS
# =============================================================================

def test_l1_unit_conversions():
    """Verifica la precisión de las constantes tipadas de conversión."""
    assert math.isclose(units.FEET_TO_METERS * units.METERS_TO_FEET, 1.0, rel_tol=1e-9)
    assert math.isclose(1.0 * units.MPH_TO_FT_MIN, 88.0, rel_tol=1e-9)
    assert math.isclose(1.0 * units.FT_MIN_TO_M_MIN, 0.3048, rel_tol=1e-9)
    assert math.isclose(1.0 * units.CHAINS_HR_TO_FT_MIN, 1.1, rel_tol=1e-9)
    assert math.isclose(units.TONS_ACRE_TO_LB_FT2, 2000.0 / 43560.0, rel_tol=1e-9)


def test_l1_bulk_density_and_packing_ratio():
    """Verifica rho_b (Eq. 2) y beta (Eq. 3)."""
    # Modelo 1: w0 = 0.03397612 lb/ft², delta = 1.0 ft
    w0 = 0.74 * (2000.0 / 43560.0)
    depth = 1.0
    rho_b = calculate_bulk_density(w0, depth)
    assert math.isclose(rho_b, 0.033976125, rel_tol=1e-5)

    beta = calculate_packing_ratio(rho_b, 32.0)
    assert math.isclose(beta, 0.033976125 / 32.0, rel_tol=1e-5)
    assert math.isclose(beta, 0.001061754, rel_tol=1e-4)


def test_l1_characteristic_sav_and_beta_opt():
    """Verifica sigma (Eq. 38) y beta_op (Eq. 37)."""
    sigma = 3500.0
    beta_op = calculate_optimum_packing_ratio(sigma)
    # beta_op = 3.348 * 3500^-0.8189 ≈ 0.00419349
    assert math.isclose(beta_op, 0.00419349, rel_tol=1e-4)


def test_l1_reaction_velocities_and_exponent():
    """Verifica Gamma_max (Eq. 36), A (Eq. 39) y Gamma' (Eq. 35)."""
    sigma = 3500.0
    gamma_max = calculate_max_reaction_velocity(sigma)
    A = calculate_reaction_exponent(sigma)
    
    assert math.isclose(gamma_max, 16.1818, rel_tol=1e-3)
    assert math.isclose(A, 0.20866, rel_tol=1e-3)

    beta = 0.001061754
    beta_op = 0.00419349
    gamma_prime = calculate_optimum_reaction_velocity(sigma, beta, beta_op)
    assert math.isclose(gamma_prime, 14.1997, rel_tol=1e-3)


def test_l1_damping_functions():
    """Verifica eta_M (Eq. 29) y eta_S (Eq. 30)."""
    # Para rm = 0.08 / 0.12 = 0.666667
    eta_M = calculate_moisture_damping(0.08, 0.12)
    assert math.isclose(eta_M, 0.5014, rel_tol=1e-3)

    # Para mf >= mx -> eta_M = 0
    assert calculate_moisture_damping(0.12, 0.12) == 0.0
    assert calculate_moisture_damping(0.15, 0.12) == 0.0

    # Mineral damping
    eta_S = calculate_mineral_damping(0.010)
    assert math.isclose(eta_S, 0.4174, rel_tol=1e-3)


def test_l1_propagating_flux_and_heating():
    """Verifica xi (Eq. 42), eps (Eq. 14) y Q_ig (Eq. 12)."""
    sigma = 3500.0
    beta = 0.001061754
    xi = calculate_propagating_flux_ratio(sigma, beta)
    assert math.isclose(xi, 0.057746, rel_tol=1e-4)

    p_class = FuelParticleClass(name="d1h", load_lb_ft2=0.034, sav_inv_ft=3500.0)
    eps = calculate_effective_heating_number([p_class])
    assert math.isclose(eps, math.exp(-138.0 / 3500.0), rel_tol=1e-5)
    assert math.isclose(eps, 0.96134, rel_tol=1e-4)

    moist = FuelMoisture(dead_1h=0.08)
    qig = calculate_heat_of_preignition([p_class], moist)
    assert math.isclose(qig, 250.0 + 1116.0 * 0.08, rel_tol=1e-5)
    assert math.isclose(qig, 339.28, rel_tol=1e-5)


def test_l1_wind_coefficients_and_factor():
    """Verifica C (Eq. 48), B (Eq. 49), E (Eq. 50) y phi_w (Eq. 47)."""
    sigma = 3500.0
    C, B, E = calculate_wind_coefficients(sigma)
    
    # C = 7.47 * exp(-0.133 * 3500^0.55) ≈ 0.00005425
    assert math.isclose(C, 0.00005425, rel_tol=1e-3)
    # B = 0.02526 * 3500^0.54 ≈ 2.0712
    assert math.isclose(B, 2.0712, rel_tol=1e-3)
    # E = 0.715 * exp(-3.59e-4 * 3500) ≈ 0.2035
    assert math.isclose(E, 0.2035, rel_tol=1e-3)

    beta = 0.001061754
    beta_op = 0.00419349
    u_5mph = 440.0 # ft/min
    phi_w, _, _, _ = calculate_wind_factor(u_5mph, sigma, beta, beta_op)
    assert math.isclose(phi_w, 21.43, rel_tol=1e-2)


def test_l1_slope_factor():
    """Verifica phi_s (Eq. 51)."""
    beta = 0.001061754
    # Pendiente 0°
    assert calculate_slope_factor_rothermel(0.0, beta) == 0.0
    # Pendiente 30° (tan(30°) = 0.57735)
    phi_s_30 = calculate_slope_factor_rothermel(30.0, beta)
    # 5.275 * (0.001061754)^-0.3 * (tan(30°))^2 ≈ 13.7181
    assert math.isclose(phi_s_30, 13.7181, rel_tol=1e-3)


def test_l1_wind_limit_function():
    """Verifica U_max = 0.9 * IR."""
    IR = 763.05
    u_max = calculate_wind_limit(IR)
    assert math.isclose(u_max, 0.9 * 763.05, rel_tol=1e-5)
    assert math.isclose(u_max, 686.745, rel_tol=1e-4)


# =============================================================================
# LEVEL 2: ANALYTICAL INTERNAL BENCHMARKS, LIMITS & MONOTONICITY
# =============================================================================

def test_l2_zero_wind_and_flat_terrain():
    """Verifica que sin viento ni pendiente, R_head coincide exactamente con R0."""
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)
    inputs = RothermelInputs(fuel_bed=bed, moisture=moist, midflame_wind_ms=0.0, slope_degrees=0.0)
    
    kernel = RothermelKernel()
    res = kernel.calculate(inputs)

    assert res.phi_w == 0.0
    assert res.phi_s == 0.0
    assert math.isclose(res.r_head_ft_min, res.r0_ft_min, rel_tol=1e-9)
    assert math.isclose(res.r0_ft_min, 3.9765, rel_tol=1e-3)
    assert math.isclose(res.r0_m_min, 1.2120, rel_tol=1e-3)
    assert math.isclose(res.r0_chains_hr, 3.615, rel_tol=1e-3)


def test_l2_extinction_moisture_cutoff():
    """Verifica extinción total cuando M_f >= M_x."""
    bed = get_standard_fuel_model(1) # Mx = 0.12
    moist_extinct = FuelMoisture(dead_1h=0.12)
    inputs = RothermelInputs(fuel_bed=bed, moisture=moist_extinct, midflame_wind_ms=5.0)
    
    kernel = RothermelKernel()
    res = kernel.calculate(inputs)

    assert res.reaction_intensity_btu_ft2_min == 0.0
    assert res.r0_m_min == 0.0
    assert res.r_head_m_min == 0.0


def test_l2_monotonicity_wind_slope_moisture():
    """Verifica monotonía estricta: dR/dU > 0, dR/dphi > 0, dR/dMf < 0."""
    bed = get_standard_fuel_model(1)
    kernel = RothermelKernel()

    # Viento creciente
    res_w0 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.08), midflame_wind_ms=0.0))
    res_w2 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.08), midflame_wind_ms=2.0))
    res_w5 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.08), midflame_wind_ms=5.0))
    assert res_w0.r_head_m_min < res_w2.r_head_m_min < res_w5.r_head_m_min

    # Pendiente creciente
    res_s0 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.08), slope_degrees=0.0))
    res_s10 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.08), slope_degrees=10.0))
    res_s30 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.08), slope_degrees=30.0))
    assert res_s0.r_head_m_min < res_s10.r_head_m_min < res_s30.r_head_m_min

    # Humedad creciente -> ROS decreciente
    res_m4 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.04), midflame_wind_ms=2.0))
    res_m8 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.08), midflame_wind_ms=2.0))
    res_m11 = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=FuelMoisture(dead_1h=0.11), midflame_wind_ms=2.0))
    assert res_m4.r_head_m_min > res_m8.r_head_m_min > res_m11.r_head_m_min


def test_l2_wind_limit_capped_vs_uncapped():
    """Verifica la política de wind limit para FM1 a 10 mph."""
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)
    u_10mph_ms = 10.0 * units.MPH_TO_M_S # 4.4704 m/s

    kernel = RothermelKernel()
    
    # Capped (rothermel_limit)
    res_capped = kernel.calculate(RothermelInputs(
        fuel_bed=bed, moisture=moist, midflame_wind_ms=u_10mph_ms, wind_limit_policy="rothermel_limit"
    ))
    assert res_capped.wind_limit_active is True
    assert math.isclose(res_capped.effective_midflame_wind_mph, 7.804, rel_tol=1e-3)
    assert math.isclose(res_capped.r_head_m_min, 66.52, rel_tol=1e-3)
    assert math.isclose(res_capped.r_head_chains_hr, 198.4, rel_tol=1e-3)

    # Uncapped (none)
    res_uncapped = kernel.calculate(RothermelInputs(
        fuel_bed=bed, moisture=moist, midflame_wind_ms=u_10mph_ms, wind_limit_policy="none"
    ))
    assert res_uncapped.wind_limit_active is False
    assert math.isclose(res_uncapped.effective_midflame_wind_mph, 10.0, rel_tol=1e-3)
    assert math.isclose(res_uncapped.r_head_m_min, 110.36, rel_tol=1e-3)
    assert math.isclose(res_uncapped.r_head_chains_hr, 329.2, rel_tol=1e-3)


def test_l2_multiclass_model4_behavior():
    """Verifica que el Modelo 4 multiclase pesado opera de forma consistente."""
    bed = get_standard_fuel_model(4)
    moist = FuelMoisture(dead_1h=0.06, dead_10h=0.07, dead_100h=0.08, live_woody=1.00)
    inputs = RothermelInputs(fuel_bed=bed, moisture=moist, midflame_wind_ms=5.0 * units.MPH_TO_M_S)
    
    kernel = RothermelKernel()
    res = kernel.calculate(inputs)

    assert res.r0_m_min > 0.0
    assert res.r_head_m_min > res.r0_m_min
    assert res.reaction_intensity_btu_ft2_min > 1000.0


# =============================================================================
# LEVEL 3: EXTERNAL GOLDEN BENCHMARKS (SCOTT 2007 / NWCG PMS 436-3 / BEHAVEPLUS)
# =============================================================================

import json
import os

def _load_golden_cases():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "rothermel_golden_cases.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["case_id"]: c for c in data["cases"]}


def test_l3_golden_benchmark_fm1_published_nomograms():
    """
    Verifica Modelo 1 frente a los Nomogramas Oficiales de Scott (2007, RMRS-GTR-192) / NWCG PMS 436-3
    cargados desde el fixture independiente tests/fixtures/rothermel_golden_cases.json.
    """
    cases = _load_golden_cases()
    bed = get_standard_fuel_model(1)
    kernel = RothermelKernel()

    for case_id in ["GB-1.1", "GB-1.2", "GB-1.3", "GB-1.4"]:
        c = cases[case_id]
        inp = c["inputs"]
        moist = FuelMoisture(
            dead_1h=inp["dead_1h"],
            dead_10h=inp["dead_10h"],
            dead_100h=inp["dead_100h"],
            live_herbaceous=inp["live_herbaceous"],
            live_woody=inp["live_woody"]
        )
        res = kernel.calculate(RothermelInputs(
            fuel_bed=bed,
            moisture=moist,
            midflame_wind_ms=inp["midflame_wind_mph"] * units.MPH_TO_M_S,
            slope_degrees=inp["slope_deg"],
            wind_limit_policy=inp["wind_limit_policy"]
        ))
        exp = c["expected_output"]
        tol = c["tolerance_rel"]
        assert math.isclose(res.r_head_chains_hr, exp["ros_chains_hr"], rel_tol=tol), f"Fallo en {case_id}: calc={res.r_head_chains_hr} vs exp={exp['ros_chains_hr']}"
        assert math.isclose(res.r_head_m_min, exp["ros_m_min"], rel_tol=tol), f"Fallo en {case_id}: calc={res.r_head_m_min} vs exp={exp['ros_m_min']}"


def test_l3_golden_benchmark_fm8_published_nomograms():
    """
    Verifica Modelo 8 (multiclase: 1h, 10h, 100h) frente a los Nomogramas Oficiales de Scott (2007, RMRS-GTR-192) / NWCG PMS 436-3
    cargados desde el fixture independiente tests/fixtures/rothermel_golden_cases.json.
    """
    cases = _load_golden_cases()
    bed = get_standard_fuel_model(8)
    kernel = RothermelKernel()

    for case_id in ["GB-8.1", "GB-8.2", "GB-8.3", "GB-8.4"]:
        c = cases[case_id]
        inp = c["inputs"]
        moist = FuelMoisture(
            dead_1h=inp["dead_1h"],
            dead_10h=inp["dead_10h"],
            dead_100h=inp["dead_100h"],
            live_herbaceous=inp["live_herbaceous"],
            live_woody=inp["live_woody"]
        )
        res = kernel.calculate(RothermelInputs(
            fuel_bed=bed,
            moisture=moist,
            midflame_wind_ms=inp["midflame_wind_mph"] * units.MPH_TO_M_S,
            slope_degrees=inp["slope_deg"],
            wind_limit_policy=inp["wind_limit_policy"]
        ))
        exp = c["expected_output"]
        tol = c["tolerance_rel"]
        assert math.isclose(res.r_head_chains_hr, exp["ros_chains_hr"], rel_tol=tol), f"Fallo en {case_id}: calc={res.r_head_chains_hr} vs exp={exp['ros_chains_hr']}"
        assert math.isclose(res.r_head_m_min, exp["ros_m_min"], rel_tol=tol), f"Fallo en {case_id}: calc={res.r_head_m_min} vs exp={exp['ros_m_min']}"


def test_l3_golden_benchmark_fm4_multiclass_live():
    """
    Verifica Modelo 4 (multiclase: 1h, 10h, 100h + live woody con extinción dinámica Albini 1976 / BehavePlus)
    cargado desde el fixture independiente tests/fixtures/rothermel_golden_cases.json.
    """
    cases = _load_golden_cases()
    bed = get_standard_fuel_model(4)
    kernel = RothermelKernel()

    for case_id in ["GB-4.1", "GB-4.2"]:
        c = cases[case_id]
        inp = c["inputs"]
        moist = FuelMoisture(
            dead_1h=inp["dead_1h"],
            dead_10h=inp["dead_10h"],
            dead_100h=inp["dead_100h"],
            live_herbaceous=inp["live_herbaceous"],
            live_woody=inp["live_woody"]
        )
        res = kernel.calculate(RothermelInputs(
            fuel_bed=bed,
            moisture=moist,
            midflame_wind_ms=inp["midflame_wind_mph"] * units.MPH_TO_M_S,
            slope_degrees=inp["slope_deg"],
            wind_limit_policy=inp["wind_limit_policy"]
        ))
        exp = c["expected_output"]
        tol = c["tolerance_rel"]
        assert math.isclose(res.r_head_chains_hr, exp["ros_chains_hr"], rel_tol=tol), f"Fallo en {case_id}: calc={res.r_head_chains_hr} vs exp={exp['ros_chains_hr']}"
        assert math.isclose(res.r_head_m_min, exp["ros_m_min"], rel_tol=tol), f"Fallo en {case_id}: calc={res.r_head_m_min} vs exp={exp['ros_m_min']}"

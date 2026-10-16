"""
tests/test_directional_rothermel.py

Suite de pruebas exhaustivas para la integración direccional de Rothermel:
Resultante Vectorial Viento + Pendiente y Acoplamiento Elíptico Anderson / FARSITE.
"""

import math
import pytest
from fire_engine import units
from fire_engine.fuels.models import get_standard_fuel_model, FuelMoisture, ORIGINAL_13_FUEL_MODELS
from fire_engine.directional import (
    wind_from_to_spread_heading,
    aspect_to_upslope_heading,
    calculate_inverse_effective_wind,
    combine_wind_and_slope_vectors,
    calculate_length_to_breadth_anderson_farsite,
    calculate_ellipse_eccentricity,
    calculate_focus_radial_factor_elliptic,
    calculate_directional_ros,
    DirectionalSpreadInputs,
    DirectionalSpreadResult,
    calculate_directional_rothermel_spread
)
from fire_engine.kernels.rothermel import (
    RothermelKernel,
    RothermelInputs,
    calculate_wind_factor,
    calculate_slope_factor_rothermel
)


# =============================================================================
# 1. CONVENCIONES ANGULARES CARDINALES
# =============================================================================

def test_angular_conventions_wind_and_aspect():
    """
    Verifica las conversiones de rumbos:
      - Viento meteorológico (FROM) -> Avance del fuego (TOWARDS / Downwind)
      - Aspect de pendiente descendente (GIS) -> Avance ladera arriba (Upslope)
    """
    # Viento:
    assert wind_from_to_spread_heading(270.0) == 90.0   # Viento del Oeste -> empuja al Este
    assert wind_from_to_spread_heading(0.0) == 180.0    # Viento del Norte -> empuja al Sur
    assert wind_from_to_spread_heading(90.0) == 270.0   # Viento del Este  -> empuja al Oeste
    assert wind_from_to_spread_heading(180.0) == 0.0    # Viento del Sur   -> empuja al Norte
    assert wind_from_to_spread_heading(360.0) == 180.0

    # Aspect:
    assert aspect_to_upslope_heading(0.0) == 180.0      # Ladera mirando al Norte -> sube al Sur
    assert aspect_to_upslope_heading(270.0) == 90.0     # Ladera mirando al Oeste -> sube al Este
    assert aspect_to_upslope_heading(90.0) == 270.0     # Ladera mirando al Este  -> sube al Oeste
    assert aspect_to_upslope_heading(180.0) == 0.0      # Ladera mirando al Sur   -> sube al Norte


# =============================================================================
# 2. INVERSIÓN ANALÍTICA DE VIENTO (ROUNDTRIP)
# =============================================================================

def test_inverse_effective_wind_roundtrip():
    """
    Verifica que la inversión U -> phi_w(U) -> inverse_phi_w(phi_w) recupere exactamente
    la velocidad de viento original para múltiples modelos de combustible.
    """
    kernel = RothermelKernel()
    moist = FuelMoisture(dead_1h=0.08, dead_10h=0.09, dead_100h=0.10)

    for model_id in [1, 4, 8]:
        bed = get_standard_fuel_model(model_id)
        base = kernel.calculate(RothermelInputs(fuel_bed=bed, moisture=moist, midflame_wind_ms=0.0))

        for u_mph in [1.0, 2.0, 5.0, 10.0, 20.0]:
            u_ft_min = u_mph * units.MPH_TO_FT_MIN
            phi_w, _, _, _ = calculate_wind_factor(
                u_ft_min, base.characteristic_sigma_inv_ft, base.beta, base.beta_opt
            )
            u_recovered_ft_min = calculate_inverse_effective_wind(
                phi_w, base.characteristic_sigma_inv_ft, base.beta, base.beta_opt
            )
            u_recovered_mph = u_recovered_ft_min / units.MPH_TO_FT_MIN
            assert math.isclose(u_recovered_mph, u_mph, rel_tol=1e-5), f"Fallo en FM{model_id} U={u_mph} mph"


# =============================================================================
# 3. CASOS ANALÍTICOS OBLIGATORIOS DE COMBINACIÓN VECTORIAL
# =============================================================================

def test_case_flat_no_wind():
    """Caso 1: Sin viento y sin pendiente -> Isotropía perfecta."""
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)
    inputs = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=0.0,
        slope_degrees=0.0
    )
    res = calculate_directional_rothermel_spread(inputs)

    assert res.phi_resultant == 0.0
    assert res.effective_wind_ms == 0.0
    assert res.length_to_breadth == 1.0
    assert res.eccentricity == 0.0
    assert math.isclose(res.r_head_m_min, res.r0_m_min, rel_tol=1e-5)

    # R(theta) constante e igual a R0 en todas las direcciones
    for angle in [0.0, 45.0, 90.0, 180.0, 270.0]:
        assert math.isclose(res.ros_at_heading(angle), res.r0_m_min, rel_tol=1e-5)


def test_case_wind_only():
    """Caso 2: Solo viento (Viento del Oeste 270° -> Avance al Este 90°)."""
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)
    inputs = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=5.0 * units.MPH_TO_M_S,
        wind_from_direction_deg=270.0,
        slope_degrees=0.0
    )
    res = calculate_directional_rothermel_spread(inputs)

    assert res.phi_s == 0.0
    assert res.phi_resultant == pytest.approx(res.phi_w, rel=1e-5)
    assert res.heading_resultant_deg == 90.0
    assert res.effective_wind_mph == pytest.approx(5.0, rel=1e-4)
    assert res.length_to_breadth == pytest.approx(1.0 + 0.125 * 5.0, rel=1e-4)


def test_case_slope_only():
    """Caso 3: Solo pendiente (Ladera mirando al Oeste 270° -> Subida al Este 90°)."""
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)
    inputs = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=0.0,
        slope_degrees=20.0,
        aspect_downslope_degrees=270.0
    )
    res = calculate_directional_rothermel_spread(inputs)

    assert res.phi_w == 0.0
    assert res.phi_resultant == pytest.approx(res.phi_s, rel=1e-5)
    assert res.heading_resultant_deg == 90.0
    assert res.effective_wind_ms > 0.0
    assert res.length_to_breadth > 1.0


def test_case_aligned_wind_and_slope():
    """Caso 4: Viento y pendiente alineados al Este (90°)."""
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)
    inputs = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=2.0 * units.MPH_TO_M_S,
        wind_from_direction_deg=270.0,           # Empuja al Este (90°)
        slope_degrees=15.0,
        aspect_downslope_degrees=270.0,          # Sube al Este (90°)
        wind_limit_policy="none"
    )
    res = calculate_directional_rothermel_spread(inputs)

    assert res.phi_resultant == pytest.approx(res.phi_w + res.phi_s, rel=1e-5)
    assert res.heading_resultant_deg == 90.0
    assert math.isclose(res.r_head_ft_min, res.r0_ft_min * (1.0 + res.phi_w + res.phi_s), rel_tol=1e-5)


def test_case_opposed_wind_and_slope():
    """Caso 5: Viento y pendiente opuestos a 180°."""
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)
    
    # Sub-caso 5A: Viento dominante sobre la pendiente
    inputs_a = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=5.0 * units.MPH_TO_M_S,  # phi_w ≈ 21.4 (Este)
        wind_from_direction_deg=270.0,            # Empuja al Este (90°)
        slope_degrees=10.0,                       # phi_s ≈ 0.2 (Oeste)
        aspect_downslope_degrees=90.0,            # Sube al Oeste (270°)
        wind_limit_policy="none"
    )
    res_a = calculate_directional_rothermel_spread(inputs_a)
    assert res_a.phi_resultant == pytest.approx(abs(res_a.phi_w - res_a.phi_s), rel=1e-5)
    assert res_a.heading_resultant_deg == 90.0    # Sigue al viento dominante

    # Sub-caso 5B: Pendiente dominante sobre el viento
    inputs_b = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=0.5 * units.MPH_TO_M_S,  # Viento muy suave al Este (90°)
        wind_from_direction_deg=270.0,
        slope_degrees=30.0,                       # Pendiente fuerte subiendo al Oeste (270°)
        aspect_downslope_degrees=90.0,
        wind_limit_policy="none"
    )
    res_b = calculate_directional_rothermel_spread(inputs_b)
    assert res_b.phi_resultant == pytest.approx(abs(res_b.phi_s - res_b.phi_w), rel=1e-5)
    assert res_b.heading_resultant_deg == 270.0   # Sigue a la pendiente dominante


def test_case_exact_cancellation():
    """Caso 6: Cancelación exacta cuando phi_w == phi_s a 180°."""
    # Sintetizamos un VectorResultant con phi_w == phi_s a 180°
    vec_res = combine_wind_and_slope_vectors(
        r0_ft_min=1.0,
        reaction_intensity_btu_ft2_min=1000.0,
        phi_w=2.5,
        wind_spread_heading_deg=90.0,   # Este
        phi_s=2.5,
        upslope_heading_deg=270.0,      # Oeste
        sigma_inv_ft=2000.0,
        beta=0.01,
        beta_opt=0.005,
        wind_limit_policy="none"
    )

    assert vec_res.phi_resultant == 0.0
    assert vec_res.heading_resultant_deg == 0.0  # Rumbo neutro sin fantasma
    assert vec_res.effective_wind_ft_min == 0.0
    assert math.isclose(vec_res.r_head_ft_min, 1.0, rel_tol=1e-6)


def test_case_perpendicular_wind_and_slope():
    """Caso 7: Viento al Norte (0°) y pendiente al Este (90°)."""
    vec_res = combine_wind_and_slope_vectors(
        r0_ft_min=1.0,
        reaction_intensity_btu_ft2_min=1000.0,
        phi_w=3.0,
        wind_spread_heading_deg=0.0,    # Norte (y = +3.0)
        phi_s=4.0,
        upslope_heading_deg=90.0,       # Este (x = +4.0)
        sigma_inv_ft=2000.0,
        beta=0.01,
        beta_opt=0.005,
        wind_limit_policy="none"
    )

    # Triángulo 3-4-5
    assert vec_res.phi_resultant == pytest.approx(5.0, rel=1e-5)
    # Rumbo = atan2(4, 3) ≈ 53.13° (Noreste)
    exp_heading = math.degrees(math.atan2(4.0, 3.0))
    assert vec_res.heading_resultant_deg == pytest.approx(exp_heading, rel=1e-5)


# =============================================================================
# 4. WIND LIMIT EN LA RESULTANTE
# =============================================================================

def test_wind_limit_on_resultant():
    """
    Verifica que la política wind_limit_policy opere sobre la resultante vectorial combinada:
    Para FM1 con viento fuerte y pendiente alineada, U_eff excede U_max.
    """
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)

    # Con límite activado (rothermel_limit)
    inputs_capped = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=10.0 * units.MPH_TO_M_S,
        wind_from_direction_deg=270.0,
        slope_degrees=15.0,
        aspect_downslope_degrees=270.0,
        wind_limit_policy="rothermel_limit"
    )
    res_capped = calculate_directional_rothermel_spread(inputs_capped)
    assert res_capped.wind_limit_active is True
    assert res_capped.effective_wind_mph == pytest.approx(7.804, rel=1e-2)

    # Sin límite (none)
    inputs_uncapped = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=10.0 * units.MPH_TO_M_S,
        wind_from_direction_deg=270.0,
        slope_degrees=15.0,
        aspect_downslope_degrees=270.0,
        wind_limit_policy="none"
    )
    res_uncapped = calculate_directional_rothermel_spread(inputs_uncapped)
    assert res_uncapped.wind_limit_active is False
    assert res_uncapped.effective_wind_mph > 10.0
    assert res_uncapped.r_head_m_min > res_capped.r_head_m_min


# =============================================================================
# 5. INVARIANZA POR ROTACIÓN
# =============================================================================

def test_rotational_invariance():
    """
    Verifica que al rotar simultáneamente viento, orientación de ladera y dirección consultada
    en múltiplos de 45°, los escalares permanezcan rigurosamente invariantes.
    """
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)

    base_wind_from = 270.0   # Viento del Oeste
    base_aspect = 180.0      # Aspect al Sur (sube al Norte)

    scalars_ref = None

    for rot in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
        inputs = DirectionalSpreadInputs(
            fuel_bed=bed,
            moisture=moist,
            midflame_wind_ms=3.0 * units.MPH_TO_M_S,
            wind_from_direction_deg=(base_wind_from + rot) % 360.0,
            slope_degrees=12.0,
            aspect_downslope_degrees=(base_aspect + rot) % 360.0,
            wind_limit_policy="none"
        )
        res = calculate_directional_rothermel_spread(inputs)

        scalars = (
            round(res.phi_resultant, 5),
            round(res.effective_wind_mph, 5),
            round(res.r_head_m_min, 5),
            round(res.length_to_breadth, 5),
            round(res.eccentricity, 5)
        )

        if scalars_ref is None:
            scalars_ref = scalars
            heading_ref = res.heading_resultant_deg
        else:
            assert scalars == scalars_ref, f"Rotación {rot}° rompió invariancia escalar: {scalars} vs {scalars_ref}"
            exp_heading = (heading_ref + rot) % 360.0
            assert math.isclose(res.heading_resultant_deg, exp_heading, abs_tol=1e-4)


# =============================================================================
# 6. CONTINUIDAD ANGULAR EN 0° / 360°
# =============================================================================

def test_angular_continuity_across_zero():
    """Verifica que no exista salto ni discontinuidad numérica al cruzar 0°/360°."""
    bed = get_standard_fuel_model(1)
    moist = FuelMoisture(dead_1h=0.08)
    inputs = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=5.0 * units.MPH_TO_M_S,
        wind_from_direction_deg=180.0,  # Empuja exactamente al Norte (0°)
        slope_degrees=0.0
    )
    res = calculate_directional_rothermel_spread(inputs)

    # R(359.99°) debe ser idéntico a R(0.01°)
    r_left = res.ros_at_heading(359.99)
    r_center = res.ros_at_heading(0.0)
    r_right = res.ros_at_heading(0.01)

    assert math.isclose(r_left, r_right, rel_tol=1e-4)
    assert math.isclose(r_left, r_center, rel_tol=1e-4)


# =============================================================================
# 7. PROPIEDADES GEOMÉTRICAS DE LA ELIPSE DE FUEGO
# =============================================================================

def test_elliptic_shape_hierarchy_and_symmetry():
    """
    Verifica las propiedades canónicas de la elipse Anderson/FARSITE:
      - R(head) == R_head
      - R(back) < R(flank) < R(head)
      - Simetría bilateral: R(head + delta) == R(head - delta)
    """
    bed = get_standard_fuel_model(4)
    moist = FuelMoisture(dead_1h=0.06, dead_10h=0.07, dead_100h=0.08, live_woody=1.00)
    inputs = DirectionalSpreadInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=5.0 * units.MPH_TO_M_S,
        wind_from_direction_deg=270.0,  # Heading = 90° (Este)
        slope_degrees=0.0
    )
    res = calculate_directional_rothermel_spread(inputs)

    r_head = res.ros_at_heading(90.0)
    r_flank_right = res.ros_at_heading(180.0)  # +90°
    r_flank_left = res.ros_at_heading(0.0)     # -90°
    r_backing = res.ros_at_heading(270.0)     # 180°

    assert math.isclose(r_head, res.r_head_m_min, rel_tol=1e-5)
    assert math.isclose(r_flank_right, r_flank_left, rel_tol=1e-5)
    assert r_backing < r_flank_right < r_head

    # Simetría bilateral a ±30°, ±45°, ±60°
    for delta in [15.0, 30.0, 45.0, 60.0, 75.0]:
        r_pos = res.ros_at_heading(90.0 + delta)
        r_neg = res.ros_at_heading(90.0 - delta)
        assert math.isclose(r_pos, r_neg, rel_tol=1e-5)


# =============================================================================
# 8. SANIDAD DIRECCIONAL EN TODOS LOS 13 MODELOS
# =============================================================================

def test_directional_spread_all_13_models():
    """Ejecuta un escenario direccional oblicuo en los 13 modelos verificando admisibilidad física."""
    moist = FuelMoisture(dead_1h=0.08, dead_10h=0.09, dead_100h=0.10, live_herbaceous=1.00, live_woody=1.00)

    for code in range(1, 14):
        bed = ORIGINAL_13_FUEL_MODELS[code]
        inputs = DirectionalSpreadInputs(
            fuel_bed=bed,
            moisture=moist,
            midflame_wind_ms=3.0 * units.MPH_TO_M_S,
            wind_from_direction_deg=225.0,   # Viento del Suroeste -> avanza al Noreste (45°)
            slope_degrees=15.0,
            aspect_downslope_degrees=180.0,  # Sube al Norte (0°)
            wind_limit_policy="rothermel_limit"
        )
        res = calculate_directional_rothermel_spread(inputs)

        assert math.isfinite(res.r_head_m_min) and res.r_head_m_min > 0.0
        assert math.isfinite(res.effective_wind_ms) and res.effective_wind_ms > 0.0
        assert 0.0 <= res.heading_resultant_deg < 360.0
        # El rumbo debe quedar entre 0° (Norte) y 45° (Noreste)
        assert 0.0 <= res.heading_resultant_deg <= 45.0
        assert res.length_to_breadth >= 1.0
        assert 0.0 <= res.eccentricity < 1.0

        for angle in [0.0, 45.0, 90.0, 180.0, 270.0]:
            ros = res.ros_at_heading(angle)
            assert math.isfinite(ros) and ros >= 0.0


# =============================================================================
# 9. VALIDACIÓN END-TO-END DE ASPECT DESDE DEM SINTÉTICO
# =============================================================================

def test_end_to_end_aspect_from_dem_cases_east_and_north():
    """
    Verifica end-to-end la cadena completa:
      DEM sintético -> calculate_slope_and_aspect_from_dem() -> aspect (downslope) -> aspect_to_upslope_heading() -> upslope
    
    Caso A: Elevación creciente hacia el Este (Columnas aumentan de 100 a 120)
      - Físico: El terreno sube hacia el Este (upslope = 90°).
      - Aspect (downslope): El agua/pendiente cae hacia el Oeste (270°).
      - Verificación: calculated_aspect ≈ 270°, upslope_heading ≈ 90°.
    
    Caso B: Elevación creciente hacia el Norte (Filas 0=120, 1=110, 2=100)
      - Físico: El terreno sube hacia el Norte (upslope = 0°).
      - Aspect (downslope): El agua/pendiente cae hacia el Sur (180°).
      - Verificación: calculated_aspect ≈ 180°, upslope_heading ≈ 0°.
    """
    import numpy as np
    from fire_engine.directional import calculate_slope_and_aspect_from_dem

    # Caso A: Creciente al Este (dx = 30m, dz = 10m por celda)
    dem_east = np.array([
        [100.0, 110.0, 120.0],
        [100.0, 110.0, 120.0],
        [100.0, 110.0, 120.0]
    ], dtype=np.float32)

    slope_a, aspect_a = calculate_slope_and_aspect_from_dem(dem_east, cell_size_m=30.0)
    calc_aspect_a = float(aspect_a[1, 1])
    calc_upslope_a = aspect_to_upslope_heading(calc_aspect_a)

    assert math.isclose(calc_aspect_a, 270.0, abs_tol=1e-3), f"Caso A: aspect={calc_aspect_a} != 270° (Oeste)"
    assert math.isclose(calc_upslope_a, 90.0, abs_tol=1e-3), f"Caso A: upslope={calc_upslope_a} != 90° (Este)"

    # Caso B: Creciente al Norte (Fila 0 = Norte = 120m, Fila 2 = Sur = 100m)
    dem_north = np.array([
        [120.0, 120.0, 120.0],
        [110.0, 110.0, 110.0],
        [100.0, 100.0, 100.0]
    ], dtype=np.float32)

    slope_b, aspect_b = calculate_slope_and_aspect_from_dem(dem_north, cell_size_m=30.0)
    calc_aspect_b = float(aspect_b[1, 1])
    calc_upslope_b = aspect_to_upslope_heading(calc_aspect_b)

    assert math.isclose(calc_aspect_b, 180.0, abs_tol=1e-3), f"Caso B: aspect={calc_aspect_b} != 180° (Sur)"
    assert math.isclose(calc_upslope_b, 0.0, abs_tol=1e-3), f"Caso B: upslope={calc_upslope_b} != 0° (Norte)"


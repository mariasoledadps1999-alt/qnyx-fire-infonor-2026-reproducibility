"""
src/fire_engine/kernels/rothermel.py

Implementación analítica y modular del modelo físico de propagación de incendios
de Rothermel (1972 INT-115 / Albini 1976 INT-30 / Andrews 2018 RMRS-GTR-371).

Este módulo es completamente independiente del ráster y opera mediante funciones puras
y estructuras de datos fuertemente tipadas.
"""

import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Dict, Any, Optional

from ..units import (
    FEET_TO_METERS, METERS_TO_FEET,
    FT_MIN_TO_M_MIN, M_MIN_TO_FT_MIN,
    M_S_TO_FT_MIN, FT_MIN_TO_M_S,
    M_S_TO_MPH, MPH_TO_M_S,
    FT_MIN_TO_CHAINS_HR, CHAINS_HR_TO_FT_MIN,
    CHAINS_HR_TO_M_MIN,
    BTU_FT2_MIN_TO_KW_M2, LB_FT3_TO_KG_M3
)
from ..fuels.models import FuelParticleClass, RothermelFuelBed, FuelMoisture, apply_dynamic_fuel_transfer
from .base import SpreadKernel, SpreadResult


# =============================================================================
# ESTRUCTURAS TIPADAS DE ENTRADA Y SALIDA
# =============================================================================

@dataclass(frozen=True)
class RothermelInputs:
    """Parámetros de entrada para el cálculo de avance puntual de Rothermel."""
    fuel_bed: RothermelFuelBed
    moisture: FuelMoisture
    midflame_wind_ms: float = 0.0           # Velocidad a media llama [m/s]
    slope_degrees: float = 0.0              # Pendiente del terreno [grados sexagesimales]
    wind_limit_policy: str = "rothermel_limit"  # "rothermel_limit" (0.9*IR) o "none"


@dataclass(frozen=True)
class RothermelResult(SpreadResult):
    """
    Resultado detallado del kernel de Rothermel conteniendo todas las variables diagnósticas intermedias.
    Hereda de SpreadResult para proveer una interfaz unificada.
    """
    # Tasas de propagación (Nativas ft/min y SI m/min)
    r0_ft_min: float = 0.0
    r0_m_min: float = 0.0
    r0_chains_hr: float = 0.0
    
    r_head_ft_min: float = 0.0
    r_head_m_min: float = 0.0
    r_head_chains_hr: float = 0.0

    # Propiedades del lecho
    rho_b_lb_ft3: float = 0.0               # Densidad aparente del lecho [lb/ft³]
    beta: float = 0.0                       # Relación de empaquetamiento adimensional
    beta_opt: float = 0.0                   # Relación de empaquetamiento óptima
    characteristic_sigma_inv_ft: float = 0.0 # SAV ponderado [1/ft]

    # Intensidad y velocidades de reacción
    gamma_max_min1: float = 0.0             # Velocidad de reacción máxima [min⁻¹]
    reaction_exponent_A: float = 0.0        # Exponente de reacción A
    gamma_prime_min1: float = 0.0           # Velocidad de reacción óptima [min⁻¹]
    net_fuel_load_lb_ft2: float = 0.0       # Carga neta wn [lb/ft²]
    eta_M: float = 0.0                      # Amortiguamiento por humedad
    eta_S: float = 0.0                      # Amortiguamiento por minerales
    reaction_intensity_btu_ft2_min: float = 0.0 # Intensidad de reacción IR [Btu/(ft² min)]
    reaction_intensity_kw_m2: float = 0.0   # Intensidad de reacción IR [kW/m²]

    # Flujo propagante y sumidero de calor
    propagating_flux_ratio_xi: float = 0.0  # Fracción de flujo propagante xi
    effective_heating_number_eps: float = 0.0 # Número de calentamiento efectivo epsilon
    heat_of_preignition_qig_btu_lb: float = 0.0 # Calor de preignición Qig [Btu/lb]
    heat_sink_btu_ft3: float = 0.0          # Sumidero volumétrico de calor (rho_b * eps * Qig)

    # Factores ambientales
    wind_coef_C: float = 0.0                # Coeficiente C de viento
    wind_coef_B: float = 0.0                # Exponente B de viento
    wind_coef_E: float = 0.0                # Exponente E de viento
    phi_w_raw: float = 0.0                  # Factor de viento sin truncamiento
    phi_w: float = 0.0                      # Factor de viento efectivo
    phi_s: float = 0.0                      # Factor de pendiente

    # Diagnósticos de viento
    input_midflame_wind_ms: float = 0.0     # Viento solicitado [m/s]
    input_midflame_wind_ft_min: float = 0.0 # Viento solicitado [ft/min]
    input_midflame_wind_mph: float = 0.0    # Viento solicitado [mph]
    wind_limit_threshold_ft_min: float = 0.0 # Umbral de corte U_max = 0.9*IR [ft/min]
    wind_limit_threshold_mph: float = 0.0   # Umbral de corte [mph]
    wind_limit_active: bool = False         # ¿Se superó el límite operacional?
    effective_midflame_wind_ms: float = 0.0 # Viento efectivo aplicado [m/s]
    effective_midflame_wind_ft_min: float = 0.0 # Viento efectivo aplicado [ft/min]
    effective_midflame_wind_mph: float = 0.0 # Viento efectivo aplicado [mph]


# =============================================================================
# FUNCIONES MATEMÁTICAS UNITARIAS AISLADAS (INT-115 / INT-30 / GTR-371)
# =============================================================================

def calculate_bulk_density(w0_lb_ft2: float, depth_ft: float) -> float:
    """
    Calcula la densidad aparente del lecho de combustible (rho_b).
    Fuente: INT-115 p. 2, Eq. 2.
    Unidad: [lb/ft³]
    """
    if depth_ft <= 0.0:
        return 0.0
    return w0_lb_ft2 / depth_ft


def calculate_packing_ratio(rho_b_lb_ft3: float, rho_p_lb_ft3: float = 32.0) -> float:
    """
    Calcula la relación de empaquetamiento adimensional (beta).
    Fuente: INT-115 p. 2, Eq. 3.
    """
    if rho_p_lb_ft3 <= 0.0:
        return 0.0
    return rho_b_lb_ft3 / rho_p_lb_ft3


def calculate_characteristic_sav(particle_classes: Sequence[FuelParticleClass]) -> float:
    """
    Calcula la relación Superficie-a-Volumen característica ponderada del lecho (sigma).
    Fuente: INT-115 p. 20, Eq. 38.
    Unidad: [1/ft]
    """
    total_area_weight = sum(p.load_lb_ft2 * p.sav_inv_ft for p in particle_classes)
    if total_area_weight <= 0.0:
        return 0.0
    sum_sigma_squared = sum(p.load_lb_ft2 * (p.sav_inv_ft ** 2) for p in particle_classes)
    return sum_sigma_squared / total_area_weight


def calculate_optimum_packing_ratio(sigma_inv_ft: float) -> float:
    """
    Calcula la relación de empaquetamiento óptima (beta_op).
    Fuente: INT-115 p. 20, Eq. 37.
    """
    if sigma_inv_ft <= 0.0:
        return 0.0
    return 3.348 * (sigma_inv_ft ** -0.8189)


def calculate_max_reaction_velocity(sigma_inv_ft: float) -> float:
    """
    Calcula la velocidad de reacción máxima (Gamma_max).
    Fuente: INT-115 p. 19, Eq. 36.
    Unidad: [min⁻¹]
    """
    if sigma_inv_ft <= 0.0:
        return 0.0
    sigma_1_5 = sigma_inv_ft ** 1.5
    return sigma_1_5 / (495.0 + 0.0594 * sigma_1_5)


def calculate_reaction_exponent(sigma_inv_ft: float) -> float:
    """
    Calcula el exponente de reacción (A).
    Fuente: INT-115 p. 20, Eq. 39.
    """
    if sigma_inv_ft <= 0.0:
        return 0.0
    return 133.0 * (sigma_inv_ft ** -0.7913)


def calculate_optimum_reaction_velocity(sigma_inv_ft: float, beta: float, beta_op: float) -> float:
    """
    Calcula la velocidad de reacción óptima (Gamma').
    Fuente: INT-115 p. 19, Eq. 35.
    Unidad: [min⁻¹]
    """
    if beta <= 0.0 or beta_op <= 0.0 or sigma_inv_ft <= 0.0:
        return 0.0
    gamma_max = calculate_max_reaction_velocity(sigma_inv_ft)
    A = calculate_reaction_exponent(sigma_inv_ft)
    ratio = beta / beta_op
    return gamma_max * (ratio ** A) * math.exp(A * (1.0 - ratio))


def calculate_moisture_damping(mf: float, mx: float) -> float:
    """
    Calcula el factor de amortiguamiento por humedad (eta_M).
    Fuente: INT-115 p. 16, Eq. 29.
    """
    if mx <= 0.0 or mf >= mx:
        return 0.0
    rm = mf / mx
    if rm < 0.0:
        rm = 0.0
    damping = 1.0 - 2.59 * rm + 5.11 * (rm ** 2) - 3.52 * (rm ** 3)
    return max(0.0, min(1.0, damping))


def calculate_mineral_damping(se: float = 0.010) -> float:
    """
    Calcula el factor de amortiguamiento por minerales (eta_S).
    Fuente: INT-115 p. 17, Eq. 30.
    """
    if se <= 0.0:
        return 1.0
    damping = 0.174 * (se ** -0.19)
    return max(0.0, min(1.0, damping))


def calculate_net_fuel_load(particle_classes: Sequence[FuelParticleClass], st: float = 0.0555) -> float:
    """
    Calcula la carga neta total de combustible seco (wn).
    Fuente: INT-115 p. 15, Eq. 26.
    Unidad: [lb/ft²]
    """
    return sum(p.load_lb_ft2 * (1.0 - st) for p in particle_classes)


def calculate_propagating_flux_ratio(sigma_inv_ft: float, beta: float) -> float:
    """
    Calcula la fracción de flujo propagante (xi).
    Fuente: INT-115 p. 22, Eq. 42.
    """
    if sigma_inv_ft <= 0.0:
        return 0.0
    denom = 192.0 + 0.2595 * sigma_inv_ft
    exp_term = (0.792 + 0.681 * math.sqrt(sigma_inv_ft)) * (beta + 0.1)
    return (1.0 / denom) * math.exp(exp_term)


def calculate_effective_heating_number(particle_classes: Sequence[FuelParticleClass]) -> float:
    """
    Calcula el número de calentamiento efectivo ponderado (epsilon).
    Fuente: INT-115 p. 9, Eq. 14.
    """
    total_area_weight = sum(p.load_lb_ft2 * p.sav_inv_ft for p in particle_classes)
    if total_area_weight <= 0.0:
        return 0.0
    
    eps_sum = 0.0
    for p in particle_classes:
        f_k = (p.load_lb_ft2 * p.sav_inv_ft) / total_area_weight
        eps_k = math.exp(-138.0 / p.sav_inv_ft) if p.sav_inv_ft > 0.0 else 0.0
        eps_sum += f_k * eps_k
    return eps_sum


def calculate_heat_of_preignition(particle_classes: Sequence[FuelParticleClass], moisture: FuelMoisture) -> float:
    """
    Calcula el calor de preignición ponderado (Q_ig).
    Fuente: INT-115 p. 8, Eq. 12.
    Unidad: [Btu/lb]
    """
    total_area_weight = sum(p.load_lb_ft2 * p.sav_inv_ft for p in particle_classes)
    if total_area_weight <= 0.0:
        return 250.0
    
    qig_sum = 0.0
    for p in particle_classes:
        f_k = (p.load_lb_ft2 * p.sav_inv_ft) / total_area_weight
        mf_k = moisture.get_moisture_for_class(p)
        qig_k = 250.0 + 1116.0 * mf_k
        qig_sum += f_k * qig_k
    return qig_sum


def calculate_wind_coefficients(sigma_inv_ft: float) -> Tuple[float, float, float]:
    """
    Calcula los coeficientes y exponentes de viento canónicos C, B, E.
    Fuente: INT-115 p. 24-25, Eq. 48, 49, 50 / GTR-371 p. 28, Eq. 33, 34, 35.
    """
    if sigma_inv_ft <= 0.0:
        return (0.0, 0.0, 0.0)
    C = 7.47 * math.exp(-0.133 * (sigma_inv_ft ** 0.55))
    B = 0.02526 * (sigma_inv_ft ** 0.54)
    E = 0.715 * math.exp(-3.59e-4 * sigma_inv_ft)
    return (C, B, E)


def calculate_wind_factor(U_ft_min: float, sigma_inv_ft: float, beta: float, beta_op: float) -> Tuple[float, float, float, float]:
    """
    Calcula el factor adimensional de viento (phi_w).
    Fuente: INT-115 p. 24, Eq. 47.
    Retorna: (phi_w, C, B, E)
    """
    if U_ft_min <= 0.0 or sigma_inv_ft <= 0.0 or beta <= 0.0 or beta_op <= 0.0:
        C, B, E = calculate_wind_coefficients(sigma_inv_ft)
        return (0.0, C, B, E)
    
    C, B, E = calculate_wind_coefficients(sigma_inv_ft)
    ratio_term = (beta / beta_op) ** (-E)
    phi_w = C * (U_ft_min ** B) * ratio_term
    return (phi_w, C, B, E)


def calculate_slope_factor_rothermel(slope_degrees: float, beta: float) -> float:
    """
    Calcula el factor adimensional de pendiente (phi_s).
    Fuente: INT-115 p. 25, Eq. 51.
    """
    if slope_degrees <= 0.0 or beta <= 0.0:
        return 0.0
    tan_phi = math.tan(math.radians(slope_degrees))
    return 5.275 * (beta ** -0.3) * (tan_phi ** 2)


def calculate_wind_limit(reaction_intensity_btu_ft2_min: float) -> float:
    """
    Calcula la velocidad máxima de viento confiable U_max en ft/min.
    Fuente: Rothermel (1972 INT-115, p. 24-25) / Andrews (2018 GTR-371, p. 29, Eq. 36).
    """
    return 0.9 * max(0.0, reaction_intensity_btu_ft2_min)


def calculate_live_fuel_extinction_moisture(
    particle_classes: Sequence[FuelParticleClass],
    moisture: FuelMoisture,
    dead_extinction_moisture: float,
    total_mineral_content: float = 0.0555
) -> float:
    """
    Calcula la humedad de extinción dinámica de los combustibles vivos (M_{x, live})
    a partir de la carga y humedad de los combustibles finos muertos activos.
    Fuente: Albini (1976 INT-30, p. 88) / Andrews (2018 RMRS-GTR-371, p. 35, Eq. 47).
    """
    dead_classes = [p for p in particle_classes if not p.is_live]
    live_classes = [p for p in particle_classes if p.is_live]

    if not live_classes or not dead_classes or dead_extinction_moisture <= 0.0:
        return dead_extinction_moisture

    w_fine_dead = sum(p.load_lb_ft2 * (1.0 - total_mineral_content) * math.exp(-138.0 / p.sav_inv_ft) for p in dead_classes)
    w_fine_live = sum(p.load_lb_ft2 * (1.0 - total_mineral_content) * math.exp(-138.0 / p.sav_inv_ft) for p in live_classes)

    if w_fine_dead <= 0.0 or w_fine_live <= 0.0:
        return dead_extinction_moisture

    mf_dead_fine = sum(
        moisture.get_moisture_for_class(p) * p.load_lb_ft2 * (1.0 - total_mineral_content) * math.exp(-138.0 / p.sav_inv_ft)
        for p in dead_classes
    ) / w_fine_dead

    alpha = w_fine_dead / w_fine_live
    dead_ratio = mf_dead_fine / dead_extinction_moisture
    if dead_ratio >= 1.0:
        return dead_extinction_moisture

    mx_live = 2.9 * alpha * (1.0 - dead_ratio) - 0.226
    return max(dead_extinction_moisture, mx_live)


# =============================================================================
# MOTOR DE CÁLCULO COMPLETO Y CLASE ROTHERMELKERNEL
# =============================================================================

def calculate_rothermel_spread(inputs: RothermelInputs) -> RothermelResult:
    """
    Ejecuta el cálculo completo y exhaustivo del modelo de Rothermel (1972).
    """
    bed = inputs.fuel_bed
    moist = inputs.moisture

    # 0. Combustible no propagable (Non-burnable) o sin carga / profundidad
    if not bed.is_burnable or bed.total_load_lb_ft2 <= 0.0 or bed.depth_ft <= 0.0:
        return RothermelResult(
            base_ros_m_min=0.0,
            head_ros_m_min=0.0,
            wind_factor=0.0,
            slope_factor=0.0,
            effective_heading_deg=0.0,
            diagnostics={
                "burnable": False,
                "r0_ft_min": 0.0,
                "r_head_ft_min": 0.0,
                "I_R": 0.0,
                "I_R_kw_m2": 0.0
            },
            r0_ft_min=0.0,
            r0_m_min=0.0,
            r0_chains_hr=0.0,
            r_head_ft_min=0.0,
            r_head_m_min=0.0,
            r_head_chains_hr=0.0
        )

    # 0b. Transferencia dinámica de combustible herbáceo (Scott & Burgan 2005)
    if bed.is_dynamic:
        bed = apply_dynamic_fuel_transfer(bed, moist)

    # 1. Propiedades del lecho
    w0_total = bed.total_load_lb_ft2
    depth = bed.depth_ft
    rho_b = calculate_bulk_density(w0_total, depth)
    beta = calculate_packing_ratio(rho_b, bed.particle_density_lb_ft3)
    sigma = calculate_characteristic_sav(bed.particle_classes)
    beta_op = calculate_optimum_packing_ratio(sigma)

    # 2. Velocidad y Reacción
    gamma_max = calculate_max_reaction_velocity(sigma)
    A = calculate_reaction_exponent(sigma)
    gamma_prime = calculate_optimum_reaction_velocity(sigma, beta, beta_op)
    wn_total = calculate_net_fuel_load(bed.particle_classes, bed.total_mineral_content)
    eta_S = calculate_mineral_damping(bed.effective_mineral_content)

    # 3. Humedad de extinción para vivos vs muertos
    mx_dead = bed.dead_extinction_moisture
    mx_live = calculate_live_fuel_extinction_moisture(
        bed.particle_classes, moist, mx_dead, bed.total_mineral_content
    )

    # 4. Intensidad de reacción integrada por estrato
    total_area_weight = sum(p.load_lb_ft2 * p.sav_inv_ft for p in bed.particle_classes)
    ir_dead = 0.0
    ir_live = 0.0
    eta_m_weighted = 0.0

    for p in bed.particle_classes:
        wn_k = p.load_lb_ft2 * (1.0 - bed.total_mineral_content)
        mf_k = moist.get_moisture_for_class(p)
        mx_k = mx_live if p.is_live else mx_dead
        eta_m_k = calculate_moisture_damping(mf_k, mx_k)
        
        ir_k = gamma_prime * wn_k * p.heat_btu_lb * eta_m_k * eta_S
        if p.is_live:
            ir_live += ir_k
        else:
            ir_dead += ir_k

        if total_area_weight > 0.0:
            f_k = (p.load_lb_ft2 * p.sav_inv_ft) / total_area_weight
            eta_m_weighted += f_k * eta_m_k

    IR = max(0.0, ir_dead + ir_live)
    IR_kw = IR * BTU_FT2_MIN_TO_KW_M2

    # 5. Flujo Propagante y Sumidero de Calor
    xi = calculate_propagating_flux_ratio(sigma, beta)
    eps = calculate_effective_heating_number(bed.particle_classes)
    qig = calculate_heat_of_preignition(bed.particle_classes, moist)
    heat_sink = rho_b * eps * qig

    # 6. Tasa base R0 (sin viento ni pendiente)
    if heat_sink > 0.0 and IR > 0.0:
        r0_ft_min = (IR * xi) / heat_sink
    else:
        r0_ft_min = 0.0
    r0_m_min = r0_ft_min * FT_MIN_TO_M_MIN
    r0_ch_hr = r0_ft_min * FT_MIN_TO_CHAINS_HR

    # 6. Factor de Viento y Política de Límite de Viento
    u_req_ms = max(0.0, inputs.midflame_wind_ms)
    u_req_ft_min = u_req_ms * M_S_TO_FT_MIN
    u_req_mph = u_req_ms * M_S_TO_MPH

    u_limit_ft_min = calculate_wind_limit(IR)
    u_limit_mph = u_limit_ft_min / 88.0

    phi_w_raw, C, B, E = calculate_wind_factor(u_req_ft_min, sigma, beta, beta_op)

    if inputs.wind_limit_policy == "rothermel_limit":
        is_limit_active = u_req_ft_min > u_limit_ft_min and u_req_ft_min > 0.0
        u_eff_ft_min = min(u_req_ft_min, u_limit_ft_min)
        phi_w, _, _, _ = calculate_wind_factor(u_eff_ft_min, sigma, beta, beta_op)
    else:
        is_limit_active = False
        u_eff_ft_min = u_req_ft_min
        phi_w = phi_w_raw

    u_eff_ms = u_eff_ft_min * FT_MIN_TO_M_S
    u_eff_mph = u_eff_ft_min / 88.0

    # 7. Factor de Pendiente
    phi_s = calculate_slope_factor_rothermel(inputs.slope_degrees, beta)

    # 8. Tasa Frontal Cabeza R_head (para condiciones alineadas en Fase 3.1)
    if r0_ft_min > 0.0:
        r_head_ft_min = r0_ft_min * (1.0 + phi_w + phi_s)
    else:
        r_head_ft_min = 0.0

    r_head_m_min = r_head_ft_min * FT_MIN_TO_M_MIN
    r_head_ch_hr = r_head_ft_min * FT_MIN_TO_CHAINS_HR

    # Rumbo de propagación por defecto alineado (0° hacia adelante)
    effective_heading = 0.0

    diagnostics = {
        "rho_b": rho_b,
        "beta": beta,
        "beta_opt": beta_op,
        "characteristic_sigma": sigma,
        "gamma_max": gamma_max,
        "gamma_prime": gamma_prime,
        "reaction_exponent_A": A,
        "eta_M": eta_m_weighted,
        "eta_S": eta_S,
        "I_R": IR,
        "I_R_kw_m2": IR_kw,
        "xi": xi,
        "epsilon": eps,
        "Q_ig": qig,
        "heat_sink": heat_sink,
        "C": C,
        "B": B,
        "E": E,
        "phi_w_raw": phi_w_raw,
        "phi_w": phi_w,
        "phi_s": phi_s,
        "wind_limit_threshold_mph": u_limit_mph,
        "wind_limit_active": is_limit_active,
        "effective_wind_mph": u_eff_mph,
        "r0_ft_min": r0_ft_min,
        "r_head_ft_min": r_head_ft_min,
        "r_head_chains_hr": r_head_ch_hr
    }

    return RothermelResult(
        base_ros_m_min=r0_m_min,
        head_ros_m_min=r_head_m_min,
        wind_factor=phi_w,
        slope_factor=phi_s,
        effective_heading_deg=effective_heading,
        diagnostics=diagnostics,
        r0_ft_min=r0_ft_min,
        r0_m_min=r0_m_min,
        r0_chains_hr=r0_ch_hr,
        r_head_ft_min=r_head_ft_min,
        r_head_m_min=r_head_m_min,
        r_head_chains_hr=r_head_ch_hr,
        rho_b_lb_ft3=rho_b,
        beta=beta,
        beta_opt=beta_op,
        characteristic_sigma_inv_ft=sigma,
        gamma_max_min1=gamma_max,
        reaction_exponent_A=A,
        gamma_prime_min1=gamma_prime,
        net_fuel_load_lb_ft2=wn_total,
        eta_M=eta_m_weighted,
        eta_S=eta_S,
        reaction_intensity_btu_ft2_min=IR,
        reaction_intensity_kw_m2=IR_kw,
        propagating_flux_ratio_xi=xi,
        effective_heating_number_eps=eps,
        heat_of_preignition_qig_btu_lb=qig,
        heat_sink_btu_ft3=heat_sink,
        wind_coef_C=C,
        wind_coef_B=B,
        wind_coef_E=E,
        phi_w_raw=phi_w_raw,
        phi_w=phi_w,
        phi_s=phi_s,
        input_midflame_wind_ms=u_req_ms,
        input_midflame_wind_ft_min=u_req_ft_min,
        input_midflame_wind_mph=u_req_mph,
        wind_limit_threshold_ft_min=u_limit_ft_min,
        wind_limit_threshold_mph=u_limit_mph,
        wind_limit_active=is_limit_active,
        effective_midflame_wind_ms=u_eff_ms,
        effective_midflame_wind_ft_min=u_eff_ft_min,
        effective_midflame_wind_mph=u_eff_mph
    )


class RothermelKernel(SpreadKernel):
    """
    Adaptador del kernel de Rothermel conforme a la interfaz SpreadKernel.
    """

    def calculate(self, inputs: RothermelInputs) -> RothermelResult:
        """Calcula el avance superficial puntual según Rothermel (1972)."""
        return calculate_rothermel_spread(inputs)

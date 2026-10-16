"""
spread.py

Modelo matemático modular de velocidad de propagación (ROS) y métodos de avance temporal.

ROS = ROS_base * factor_viento * factor_pendiente * factor_humedad * factor_direccion * factor_combustible
"""

import math
import heapq
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from .domain import cells_crossed_by_edge
from .fuels.models import ORIGINAL_13_FUEL_MODELS, FuelMoisture
from .directional import (
    DirectionalSpreadInputs,
    DirectionalSpreadResult,
    calculate_directional_rothermel_spread
)

# --- FUNCIONES INDEPENDIENTES DE FACTORES FISICOS ---

def calculate_wind_factor(
    wind_speed_ms: float,
    c_w: float = 0.4,
    e_w: float = 1.3,
    max_factor: float = 15.0
) -> float:
    """
    Calcula el factor de amplificación de ROS por viento.
    Unidades: wind_speed_ms en m/s.
    Retorna un valor adimensional >= 1.0, acotado por max_factor.
    """
    if wind_speed_ms <= 0:
        return 1.0
    factor = 1.0 + c_w * (wind_speed_ms ** e_w)
    return float(np.clip(factor, 1.0, max_factor))

def calculate_slope_factor(
    slope_deg: float,
    c_s: float = 5.27,
    e_s: float = 2.0,
    max_factor: float = 8.0
) -> float:
    """
    Calcula el factor de amplificación de ROS por pendiente terreno en la dirección ascendente.
    Unidades: slope_deg en grados.
    Retorna un valor adimensional >= 1.0, acotado por max_factor.
    """
    if slope_deg <= 0:
        return 1.0
    slope_rad = math.radians(slope_deg)
    tan_slope = math.tan(slope_rad)
    factor = 1.0 + c_s * (tan_slope ** e_s)
    return float(np.clip(factor, 1.0, max_factor))

def calculate_moisture_factor(
    fuel_moisture_fraction: float,
    extinction_moisture_fraction: float,
    min_factor: float = 0.001
) -> float:
    """
    Calcula el factor de reducción de ROS por humedad del combustible.
    Unidades: fracciones de humedad (0.0 a 1.0).
    Si fuel_moisture >= extinction_moisture, retorna min_factor (o 0 si se apaga).
    """
    if extinction_moisture_fraction <= 0:
        return min_factor
    if fuel_moisture_fraction >= extinction_moisture_fraction:
        return min_factor
    ratio = fuel_moisture_fraction / extinction_moisture_fraction
    factor = 1.0 - (ratio ** 2)
    return float(np.clip(factor, min_factor, 1.0))

def angular_difference_deg(angle_a: float, angle_b: float) -> float:
    """
    Calcula la mínima separación angular entre dos rumbos en grados, en el intervalo [0.0, 180.0].
    """
    return float(abs((angle_a - angle_b + 180.0) % 360.0 - 180.0))

def calculate_direction_factor(
    propagation_heading_deg: float,
    effective_push_heading_deg: float,
    eccentricity: float = 2.5
) -> float:
    """
    Calcula el factor direccional entre el sentido de avance al vecino y la dirección
    efectiva del empuje (viento + pendiente subida).
    Unidades: grados respecto al norte.
    Retorna un valor en [0.05, 1.0].
    """
    angle_diff = math.radians(angular_difference_deg(propagation_heading_deg, effective_push_heading_deg))
    cos_diff = math.cos(angle_diff)
    # Modelo elíptico simplificado: max en 0° (cos=1), min en 180° (cos=-1)
    factor = math.exp(eccentricity * (cos_diff - 1.0))
    return float(np.clip(factor, 0.05, 1.0))

# --- KERNEL GEOMÉTRICO ELÍPTICO CANÓNICO (FARSITE / ANDERSON 1983) ---

def calculate_length_to_breadth_anderson_farsite(
    effective_midflame_wind_ms: float
) -> float:
    """
    Calcula la relación largo-ancho (L/B) de la elipse de propagación según
    el modelo de Anderson (1983, INT-305) con la corrección canónica de FARSITE (Finney 1998):
    LB = 0.936 * exp(0.1147 * U_mph) + 0.461 * exp(-0.0527 * U_mph) - 0.397

    Parámetros:
        effective_midflame_wind_ms: Velocidad efectiva del viento a media llama (m/s)
                                    que sintetiza viento y pendiente para la geometría del frente.
    Retorna:
        float >= 1.0 (1.0 = círculo perfecto en ausencia de viento efectivo).
    """
    if effective_midflame_wind_ms <= 0.0:
        return 1.0
    u_mph = effective_midflame_wind_ms * 2.23693629
    lb = 0.936 * math.exp(0.1147 * u_mph) + 0.461 * math.exp(-0.0527 * u_mph) - 0.397
    return max(1.0, float(lb))

def calculate_ellipse_eccentricity(length_to_breadth: float) -> float:
    """
    Calcula la excentricidad geométrica e de la elipse a partir de la relación largo-ancho:
    e = sqrt(1 - 1 / (LB ** 2))

    Parámetros:
        length_to_breadth: Relación largo-ancho (LB >= 1.0).
    Retorna:
        float en el intervalo [0.0, 1.0) (0.0 para círculo, tendiendo a 1.0 para elipses alargadas).
    """
    if length_to_breadth <= 1.0:
        return 0.0
    return float(math.sqrt(1.0 - (1.0 / (length_to_breadth ** 2))))

def calculate_focus_radial_factor_elliptic(
    propagation_heading_deg: float,
    effective_heading_deg: float,
    eccentricity: float
) -> float:
    """
    Calcula el factor de modulación radial R(theta) / R_head medido desde el foco posterior
    de la elipse (punto de ignición) hacia el rumbo de propagación:
    R(theta) / R_head = (1 - e) / (1 - e * cos(theta - theta_eff))

    Propiedades analíticas:
        - theta = theta_eff (cabeza, 0°): ratio = 1.0
        - theta = theta_eff + 180° (cola, 180°): ratio = (1 - e) / (1 + e)
        - theta = theta_eff +- 90° (radial a 90°): ratio = 1 - e
        - e = 0 (círculo): ratio = 1.0 en todas las direcciones.

    Parámetros:
        propagation_heading_deg: Rumbo hacia la celda vecina en grados [0, 360).
        effective_heading_deg: Rumbo efectivo de empuje (eje mayor de la elipse) [0, 360).
        eccentricity: Excentricidad geométrica en [0.0, 1.0).
    Retorna:
        float en el intervalo (0.0, 1.0].
    """
    if eccentricity <= 0.0:
        return 1.0
    e = min(0.999999, max(0.0, eccentricity))
    diff_deg = angular_difference_deg(propagation_heading_deg, effective_heading_deg)
    diff_rad = math.radians(diff_deg)
    denom = 1.0 - e * math.cos(diff_rad)
    if denom <= 0.0:
        return 0.0
    return float((1.0 - e) / denom)

def calculate_cell_ros(
    base_ros_m_min: float,
    wind_factor: float,
    slope_factor: float,
    moisture_factor: float,
    direction_factor: float,
    flammability_factor: float,
    ros_reduction_barrier: float = 1.0,
    max_ros_m_min: float = 150.0
) -> float:
    """
    Calcula la velocidad de propagación superficial (ROS) en metros/minuto.
    ROS = ROS_base * f_viento * f_pendiente * f_humedad * f_direccion * f_combustible * f_barrera
    """
    if base_ros_m_min <= 0 or moisture_factor <= 0 or ros_reduction_barrier <= 0:
        return 0.0

    ros = (
        base_ros_m_min *
        wind_factor *
        slope_factor *
        moisture_factor *
        direction_factor *
        flammability_factor *
        ros_reduction_barrier
    )
    
    if np.isnan(ros) or np.isinf(ros) or ros < 0:
        return 0.0
        
    return float(np.clip(ros, 0.0, max_ros_m_min))

# --- ALGORITMO DE TIEMPO MINIMO DE LLEGADA (DIJKSTRA ANISOTRÓPICO) ---

def run_minimum_arrival_time(
    domain,
    elevation_grid: np.ndarray,
    slope_grid: np.ndarray,
    aspect_grid: np.ndarray,
    fuel_code_grid: np.ndarray,
    barrier_code_grid: np.ndarray,
    base_ros_grid: np.ndarray,
    ext_moisture_grid: np.ndarray,
    flammability_grid: np.ndarray,
    is_absolute_barrier: np.ndarray,
    ros_reduction_barrier: np.ndarray,
    ignition_row: int,
    ignition_col: int,
    weather_info: Dict[str, Any] = None,
    weather_manager: Any = None,
    fuel_moisture_provider: Any = None,
    fuel_moisture_fraction: float = 0.08,
    fuel_moisture_dead_1h: Optional[float] = None,
    fuel_moisture_dead_10h: Optional[float] = None,
    fuel_moisture_dead_100h: Optional[float] = None,
    fuel_moisture_live_herb: Optional[float] = None,
    fuel_moisture_live_woody: Optional[float] = None,
    fuel_models: Optional[Dict[int, Any]] = None,
    fuel_model_set: str = "ORIGINAL_13",
    wind_limit_policy: str = "rothermel_limit",
    apply_partial_barrier_reduction: bool = False,
    rothermel_max_ros: Optional[float] = None,
    spread_kernel: str = "experimental_qnyx",
    neighborhood_type: str = "moore",
    max_duration_minutes: float = 360.0,
    max_ros: float = 150.0,
    return_diagnostics: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Ejecuta el método de propagación por Dijkstra anisotrópico de tiempo mínimo sobre grilla ráster.
    Retorna: (arrival_time_grid, ros_grid, state_grid) por defecto.
    Si return_diagnostics=True, retorna: (arrival_time_grid, ros_grid, state_grid, diagnostics_dict).

    Parámetros de Selección de Kernel:
      - spread_kernel: "experimental_qnyx" (default para compatibilidad y regresión)
                       "rothermel" (Rothermel 1972 para ROS superficial + combinación vectorial de viento/pendiente + geometría direccional Anderson/FARSITE + propagador Dijkstra propio)
    """
    if weather_info is None and weather_manager is not None:
        weather_info = weather_manager.get_weather_at_time(0.0)
    elif weather_info is None:
        weather_info = {}

    is_dynamic_weather = bool(weather_manager is not None and hasattr(weather_manager, "is_time_dependent") and weather_manager.is_time_dependent())
    is_dynamic_fmc = bool(fuel_moisture_provider is not None and hasattr(fuel_moisture_provider, "is_time_dependent") and fuel_moisture_provider.is_time_dependent())

    # Resolución de catálogo de combustibles para evaluación física
    if fuel_models is not None:
        active_fuel_models = fuel_models
    elif fuel_model_set in ("SCOTT_BURGAN_STANDARD_FUEL", "SCOTT_BURGAN_40", "SCOTT_BURGAN"):
        from .fuels.models import SCOTT_BURGAN_STANDARD_FUEL_MODELS, load_scott_burgan_catalog
        active_fuel_models = SCOTT_BURGAN_STANDARD_FUEL_MODELS if SCOTT_BURGAN_STANDARD_FUEL_MODELS else load_scott_burgan_catalog()
    else:
        active_fuel_models = ORIGINAL_13_FUEL_MODELS

    non_burnable_codes = {int(code) for code, bed in active_fuel_models.items() if hasattr(bed, "is_burnable") and not bed.is_burnable}

    rows, cols = domain.rows, domain.cols
    arrival_time = np.full((rows, cols), np.inf, dtype=np.float32)
    ros_grid = np.zeros((rows, cols), dtype=np.float32)
    intensity_grid = np.zeros((rows, cols), dtype=np.float32)
    flame_length_grid = np.zeros((rows, cols), dtype=np.float32)
    # Estados: 0=no_fuel, 1=unburned, 2=burning, 3=burned, 4=barrier
    state_grid = np.ones((rows, cols), dtype=np.uint8)

    # Predecessor tracking para reconstrucción exacta de caminos Dijkstra
    if return_diagnostics:
        pred_r = np.full((rows, cols), -1, dtype=np.int32)
        pred_c = np.full((rows, cols), -1, dtype=np.int32)
        pred_r[ignition_row, ignition_col] = ignition_row
        pred_c[ignition_row, ignition_col] = ignition_col
    else:
        pred_r = None
        pred_c = None

    # Contadores y flags de auditoría de terminación
    horizon_rejections = 0
    physical_zero_ros_rejections = 0
    barrier_rejections = 0
    last_processed_time = 0.0
    next_pending_arrival_time = None
    terminated_by_horizon = False
    active_candidates_beyond_horizon = 0

    # Estadísticas cuantitativas de evaluaciones Rothermel
    ros_evaluations_count = 0
    ros_cap_hit_count = 0
    all_ros_evaluations: List[float] = []

    # Aplicar barreras absolutas y sin combustible / no combustibles
    state_grid[fuel_code_grid == 0] = 0
    if non_burnable_codes:
        state_grid[np.isin(fuel_code_grid, list(non_burnable_codes))] = 0
    state_grid[is_absolute_barrier] = 4

    # Validar ignición
    if not domain.is_inside(ignition_row, ignition_col):
        raise ValueError(f"Punto de ignición ({ignition_row}, {ignition_col}) fuera del dominio.")
    if state_grid[ignition_row, ignition_col] in (0, 4):
        raise ValueError(f"Punto de ignición ({ignition_row}, {ignition_col}) está en celda no combustible o barrera absoluta.")

    # Inicializar ignición
    arrival_time[ignition_row, ignition_col] = 0.0
    state_grid[ignition_row, ignition_col] = 2  # burning

    # Priority Queue: (arrival_time, row, col)
    pq = []
    heapq.heappush(pq, (0.0, ignition_row, ignition_col))
    visited = np.zeros((rows, cols), dtype=bool)

    # =========================================================================
    # RAMA A: KERNEL FÍSICO CANÓNICO ROTHERMEL (1972) + RESULTANTE + FARSITE
    # =========================================================================
    if spread_kernel == "rothermel":
        # 1. Extracción Estricta de Viento a Altura Media de Llama (Midflame)
        if "midflame_wind_ms" not in weather_info and not is_dynamic_weather:
            raise ValueError(
                "Rothermel requires explicit midflame_wind_ms. "
                "Reference/10m wind must be converted externally using a documented Wind Adjustment Factor."
            )
        midflame_wind_ms = float(weather_info.get("midflame_wind_ms", 0.0))

        if midflame_wind_ms > 0.0 and not is_dynamic_weather:
            if "wind_from_direction_deg" not in weather_info and "wind_direction_deg" not in weather_info:
                raise ValueError("Rothermel requires explicit wind_from_direction_deg when midflame_wind_ms > 0.")
            wind_from_deg = float(weather_info.get("wind_from_direction_deg", weather_info.get("wind_direction_deg", 0.0)))
        else:
            wind_from_deg = float(weather_info.get("wind_from_direction_deg", weather_info.get("wind_direction_deg", 0.0)))

        # 2. Configuración Estricta de Humedades Muertas y Vivas (sin defaults silenciosos)
        if fuel_moisture_provider is not None:
            fmc_t0 = fuel_moisture_provider.get_state(0.0, ignition_row, ignition_col)
            d1_val = float(fmc_t0.dead_1h)
            d10_val = float(fmc_t0.dead_10h)
            d100_val = float(fmc_t0.dead_100h)
            lh_val = float(fmc_t0.live_herbaceous)
            lw_val = float(fmc_t0.live_woody)
        else:
            if (
                fuel_moisture_dead_1h is None or
                fuel_moisture_dead_10h is None or
                fuel_moisture_dead_100h is None or
                fuel_moisture_live_herb is None or
                fuel_moisture_live_woody is None
            ):
                raise ValueError(
                    "Rothermel requires explicit moisture vector: "
                    "fuel_moisture_dead_1h, fuel_moisture_dead_10h, fuel_moisture_dead_100h, "
                    "fuel_moisture_live_herb, and fuel_moisture_live_woody."
                )
            d1_val = float(fuel_moisture_dead_1h)
            d10_val = float(fuel_moisture_dead_10h)
            d100_val = float(fuel_moisture_dead_100h)
            lh_val = float(fuel_moisture_live_herb)
            lw_val = float(fuel_moisture_live_woody)

        fuel_moist = FuelMoisture(
            dead_1h=d1_val,
            dead_10h=d10_val,
            dead_100h=d100_val,
            live_herbaceous=lh_val,
            live_woody=lw_val
        )

        # 3. Validación / Advertencia de Stencil para Rothermel
        if neighborhood_type.lower() not in ("stencil_16", "16_neighbors", "stencil_24", "24_neighbors"):
            warnings.warn(
                f"Rothermel directional raster spread produces metric anisotropy distortion on 8-neighbor stencil ('{neighborhood_type}'). "
                "'stencil_16' is recommended for high fidelity.",
                UserWarning
            )

        # Cache de celdas por (fuel_code, cell_slope, cell_aspect, ...) sin cuantización
        directional_cache: Dict[Tuple, DirectionalSpreadResult] = {}

        while pq:
            curr_t, r, c = heapq.heappop(pq)

            if curr_t > max_duration_minutes:
                terminated_by_horizon = True
                next_pending_arrival_time = float(curr_t)
                active_candidates_beyond_horizon = 1 + len(pq)
                break

            last_processed_time = float(curr_t)

            if visited[r, c]:
                continue

            fuel_code = int(fuel_code_grid[r, c])
            if fuel_code == 0 or fuel_code in non_burnable_codes:
                visited[r, c] = True
                continue
            if fuel_code not in active_fuel_models:
                raise ValueError(
                    f"Invalid fuel code {fuel_code} at cell ({r}, {c}) for Rothermel kernel. "
                    f"Valid codes in {fuel_model_set}: {sorted(list(active_fuel_models.keys()))}."
                )
            bed = active_fuel_models[fuel_code]
            if not bed.is_burnable:
                visited[r, c] = True
                continue

            visited[r, c] = True
            state_grid[r, c] = 3  # burned

            cell_slope = float(slope_grid[r, c])
            cell_aspect = float(aspect_grid[r, c])
            cell_barrier_ros = float(ros_reduction_barrier[r, c])

            # Evaluación direccional de la celda fuente (source cell semantics)
            if is_dynamic_weather:
                w_curr = weather_manager.get_weather_at_time(curr_t)
                curr_midflame = float(w_curr.get("midflame_wind_ms", midflame_wind_ms))
                curr_wind_from = float(w_curr.get("wind_direction_deg", w_curr.get("wind_from_direction_deg", wind_from_deg)))
            else:
                curr_midflame = midflame_wind_ms
                curr_wind_from = wind_from_deg

            if is_dynamic_fmc:
                fmc_curr = fuel_moisture_provider.get_state(curr_t, r, c)
                fuel_moist_curr = FuelMoisture(
                    dead_1h=float(fmc_curr.dead_1h),
                    dead_10h=float(fmc_curr.dead_10h),
                    dead_100h=float(fmc_curr.dead_100h),
                    live_herbaceous=float(fmc_curr.live_herbaceous),
                    live_woody=float(fmc_curr.live_woody)
                )
            else:
                fuel_moist_curr = fuel_moist

            if is_dynamic_weather or is_dynamic_fmc:
                cache_key = (
                    fuel_code, cell_slope, cell_aspect, curr_midflame, curr_wind_from,
                    fuel_moist_curr.dead_1h, fuel_moist_curr.dead_10h, fuel_moist_curr.dead_100h,
                    fuel_moist_curr.live_herbaceous, fuel_moist_curr.live_woody
                )
            else:
                cache_key = (fuel_code, cell_slope, cell_aspect)

            if cache_key not in directional_cache:
                dir_inputs = DirectionalSpreadInputs(
                    fuel_bed=bed,
                    moisture=fuel_moist_curr,
                    midflame_wind_ms=curr_midflame,
                    wind_from_direction_deg=curr_wind_from,
                    slope_degrees=cell_slope,
                    aspect_downslope_degrees=cell_aspect,
                    wind_limit_policy=wind_limit_policy
                )
                dir_result = calculate_directional_rothermel_spread(dir_inputs)
                directional_cache[cache_key] = dir_result
            else:
                dir_result = directional_cache[cache_key]

            # Inicializar intensidad y longitud de llama para el píxel de ignición
            if r == ignition_row and c == ignition_col:
                ign_ros = float(dir_result.r0_m_min)
                ign_sigma = float(dir_result.characteristic_sigma_inv_ft)
                ign_tr = (384.0 / ign_sigma) if ign_sigma > 0.0 else 0.0
                ign_ib = float(dir_result.reaction_intensity_kw_m2 * ign_tr * ign_ros)
                ign_fl = float(0.0775 * (ign_ib ** 0.46)) if ign_ib > 0.0 else 0.0
                ros_grid[r, c] = ign_ros
                intensity_grid[r, c] = ign_ib
                flame_length_grid[r, c] = ign_fl

            for nr, nc, dist_m, heading_deg in domain.get_neighbors(r, c, neighborhood_type):
                if visited[nr, nc]:
                    continue
                neighbor_fuel = int(fuel_code_grid[nr, nc])
                if (
                    state_grid[nr, nc] in (0, 4)
                    or neighbor_fuel == 0
                    or neighbor_fuel in non_burnable_codes
                    or is_absolute_barrier[nr, nc]
                ):
                    barrier_rejections += 1
                    continue

                # Traversabilidad de aristas largas (supercover): verificar celdas intermedias
                if max(abs(nr - r), abs(nc - c)) > 1:
                    blocked = False
                    for ir, ic in cells_crossed_by_edge(r, c, nr, nc):
                        mid_fuel = int(fuel_code_grid[ir, ic])
                        if (
                            state_grid[ir, ic] in (0, 4)
                            or mid_fuel == 0
                            or mid_fuel in non_burnable_codes
                            or is_absolute_barrier[ir, ic]
                        ):
                            blocked = True
                            break
                    if blocked:
                        barrier_rejections += 1
                        continue

                # ROS direccional del arco (en m/min)
                ros = float(dir_result.ros_at_heading(heading_deg))

                # Garantía de invariantes físicos Rothermel
                if not (math.isfinite(ros) and ros >= 0.0):
                    raise ValueError(f"Non-finite or negative ROS ({ros}) at ({r},{c}) -> ({nr},{nc})")
                if not (math.isfinite(dir_result.reaction_intensity_kw_m2) and dir_result.reaction_intensity_kw_m2 >= 0.0):
                    raise ValueError(f"Non-finite or negative reaction intensity ({dir_result.reaction_intensity_kw_m2})")

                ros_evaluations_count += 1
                all_ros_evaluations.append(ros)

                # Modificador empírico opcional de barreras parciales (OFF por defecto para perfil físico puro)
                if apply_partial_barrier_reduction:
                    ros *= cell_barrier_ros

                # Cap numérico opcional (None por defecto para perfil de referencia)
                if rothermel_max_ros is not None:
                    if ros >= rothermel_max_ros:
                        ros_cap_hit_count += 1
                        ros = rothermel_max_ros

                # Extinción física: ROS exactamente nulo/no propagable según Rothermel
                if ros <= 0.0:
                    physical_zero_ros_rejections += 1
                    continue

                travel_time = dist_m / ros
                new_t = curr_t + travel_time

                if new_t > max_duration_minutes:
                    horizon_rejections += 1

                if new_t < arrival_time[nr, nc]:
                    arrival_time[nr, nc] = new_t
                    ros_grid[nr, nc] = ros

                    # Cálculo Byram de intensidad de línea de fuego y longitud de llama
                    cell_sigma = float(dir_result.characteristic_sigma_inv_ft)
                    t_res_min = (384.0 / cell_sigma) if cell_sigma > 0.0 else 0.0
                    byram_ib_kw_m = float(dir_result.reaction_intensity_kw_m2 * t_res_min * ros)
                    flame_len_m = float(0.0775 * (byram_ib_kw_m ** 0.46)) if byram_ib_kw_m > 0.0 else 0.0
                    intensity_grid[nr, nc] = byram_ib_kw_m
                    flame_length_grid[nr, nc] = flame_len_m

                    state_grid[nr, nc] = 2  # burning active
                    heapq.heappush(pq, (new_t, nr, nc))
                    if pred_r is not None:
                        pred_r[nr, nc] = r
                        pred_c[nr, nc] = c

        queue_emptied = not terminated_by_horizon and len(pq) == 0
        if not terminated_by_horizon:
            active_candidates_beyond_horizon = 0

        boundary_mask = np.zeros((rows, cols), dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        boundary_reached = bool(np.any(visited & boundary_mask))

        combustible_mask = (fuel_code_grid > 0) & (~is_absolute_barrier)
        if non_burnable_codes:
            combustible_mask &= (~np.isin(fuel_code_grid, list(non_burnable_codes)))
        domain_exhausted = bool(np.all(visited[combustible_mask]))

        if terminated_by_horizon:
            termination_reason = "TIME_HORIZON_REACHED"
        elif domain_exhausted:
            termination_reason = "DOMAIN_EXHAUSTED"
        elif queue_emptied:
            if physical_zero_ros_rejections > 0:
                termination_reason = "PHYSICAL_EXTINCTION"
            else:
                termination_reason = "OTHER_QUEUE_EMPTY"
        else:
            termination_reason = "OTHER_QUEUE_EMPTY"

        if return_diagnostics:
            diag = {
                "termination_reason": termination_reason,
                "queue_emptied": queue_emptied,
                "boundary_reached": boundary_reached,
                "domain_exhausted": domain_exhausted,
                "horizon_rejections": horizon_rejections,
                "physical_zero_ros_rejections": physical_zero_ros_rejections,
                "barrier_rejections": barrier_rejections,
                "active_candidates_beyond_horizon": active_candidates_beyond_horizon,
                "last_processed_time": last_processed_time,
                "next_pending_arrival_time": next_pending_arrival_time,
                "predecessor_row": pred_r,
                "predecessor_col": pred_c,
                "intensity_grid": intensity_grid,
                "flame_length_grid": flame_length_grid,
                "number_ros_evaluations": ros_evaluations_count,
                "ros_cap_hit_count": ros_cap_hit_count,
                "min_ros": float(np.min(all_ros_evaluations)) if all_ros_evaluations else 0.0,
                "mean_ros": float(np.mean(all_ros_evaluations)) if all_ros_evaluations else 0.0,
                "p95_ros": float(np.percentile(all_ros_evaluations, 95)) if all_ros_evaluations else 0.0,
                "max_ros": float(np.max(all_ros_evaluations)) if all_ros_evaluations else 0.0,
            }
            return arrival_time, ros_grid, state_grid, diag

        return arrival_time, ros_grid, state_grid

    # =========================================================================
    # RAMA B: KERNEL EXPERIMENTAL Q-NYX (DEFAULT PRESERVADO)
    # =========================================================================
    wind_speed = weather_info.get("wind_speed_ms", 4.0)
    wind_spread_heading = weather_info.get("spread_heading_deg", 45.0)
    f_wind = calculate_wind_factor(wind_speed)

    if fuel_moisture_provider is not None:
        fmc_t0 = fuel_moisture_provider.get_state(0.0, ignition_row, ignition_col)
        fuel_moist_frac_base = float(fmc_t0.dead_1h)
    else:
        fuel_moist_frac_base = fuel_moisture_fraction

    while pq:
        curr_t, r, c = heapq.heappop(pq)

        if curr_t > max_duration_minutes:
            terminated_by_horizon = True
            next_pending_arrival_time = float(curr_t)
            active_candidates_beyond_horizon = 1 + len(pq)
            break

        last_processed_time = float(curr_t)

        if visited[r, c]:
            continue
        visited[r, c] = True
        state_grid[r, c] = 3  # burned

        cell_slope = slope_grid[r, c]
        cell_aspect = aspect_grid[r, c]  # aspect: orientación de la pendiente (dirección bajada)
        cell_base_ros = base_ros_grid[r, c]
        cell_ext_moist = ext_moisture_grid[r, c]
        cell_flam = flammability_grid[r, c]
        cell_barrier_ros = ros_reduction_barrier[r, c]

        if cell_base_ros <= 0:
            continue

        if is_dynamic_weather:
            w_curr = weather_manager.get_weather_at_time(curr_t)
            curr_wind_speed = w_curr.get("wind_speed_ms", 4.0)
            curr_wind_spread_heading = w_curr.get("spread_heading_deg", 45.0)
            curr_f_wind = calculate_wind_factor(curr_wind_speed)
        else:
            curr_wind_speed = wind_speed
            curr_wind_spread_heading = wind_spread_heading
            curr_f_wind = f_wind

        if is_dynamic_fmc:
            fmc_curr = fuel_moisture_provider.get_state(curr_t, r, c)
            curr_fuel_moist_frac = float(fmc_curr.dead_1h)
        else:
            curr_fuel_moist_frac = fuel_moist_frac_base

        f_slope_upslope = calculate_slope_factor(cell_slope)
        f_moist = calculate_moisture_factor(curr_fuel_moist_frac, cell_ext_moist)

        # Dirección efectiva de empuje (composición vectorial de incrementos netos de viento y pendiente)
        # Pendiente arriba es (aspect + 180) % 360
        upslope_heading = (cell_aspect + 180.0) % 360.0

        delta_wind = max(0.0, curr_f_wind - 1.0) if curr_wind_speed > 0.0 else 0.0
        delta_slope = max(0.0, f_slope_upslope - 1.0) if cell_slope > 0.0 else 0.0

        w_rad = math.radians(curr_wind_spread_heading)
        s_rad = math.radians(upslope_heading)

        vx = math.sin(w_rad) * delta_wind + math.sin(s_rad) * delta_slope
        vy = math.cos(w_rad) * delta_wind + math.cos(s_rad) * delta_slope

        push_mag = math.hypot(vx, vy)

        if push_mag > 1e-6:
            eff_heading = math.degrees(math.atan2(vx, vy)) % 360.0
            u_eff_geom = (push_mag / 0.4) ** (1.0 / 1.3)
            cell_lb = calculate_length_to_breadth_anderson_farsite(u_eff_geom)
            cell_ecc = calculate_ellipse_eccentricity(cell_lb)
        else:
            eff_heading = 0.0
            cell_ecc = 0.0

        for nr, nc, dist_m, heading_deg in domain.get_neighbors(r, c, neighborhood_type):
            if visited[nr, nc]:
                continue
            if state_grid[nr, nc] in (0, 4) or fuel_code_grid[nr, nc] == 0 or is_absolute_barrier[nr, nc]:
                barrier_rejections += 1
                continue

            # Traversabilidad de aristas largas (supercover): verificar que celdas intermedias no sean barrera/agua
            if max(abs(nr - r), abs(nc - c)) > 1:
                blocked = False
                for ir, ic in cells_crossed_by_edge(r, c, nr, nc):
                    if state_grid[ir, ic] in (0, 4) or fuel_code_grid[ir, ic] == 0 or is_absolute_barrier[ir, ic]:
                        blocked = True
                        break
                if blocked:
                    barrier_rejections += 1
                    continue

            f_dir = calculate_focus_radial_factor_elliptic(heading_deg, eff_heading, cell_ecc)

            ros = calculate_cell_ros(
                base_ros_m_min=cell_base_ros,
                wind_factor=f_wind,
                slope_factor=f_slope_upslope if angular_difference_deg(heading_deg, upslope_heading) < 90.0 else 1.0,
                moisture_factor=f_moist,
                direction_factor=f_dir,
                flammability_factor=cell_flam,
                ros_reduction_barrier=cell_barrier_ros,
                max_ros_m_min=max_ros
            )

            if ros <= 0.0:
                physical_zero_ros_rejections += 1
                continue

            travel_time = dist_m / ros
            new_t = curr_t + travel_time

            if new_t > max_duration_minutes:
                horizon_rejections += 1

            if new_t < arrival_time[nr, nc]:
                arrival_time[nr, nc] = new_t
                ros_grid[nr, nc] = ros
                state_grid[nr, nc] = 2  # burning active
                heapq.heappush(pq, (new_t, nr, nc))
                if pred_r is not None:
                    pred_r[nr, nc] = r
                    pred_c[nr, nc] = c

    queue_emptied = not terminated_by_horizon and len(pq) == 0
    if not terminated_by_horizon:
        active_candidates_beyond_horizon = 0

    boundary_mask = np.zeros((rows, cols), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_reached = bool(np.any(visited & boundary_mask))

    combustible_mask = (fuel_code_grid > 0) & (~is_absolute_barrier)
    domain_exhausted = bool(np.all(visited[combustible_mask]))

    if terminated_by_horizon:
        termination_reason = "TIME_HORIZON_REACHED"
    elif domain_exhausted:
        termination_reason = "DOMAIN_EXHAUSTED"
    elif queue_emptied:
        if physical_zero_ros_rejections > 0:
            termination_reason = "PHYSICAL_EXTINCTION"
        else:
            termination_reason = "OTHER_QUEUE_EMPTY"
    else:
        termination_reason = "OTHER_QUEUE_EMPTY"

    if return_diagnostics:
        diag = {
            "termination_reason": termination_reason,
            "queue_emptied": queue_emptied,
            "boundary_reached": boundary_reached,
            "domain_exhausted": domain_exhausted,
            "horizon_rejections": horizon_rejections,
            "physical_zero_ros_rejections": physical_zero_ros_rejections,
            "barrier_rejections": barrier_rejections,
            "active_candidates_beyond_horizon": active_candidates_beyond_horizon,
            "last_processed_time": last_processed_time,
            "next_pending_arrival_time": next_pending_arrival_time,
            "predecessor_row": pred_r,
            "predecessor_col": pred_c,
        }
        return arrival_time, ros_grid, state_grid, diag

    return arrival_time, ros_grid, state_grid

# --- METODO A: PROPAGACION TEMPORAL ACUMULATIVA ---

def run_cumulative_propagation(
    domain,
    elevation_grid: np.ndarray,
    slope_grid: np.ndarray,
    aspect_grid: np.ndarray,
    fuel_code_grid: np.ndarray,
    barrier_code_grid: np.ndarray,
    base_ros_grid: np.ndarray,
    ext_moisture_grid: np.ndarray,
    flammability_grid: np.ndarray,
    is_absolute_barrier: np.ndarray,
    ros_reduction_barrier: np.ndarray,
    ignition_row: int,
    ignition_col: int,
    weather_info: Dict[str, Any] = None,
    weather_manager: Any = None,
    dt_minutes: float = 1.0,
    fuel_moisture_fraction: float = 0.08,
    fuel_moisture_provider: Any = None,
    neighborhood_type: str = "moore",
    max_duration_minutes: float = 360.0,
    max_ros: float = 150.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ejecuta el método A de propagación por Acumulación de Exposición Temporal.
    """
    if weather_info is None and weather_manager is not None:
        weather_info = weather_manager.get_weather_at_time(0.0)
    elif weather_info is None:
        weather_info = {}

    is_dynamic_weather = bool(weather_manager is not None and hasattr(weather_manager, "is_time_dependent") and weather_manager.is_time_dependent())
    is_dynamic_fmc = bool(fuel_moisture_provider is not None and hasattr(fuel_moisture_provider, "is_time_dependent") and fuel_moisture_provider.is_time_dependent())

    rows, cols = domain.rows, domain.cols
    arrival_time = np.full((rows, cols), np.inf, dtype=np.float32)
    exposure_progress = np.zeros((rows, cols), dtype=np.float32)
    ros_grid = np.zeros((rows, cols), dtype=np.float32)
    state_grid = np.ones((rows, cols), dtype=np.uint8)

    state_grid[fuel_code_grid == 0] = 0
    state_grid[is_absolute_barrier] = 4

    arrival_time[ignition_row, ignition_col] = 0.0
    state_grid[ignition_row, ignition_col] = 2  # burning

    active_front = {(ignition_row, ignition_col)}
    curr_time = 0.0

    wind_speed = weather_info.get("wind_speed_ms", 4.0)
    wind_spread_heading = weather_info.get("spread_heading_deg", 45.0)
    f_wind = calculate_wind_factor(wind_speed)

    if fuel_moisture_provider is not None:
        fmc_t0 = fuel_moisture_provider.get_state(0.0, ignition_row, ignition_col)
        fuel_moist_frac_base = float(fmc_t0.dead_1h)
    else:
        fuel_moist_frac_base = fuel_moisture_fraction

    while curr_time < max_duration_minutes and active_front:
        curr_time += dt_minutes
        new_ignitions = set()
        completed_cells = set()

        if is_dynamic_weather:
            w_curr = weather_manager.get_weather_at_time(curr_time)
            curr_wind_speed = w_curr.get("wind_speed_ms", 4.0)
            curr_wind_spread_heading = w_curr.get("spread_heading_deg", 45.0)
            curr_f_wind = calculate_wind_factor(curr_wind_speed)
        else:
            curr_wind_speed = wind_speed
            curr_wind_spread_heading = wind_spread_heading
            curr_f_wind = f_wind

        for r, c in list(active_front):
            cell_slope = slope_grid[r, c]
            cell_aspect = aspect_grid[r, c]
            cell_base_ros = base_ros_grid[r, c]
            cell_ext_moist = ext_moisture_grid[r, c]
            cell_flam = flammability_grid[r, c]
            cell_barrier_ros = ros_reduction_barrier[r, c]

            if cell_base_ros <= 0:
                completed_cells.add((r, c))
                continue

            if is_dynamic_fmc:
                fmc_curr = fuel_moisture_provider.get_state(curr_time, r, c)
                curr_fuel_moist_frac = float(fmc_curr.dead_1h)
            else:
                curr_fuel_moist_frac = fuel_moist_frac_base

            f_slope_upslope = calculate_slope_factor(cell_slope)
            f_moist = calculate_moisture_factor(curr_fuel_moist_frac, cell_ext_moist)
            upslope_heading = (cell_aspect + 180.0) % 360.0

            delta_wind = max(0.0, curr_f_wind - 1.0) if curr_wind_speed > 0.0 else 0.0
            delta_slope = max(0.0, f_slope_upslope - 1.0) if cell_slope > 0.0 else 0.0

            w_rad = math.radians(curr_wind_spread_heading)
            s_rad = math.radians(upslope_heading)

            vx = math.sin(w_rad) * delta_wind + math.sin(s_rad) * delta_slope
            vy = math.cos(w_rad) * delta_wind + math.cos(s_rad) * delta_slope

            push_mag = math.hypot(vx, vy)

            if push_mag > 1e-6:
                eff_heading = math.degrees(math.atan2(vx, vy)) % 360.0
                u_eff_geom = (push_mag / 0.4) ** (1.0 / 1.3)
                cell_lb = calculate_length_to_breadth_anderson_farsite(u_eff_geom)
                cell_ecc = calculate_ellipse_eccentricity(cell_lb)
            else:
                eff_heading = 0.0
                cell_ecc = 0.0

            for nr, nc, dist_m, heading_deg in domain.get_neighbors(r, c, neighborhood_type):
                if state_grid[nr, nc] in (0, 3, 4):
                    continue

                # Traversabilidad de aristas largas (supercover): verificar que celdas intermedias no sean barrera/agua
                if max(abs(nr - r), abs(nc - c)) > 1:
                    blocked = False
                    for ir, ic in cells_crossed_by_edge(r, c, nr, nc):
                        if state_grid[ir, ic] in (0, 4) or fuel_code_grid[ir, ic] == 0 or is_absolute_barrier[ir, ic]:
                            blocked = True
                            break
                    if blocked:
                        continue

                f_dir = calculate_focus_radial_factor_elliptic(heading_deg, eff_heading, cell_ecc)

                ros = calculate_cell_ros(
                    base_ros_m_min=cell_base_ros,
                    wind_factor=f_wind,
                    slope_factor=f_slope_upslope if angular_difference_deg(heading_deg, upslope_heading) < 90.0 else 1.0,
                    moisture_factor=f_moist,
                    direction_factor=f_dir,
                    flammability_factor=cell_flam,
                    ros_reduction_barrier=cell_barrier_ros,
                    max_ros_m_min=max_ros
                )

                if ros <= 0.001:
                    continue

                ros_grid[nr, nc] = max(ros_grid[nr, nc], ros)
                dist_covered = ros * dt_minutes
                exposure_progress[nr, nc] += (dist_covered / dist_m)

                if exposure_progress[nr, nc] >= 1.0 and state_grid[nr, nc] == 1:
                    state_grid[nr, nc] = 2  # burning
                    arrival_time[nr, nc] = curr_time
                    new_ignitions.add((nr, nc))

            completed_cells.add((r, c))

        for r, c in completed_cells:
            state_grid[r, c] = 3  # burned
            active_front.remove((r, c))

        active_front.update(new_ignitions)

    return arrival_time, ros_grid, state_grid

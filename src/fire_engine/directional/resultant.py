"""
src/fire_engine/directional/resultant.py

Cálculo de la resultante vectorial viento + pendiente y velocidad de viento equivalente/efectiva
según la formulación canónica de Albini (1976 INT-30) / Rothermel (1983 INT-GTR-143) / Finney (1998 RMRS-GTR-4) / Andrews (2018 RMRS-GTR-371).
"""

import math
from dataclasses import dataclass
from typing import Tuple
from ..units import (
    MPH_TO_M_S,
    M_S_TO_MPH,
    M_S_TO_FT_MIN,
    FT_MIN_TO_M_S,
    MPH_TO_FT_MIN,
    FT_MIN_TO_MPH,
    FT_MIN_TO_CHAINS_HR,
    FT_MIN_TO_M_MIN
)
from ..kernels.rothermel import (
    calculate_wind_coefficients,
    calculate_wind_factor,
    calculate_slope_factor_rothermel,
    calculate_wind_limit
)


def wind_from_to_spread_heading(wind_from_deg: float) -> float:
    """
    Convierte la dirección meteorológica de procedencia del viento (FROM)
    al rumbo de avance hacia el que empuja el fuego (TOWARDS / Downwind).
    
    Convención:
      0° = Norte, 90° = Este, 180° = Sur, 270° = Oeste (horario).
      - Viento del Oeste (270°) -> Fuego avanza hacia el Este (90°).
      - Viento del Norte (0°)   -> Fuego avanza hacia el Sur (180°).
    """
    return (wind_from_deg + 180.0) % 360.0


def aspect_to_upslope_heading(aspect_downslope_deg: float) -> float:
    """
    Convierte la orientación topográfica de pendiente descendente (aspect SIG convencional)
    al rumbo ascendente de máxima pendiente hacia el que avanza el fuego (Upslope heading).
    
    Convención:
      - Ladera con exposición Norte (aspect=0°, pendiente baja hacia el Norte)
        -> La subida (upslope) se orienta hacia el Sur (180°).
      - Ladera con exposición Oeste (aspect=270°, pendiente baja hacia el Oeste)
        -> La subida (upslope) se orienta hacia el Este (90°).
    """
    return (aspect_downslope_deg + 180.0) % 360.0


def calculate_slope_and_aspect_from_dem(
    elevation_grid: "np.ndarray",
    cell_size_m: float = 30.0
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Calcula la matriz de pendiente (en grados) y aspect (dirección descendente en grados azimutales)
    a partir de una matriz DEM utilizando diferencias finitas centradas (Horn 1981 / GDAL).
    
    Convención de Grilla:
      - Filas: Fila 0 es el Norte, Fila N-1 es el Sur.
      - Columnas: Columna 0 es el Oeste, Columna M-1 es el Este.
      - Aspect: Azimut en [0°, 360°) hacia donde cae el terreno (downslope).
    """
    import numpy as np
    
    # dz_dy (Norte-Sur, filas) y dz_dx (Este-Oeste, columnas)
    dz_dy, dz_dx = np.gradient(elevation_grid, cell_size_m)
    
    # En la grilla, 'y' aumenta hacia el Norte (filas decrecientes), por lo que dy = -d(fila)
    dz_dnorth = -dz_dy
    dz_deast = dz_dx

    # Pendiente en grados
    slope_rad = np.arctan(np.hypot(dz_deast, dz_dnorth))
    slope_deg = np.degrees(slope_rad)

    # Aspect: Vector descendente = (-dz_deast, -dz_dnorth)
    aspect_deg = np.degrees(np.arctan2(-dz_deast, -dz_dnorth)) % 360.0
    return slope_deg, aspect_deg


def calculate_inverse_effective_wind(
    phi_net: float,
    sigma_inv_ft: float,
    beta: float,
    beta_opt: float
) -> float:
    """
    Despeja analíticamente la velocidad de viento equivalente U_eff [ft/min] a partir
    del factor adimensional neto de propagación phi_net, invirtiendo la ecuación canónica:
        phi_w = C * U^B * (beta / beta_opt)^(-E)
        U_eff = [ phi_net / (C * (beta / beta_opt)^(-E)) ]^(1 / B)
    
    Fuente: Rothermel (1972 INT-115, Eq. 47) / Rothermel (1983 INT-GTR-143, p. 12) / Andrews (2018, Eq. 32).
    """
    if phi_net <= 0.0 or sigma_inv_ft <= 0.0 or beta <= 0.0 or beta_opt <= 0.0:
        return 0.0

    C, B, E = calculate_wind_coefficients(sigma_inv_ft)
    if C <= 0.0 or B <= 0.0:
        return 0.0

    ratio_term = (beta / beta_opt) ** (-E)
    denom = C * ratio_term
    if denom <= 0.0:
        return 0.0

    base = phi_net / denom
    if base <= 0.0:
        return 0.0

    return base ** (1.0 / B)


@dataclass(frozen=True)
class VectorResultant:
    """
    Representa el resultado de la combinación vectorial física viento + pendiente.
    """
    phi_w: float
    phi_s: float
    wind_spread_heading_deg: float
    upslope_heading_deg: float
    phi_x: float
    phi_y: float
    phi_resultant: float
    heading_resultant_deg: float
    effective_wind_ft_min: float
    effective_wind_m_s: float
    effective_wind_mph: float
    wind_limit_active: bool
    r_head_ft_min: float
    r_head_m_min: float
    r_head_chains_hr: float


def combine_wind_and_slope_vectors(
    r0_ft_min: float,
    reaction_intensity_btu_ft2_min: float,
    phi_w: float,
    wind_spread_heading_deg: float,
    phi_s: float,
    upslope_heading_deg: float,
    sigma_inv_ft: float,
    beta: float,
    beta_opt: float,
    wind_limit_policy: str = "rothermel_limit"
) -> VectorResultant:
    """
    Sintetiza vectorialmente los factores adimensionales de viento y pendiente en un único
    vector neto Phi_net = Phi_w + Phi_s, calcula el rumbo de avance máximo y la velocidad efectiva.
    
    Fuentes:
      - Albini (1976 INT-30, pp. 18-20)
      - Rothermel (1983 INT-GTR-143, pp. 11-13)
      - Finney (1998 RMRS-GTR-4, pp. 4-5)
      - Andrews (2018 RMRS-GTR-371, pp. 30-31)
    """
    # 1. Componentes Cartesianas Navigacionales (x = Este, y = Norte)
    rad_w = math.radians(wind_spread_heading_deg % 360.0)
    rad_s = math.radians(upslope_heading_deg % 360.0)

    phi_w_x = phi_w * math.sin(rad_w)
    phi_w_y = phi_w * math.cos(rad_w)

    phi_s_x = phi_s * math.sin(rad_s)
    phi_s_y = phi_s * math.cos(rad_s)

    phi_x = phi_w_x + phi_s_x
    phi_y = phi_w_y + phi_s_y

    phi_resultant = math.sqrt(phi_x ** 2 + phi_y ** 2)

    # 2. Rumbo de Avance Máximo (Heading)
    if phi_resultant > 1e-9:
        heading_deg = math.degrees(math.atan2(phi_x, phi_y)) % 360.0
    else:
        # Cancelación exacta o ausencia de viento/pendiente -> Comportamiento Isótropo
        phi_resultant = 0.0
        heading_deg = 0.0

    # 3. Velocidad de Viento Efectiva / Equivalente
    u_eff_ft_min = calculate_inverse_effective_wind(phi_resultant, sigma_inv_ft, beta, beta_opt)
    
    # 4. Política de Límite de Viento (Rothermel 1972 / Andrews 2018 / Finney 1998)
    wind_limit_active = False
    if wind_limit_policy == "rothermel_limit":
        u_max_ft_min = calculate_wind_limit(reaction_intensity_btu_ft2_min)
        if u_max_ft_min > 0.0 and u_eff_ft_min > u_max_ft_min:
            u_eff_ft_min = u_max_ft_min
            wind_limit_active = True
            # Recalcular phi_resultant correspondiente al viento límite efectivo
            phi_resultant, _, _, _ = calculate_wind_factor(u_eff_ft_min, sigma_inv_ft, beta, beta_opt)

    u_eff_m_s = u_eff_ft_min * FT_MIN_TO_M_S
    u_eff_mph = u_eff_ft_min * FT_MIN_TO_MPH

    # 5. Tasa Máxima Resultante R_head
    r_head_ft_min = r0_ft_min * (1.0 + phi_resultant)
    r_head_m_min = r_head_ft_min * FT_MIN_TO_M_MIN
    r_head_chains_hr = r_head_ft_min * FT_MIN_TO_CHAINS_HR

    return VectorResultant(
        phi_w=phi_w,
        phi_s=phi_s,
        wind_spread_heading_deg=wind_spread_heading_deg % 360.0,
        upslope_heading_deg=upslope_heading_deg % 360.0,
        phi_x=phi_x,
        phi_y=phi_y,
        phi_resultant=phi_resultant,
        heading_resultant_deg=heading_deg,
        effective_wind_ft_min=u_eff_ft_min,
        effective_wind_m_s=u_eff_m_s,
        effective_wind_mph=u_eff_mph,
        wind_limit_active=wind_limit_active,
        r_head_ft_min=r_head_ft_min,
        r_head_m_min=r_head_m_min,
        r_head_chains_hr=r_head_chains_hr
    )

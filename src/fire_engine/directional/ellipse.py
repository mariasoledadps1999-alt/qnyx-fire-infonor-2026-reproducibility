"""
src/fire_engine/directional/ellipse.py

Geometría elíptica canónica de propagación desde el foco posterior (Anderson 1983 / FARSITE / Finney 1998).
"""

import math


def calculate_length_to_breadth_anderson_farsite(wind_speed_m_s: float) -> float:
    """
    Calcula la relación largo/ancho (L/B) de la elipse de fuego en función del viento efectivo en m/s.
    Fórmula canónica: L/B = 1.0 + 0.125 * U_mph = 1.0 + 0.2796 * U_ms.
    Fuentes: Anderson (1983 INT-GTR-143) / Finney (1998 RMRS-GTR-4, Eq. 5) / Andrews (2018, Eq. 52).
    """
    if wind_speed_m_s <= 0.0:
        return 1.0
    return 1.0 + 0.279604 * wind_speed_m_s


def calculate_ellipse_eccentricity(length_to_breadth: float) -> float:
    """
    Calcula la excentricidad geométrica de la elipse a partir de L/B:
        e = sqrt( (L/B)^2 - 1 ) / (L/B)
    """
    lb = max(1.0, float(length_to_breadth))
    if lb <= 1.0:
        return 0.0
    return math.sqrt(lb ** 2 - 1.0) / lb


def calculate_focus_radial_factor_elliptic(angle_difference_deg: float, eccentricity: float) -> float:
    """
    Calcula el factor de reducción radial focal f_dir(theta) desde el foco posterior de ignición:
        f_dir(theta) = (1 - e) / (1 - e * cos(theta))
    
    Donde:
      - theta = 0°   (Cabeza / Head)    -> f_dir = 1.0
      - theta = 90°  (Flanco / Flank)   -> f_dir = 1 - e
      - theta = 180° (Cola / Backing)   -> f_dir = (1 - e) / (1 + e)
    
    Fuente: Finney (1998 RMRS-GTR-4, Eq. 8).
    """
    e = max(0.0, min(0.999999, float(eccentricity)))
    if e <= 0.0:
        return 1.0
    
    rad = math.radians(angle_difference_deg)
    cos_theta = math.cos(rad)
    denom = 1.0 - e * cos_theta
    if denom <= 1e-9:
        return 1.0
    return (1.0 - e) / denom


def calculate_directional_ros(
    r_head: float,
    heading_resultant_deg: float,
    query_angle_deg: float,
    eccentricity: float
) -> float:
    """
    Evalúa la tasa de propagación R(theta) hacia un rumbo arbitrario query_angle_deg:
        delta_theta = |query_angle_deg - heading_resultant_deg|
        R(theta) = R_head * f_dir(delta_theta)
    """
    if r_head <= 0.0:
        return 0.0
    
    # Diferencia angular circular mínima en [0°, 180°]
    diff = abs((query_angle_deg - heading_resultant_deg + 180.0) % 360.0 - 180.0)
    factor = calculate_focus_radial_factor_elliptic(diff, eccentricity)
    return r_head * factor

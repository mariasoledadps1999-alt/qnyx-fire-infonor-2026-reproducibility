"""
src/fire_engine/directional/spread.py

Integración física y direccional completa:
Rothermel (1972 INT-115) + Resultante Vectorial (Albini 1976 / Finney 1998) + Geometría Elíptica (Anderson 1983 / FARSITE).
"""

from dataclasses import dataclass
from typing import Optional
from ..units import (
    MPH_TO_M_S,
    M_S_TO_MPH,
    M_S_TO_FT_MIN,
    FT_MIN_TO_M_MIN,
    FT_MIN_TO_CHAINS_HR,
    BTU_FT2_MIN_TO_KW_M2
)
from ..fuels.models import RothermelFuelBed, FuelMoisture
from ..kernels.rothermel import (
    RothermelKernel,
    RothermelInputs,
    calculate_slope_factor_rothermel,
    calculate_wind_factor
)
from .resultant import (
    wind_from_to_spread_heading,
    aspect_to_upslope_heading,
    combine_wind_and_slope_vectors,
    VectorResultant
)
from .ellipse import (
    calculate_length_to_breadth_anderson_farsite,
    calculate_ellipse_eccentricity,
    calculate_directional_ros,
    calculate_focus_radial_factor_elliptic
)


@dataclass(frozen=True)
class DirectionalSpreadInputs:
    """
    Entradas ambientales y topográficas completas para la propagación direccional.
    """
    fuel_bed: RothermelFuelBed
    moisture: FuelMoisture
    midflame_wind_ms: float = 0.0               # Velocidad del viento a la altura media de la llama [m/s]
    wind_from_direction_deg: float = 0.0        # Dirección DESDE la cual sopla el viento [0°=N, 90°=E]
    slope_degrees: float = 0.0                  # Inclinación del terreno [grados]
    aspect_downslope_degrees: float = 0.0       # Orientación descendente de la ladera (aspect SIG) [0°=N, 90°=E]
    wind_limit_policy: str = "rothermel_limit"  # "rothermel_limit" o "none"


@dataclass(frozen=True)
class DirectionalSpreadResult:
    """
    Resultado integral de propagación física y direccional de Rothermel.
    """
    r0_ft_min: float
    r0_m_min: float
    r0_chains_hr: float
    
    r_head_ft_min: float
    r_head_m_min: float
    r_head_chains_hr: float

    phi_w: float
    phi_s: float
    phi_resultant: float

    wind_from_deg: float
    wind_spread_heading_deg: float
    aspect_downslope_deg: float
    upslope_heading_deg: float
    heading_resultant_deg: float

    input_midflame_wind_ms: float
    input_midflame_wind_mph: float
    effective_wind_ms: float
    effective_wind_mph: float
    effective_wind_ft_min: float

    wind_limit_active: bool
    wind_limit_policy: str

    length_to_breadth: float
    eccentricity: float

    reaction_intensity_btu_ft2_min: float
    reaction_intensity_kw_m2: float
    characteristic_sigma_inv_ft: float
    beta: float
    beta_opt: float

    def ros_at_heading(self, query_angle_deg: float) -> float:
        """
        Calcula la velocidad de propagación R(theta) hacia un rumbo arbitrario en metros/minuto [m/min].
        """
        return calculate_directional_ros(
            self.r_head_m_min,
            self.heading_resultant_deg,
            query_angle_deg,
            self.eccentricity
        )

    def ros_chains_hr_at_heading(self, query_angle_deg: float) -> float:
        """
        Calcula la velocidad de propagación R(theta) hacia un rumbo arbitrario en chains/hora [ch/h].
        """
        return calculate_directional_ros(
            self.r_head_chains_hr,
            self.heading_resultant_deg,
            query_angle_deg,
            self.eccentricity
        )

    def ros_ft_min_at_heading(self, query_angle_deg: float) -> float:
        """
        Calcula la velocidad de propagación R(theta) hacia un rumbo arbitrario en pies/minuto [ft/min].
        """
        return calculate_directional_ros(
            self.r_head_ft_min,
            self.heading_resultant_deg,
            query_angle_deg,
            self.eccentricity
        )


def calculate_directional_rothermel_spread(inputs: DirectionalSpreadInputs) -> DirectionalSpreadResult:
    """
    Ejecuta el pipeline completo de propagación direccional:
      1. Evaluación del lecho físico mediante RothermelKernel (R0, I_R, sigma, beta, beta_opt, phi_w base).
      2. Cálculo del factor de pendiente phi_s (Rothermel 1972 Eq. 51).
      3. Conversión inequívoca de rumbos: wind_spread_heading y upslope_heading.
      4. Combinación vectorial Phi_net = Phi_w + Phi_s (Albini 1976 / Rothermel 1983 / Finney 1998).
      5. Inversión analítica hacia velocidad de viento efectiva U_eff y aplicación de política de wind limit.
      6. Geometría elíptica Anderson/FARSITE: L/B, excentricidad e y distribución direccional desde el foco posterior.
    """
    bed = inputs.fuel_bed
    moist = inputs.moisture

    # 1. Evaluación física base de Rothermel
    kernel = RothermelKernel()
    base_res = kernel.calculate(RothermelInputs(
        fuel_bed=bed,
        moisture=moist,
        midflame_wind_ms=inputs.midflame_wind_ms,
        slope_degrees=0.0,
        wind_limit_policy="none"  # El límite se aplicará sobre la resultante vectorial combinada
    ))

    # 2. Factor de pendiente físico de Rothermel
    phi_s = calculate_slope_factor_rothermel(inputs.slope_degrees, base_res.beta)

    # 3. Conversión de rumbos hacia direcciones de avance del fuego (TOWARDS)
    wind_spread_heading = wind_from_to_spread_heading(inputs.wind_from_direction_deg)
    upslope_heading = aspect_to_upslope_heading(inputs.aspect_downslope_degrees)

    # 4. Combinación vectorial de viento y pendiente
    vec_res = combine_wind_and_slope_vectors(
        r0_ft_min=base_res.r0_ft_min,
        reaction_intensity_btu_ft2_min=base_res.reaction_intensity_btu_ft2_min,
        phi_w=base_res.phi_w,
        wind_spread_heading_deg=wind_spread_heading,
        phi_s=phi_s,
        upslope_heading_deg=upslope_heading,
        sigma_inv_ft=base_res.characteristic_sigma_inv_ft,
        beta=base_res.beta,
        beta_opt=base_res.beta_opt,
        wind_limit_policy=inputs.wind_limit_policy
    )

    # 5. Geometría Elíptica Anderson / FARSITE
    lb = calculate_length_to_breadth_anderson_farsite(vec_res.effective_wind_m_s)
    eccentricity = calculate_ellipse_eccentricity(lb)

    return DirectionalSpreadResult(
        r0_ft_min=base_res.r0_ft_min,
        r0_m_min=base_res.r0_m_min,
        r0_chains_hr=base_res.r0_chains_hr,
        r_head_ft_min=vec_res.r_head_ft_min,
        r_head_m_min=vec_res.r_head_m_min,
        r_head_chains_hr=vec_res.r_head_chains_hr,
        phi_w=vec_res.phi_w,
        phi_s=vec_res.phi_s,
        phi_resultant=vec_res.phi_resultant,
        wind_from_deg=inputs.wind_from_direction_deg % 360.0,
        wind_spread_heading_deg=vec_res.wind_spread_heading_deg,
        aspect_downslope_deg=inputs.aspect_downslope_degrees % 360.0,
        upslope_heading_deg=vec_res.upslope_heading_deg,
        heading_resultant_deg=vec_res.heading_resultant_deg,
        input_midflame_wind_ms=inputs.midflame_wind_ms,
        input_midflame_wind_mph=inputs.midflame_wind_ms * M_S_TO_MPH,
        effective_wind_ms=vec_res.effective_wind_m_s,
        effective_wind_mph=vec_res.effective_wind_mph,
        effective_wind_ft_min=vec_res.effective_wind_ft_min,
        wind_limit_active=vec_res.wind_limit_active,
        wind_limit_policy=inputs.wind_limit_policy,
        length_to_breadth=lb,
        eccentricity=eccentricity,
        reaction_intensity_btu_ft2_min=base_res.reaction_intensity_btu_ft2_min,
        reaction_intensity_kw_m2=base_res.reaction_intensity_kw_m2,
        characteristic_sigma_inv_ft=base_res.characteristic_sigma_inv_ft,
        beta=base_res.beta,
        beta_opt=base_res.beta_opt
    )

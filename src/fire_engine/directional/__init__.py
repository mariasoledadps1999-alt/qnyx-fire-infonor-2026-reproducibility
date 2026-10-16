"""
src/fire_engine/directional/__init__.py
"""

from .resultant import (
    wind_from_to_spread_heading,
    aspect_to_upslope_heading,
    calculate_slope_and_aspect_from_dem,
    calculate_inverse_effective_wind,
    combine_wind_and_slope_vectors,
    VectorResultant
)
from .ellipse import (
    calculate_length_to_breadth_anderson_farsite,
    calculate_ellipse_eccentricity,
    calculate_focus_radial_factor_elliptic,
    calculate_directional_ros
)
from .spread import (
    DirectionalSpreadInputs,
    DirectionalSpreadResult,
    calculate_directional_rothermel_spread
)

__all__ = [
    "wind_from_to_spread_heading",
    "aspect_to_upslope_heading",
    "calculate_slope_and_aspect_from_dem",
    "calculate_inverse_effective_wind",
    "combine_wind_and_slope_vectors",
    "VectorResultant",
    "calculate_length_to_breadth_anderson_farsite",
    "calculate_ellipse_eccentricity",
    "calculate_focus_radial_factor_elliptic",
    "calculate_directional_ros",
    "DirectionalSpreadInputs",
    "DirectionalSpreadResult",
    "calculate_directional_rothermel_spread"
]

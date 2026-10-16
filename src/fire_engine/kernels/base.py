"""
src/fire_engine/kernels/base.py

Interfaz base abstracta para kernels físicos de propagación de incendios en Q-NYX Fire.
Permite desacoplar completamente la física puntual del propagador espacial (Dijkstra / Ráster).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class SpreadResult:
    """
    Resultado conceptual común producido por cualquier kernel de propagación física.
    """
    base_ros_m_min: float           # Tasa base de avance sin viento ni pendiente R0 [m/min]
    head_ros_m_min: float           # Tasa máxima de avance en la cabeza R_head [m/min]
    wind_factor: float              # Factor adimensional de viento (phi_w)
    slope_factor: float             # Factor adimensional de pendiente (phi_s)
    effective_heading_deg: float    # Dirección neta de avance del frente [grados]
    diagnostics: Dict[str, Any] = field(default_factory=dict) # Variables intermedias diagnósticas


class SpreadKernel(ABC):
    """
    Clase base abstracta para todos los kernels de cálculo físico de avance.
    """

    @abstractmethod
    def calculate(self, *args, **kwargs) -> SpreadResult:
        """Calcula las tasas de avance y variables diagnósticas a partir de las entradas físicas."""
        pass

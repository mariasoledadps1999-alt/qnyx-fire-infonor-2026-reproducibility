"""
barriers.py

Manejo de barreras territoriales (agua, caminos, zonas urbanas, suelo desnudo).
Ajuste de velocidad de propagación, permeabilidad y cruce probabilístico.
"""

from typing import Dict, Any, Tuple
import numpy as np
from fire_engine.config import get_barrier_parameters

class BarrierManager:
    def __init__(self, custom_params: Dict[str, Any] = None, seed: int = 42):
        if custom_params:
            self.catalog = custom_params
        else:
            self.catalog = get_barrier_parameters()
            
        self.barrier_categories = self.catalog.get("barrier_categories", {})
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_barrier_info(self, barrier_code: int) -> Dict[str, Any]:
        code_str = str(int(barrier_code))
        if code_str not in self.barrier_categories:
            # Por defecto: terreno normal si no se reconoce
            return {
                "name": "desconocido",
                "is_absolute_barrier": False,
                "permeability": 1.0,
                "ros_reduction_factor": 1.0,
                "crossing_probability": 1.0,
                "max_jump_distance_m": 30.0
            }
        return self.barrier_categories[code_str]

    def create_barrier_matrices(self, barrier_raster: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera matrices de (is_absolute_barrier_mask, ros_reduction_grid, permeability_grid).
        """
        rows, cols = barrier_raster.shape
        is_absolute = np.zeros((rows, cols), dtype=bool)
        ros_factor = np.ones((rows, cols), dtype=np.float32)
        permeability = np.ones((rows, cols), dtype=np.float32)

        unique_codes = np.unique(barrier_raster)
        for code in unique_codes:
            if np.isnan(code):
                continue
            info = self.get_barrier_info(int(code))
            mask = (barrier_raster == code)
            is_absolute[mask] = info["is_absolute_barrier"]
            ros_factor[mask] = info["ros_reduction_factor"]
            permeability[mask] = info["permeability"]

        return is_absolute, ros_factor, permeability

    def can_cross_barrier(self, barrier_code: int) -> bool:
        """Determina de forma probabilística si el fuego puede cruzar una celda de barrera."""
        info = self.get_barrier_info(barrier_code)
        if info["is_absolute_barrier"]:
            return False
        prob = info.get("crossing_probability", 1.0)
        if prob >= 1.0:
            return True
        if prob <= 0.0:
            return False
        return self.rng.random() <= prob

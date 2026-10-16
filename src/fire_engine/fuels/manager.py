"""
src/fire_engine/fuels/manager.py

Mapeo de modelos de combustible, parámetros físicos y cálculo del factor de combustible.
Mantiene compatibilidad con el pipeline ráster productivo de Q-NYX Fire.
"""

from typing import Dict, Any, Tuple
import numpy as np
from ..config import get_fuel_parameters


class FuelManager:
    def __init__(self, custom_params: Dict[str, Any] = None):
        if custom_params:
            self.catalog = custom_params
        else:
            self.catalog = get_fuel_parameters()
            
        self.fuel_models = self.catalog.get("fuel_models", {})

    def get_cell_properties(self, fuel_code: int) -> Dict[str, Any]:
        """Obtiene las propiedades físicas asociadas al código de combustible."""
        code_str = str(int(fuel_code))
        if code_str not in self.fuel_models:
            raise ValueError(
                f"Código de combustible no reconocido en catálogo: {fuel_code}. "
                f"Códigos válidos: {list(self.fuel_models.keys())}"
            )
        return self.fuel_models[code_str]

    def create_fuel_layers(self, fuel_raster: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera matrices de parámetros físicos a partir de la grilla raster de modelos de combustible.
        Retorna (base_ros_grid, extinction_moisture_grid, flammability_grid, burn_duration_grid).
        """
        rows, cols = fuel_raster.shape
        base_ros = np.zeros((rows, cols), dtype=np.float32)
        ext_moisture = np.zeros((rows, cols), dtype=np.float32)
        flammability = np.zeros((rows, cols), dtype=np.float32)
        burn_duration = np.zeros((rows, cols), dtype=np.float32)

        unique_codes = np.unique(fuel_raster)
        for code in unique_codes:
            if np.isnan(code):
                continue
            code_int = int(code)
            props = self.get_cell_properties(code_int)
            mask = (fuel_raster == code)
            base_ros[mask] = props["base_ros_m_min"]
            ext_moisture[mask] = props["extinction_moisture"]
            flammability[mask] = props["flammability_factor"]
            burn_duration[mask] = props["burn_duration_min"]

        return base_ros, ext_moisture, flammability, burn_duration

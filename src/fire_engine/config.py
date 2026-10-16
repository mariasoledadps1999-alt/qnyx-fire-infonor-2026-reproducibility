"""
config.py

Gestión y carga de la configuración global del motor de simulación.
"""

import json
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / "configs"

class EngineConfig(BaseModel):
    engine_name: str = "Maule Experimental Wildfire Engine"
    version: str = "1.0.0"
    neighborhood_type: str = "moore"  # 'moore' (8) or 'von_neumann' (4)
    propagation_method: str = "min_arrival_time"  # 'min_arrival_time' or 'cumulative'
    cell_size_m: float = 30.0
    default_max_ros_m_min: float = 150.0
    min_ros_m_min: float = 0.0
    wind_factor_max: float = 15.0
    slope_factor_max: float = 8.0
    moisture_factor_min: float = 0.01
    random_seed: int = 42
    disclaimer: str = (
        "Prototipo experimental de simulación. No utilizar para coordinación "
        "de emergencias, evacuaciones ni decisiones operacionales sin validación "
        "científica e institucional."
    )

def load_json_config(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_engine_config() -> EngineConfig:
    cfg_file = CONFIG_DIR / "engine.json"
    if cfg_file.exists():
        data = load_json_config(cfg_file)
        return EngineConfig(**data)
    return EngineConfig()

def get_fuel_parameters() -> Dict[str, Any]:
    cfg_file = CONFIG_DIR / "fuel_parameters.json"
    return load_json_config(cfg_file)

def get_barrier_parameters() -> Dict[str, Any]:
    cfg_file = CONFIG_DIR / "barriers.json"
    return load_json_config(cfg_file)

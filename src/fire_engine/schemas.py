"""
schemas.py

Esquemas Pydantic para la API, escenarios, ignición, meteorología y resultados.
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, model_validator

VALID_NELSON_RUNTIME_MODES: Set[str] = {
    "hourly_anchored_exact_sampled_5min",
    "nfdrs4_sequential_5min",
    "exact_sampled",
    "exact_sampled_5min",
    "dense_timeseries",
    "adaptive_exact_sampled",
}

class IgnitionPoint(BaseModel):
    lon: Optional[float] = Field(None, description="Longitud WGS84")
    lat: Optional[float] = Field(None, description="Latitud WGS84")
    row: Optional[int] = Field(None, description="Fila en la grilla raster (0-indexed)")
    col: Optional[int] = Field(None, description="Columna en la grilla raster (0-indexed)")
    file: Optional[str] = Field(None, description="Ruta relativa a GeoJSON de ignición")
    status: str = Field("synthetic", description="Estado de los datos de ignición")

class WeatherSeries(BaseModel):
    file: Optional[str] = None
    status: str = "synthetic"
    wind_speed_ms: float = Field(4.0, ge=0.0, le=100.0, description="Velocidad del viento en m/s")
    midflame_wind_ms: Optional[float] = Field(None, ge=0.0, le=100.0, description="Velocidad del viento a altura media de llama en m/s (requerido para Rothermel)")
    wind_direction_deg: float = Field(225.0, ge=0.0, lt=360.0, description="Dirección meteorológica del viento en grados (desde donde viene)")
    temperature_c: float = Field(24.0, description="Temperatura en °C")
    relative_humidity_pct: float = Field(38.0, ge=0.0, le=100.0, description="Humedad relativa en %")
    precipitation_mm: float = Field(0.0, ge=0.0, description="Precipitación en mm")
    solar_radiation_w_m2: Optional[float] = Field(None, ge=0.0, description="Flujo medio de radiación solar descendente en W/m²")
    temporal_mode: str = Field("constant", description="Modo temporal: 'constant' o 'hourly_stepwise'")
    start_datetime_utc: Optional[str] = Field(None, description="Datetime ISO en UTC para indexar la serie horaria")
    simulation_start_datetime_utc: Optional[str] = Field(None, description="Datetime ISO en UTC específico del inicio de simulación t=0")


class InitialFuelMoisture(BaseModel):
    dead_1h_fraction: float = Field(0.07, ge=0.0, le=1.0)
    dead_10h_fraction: float = Field(0.09, ge=0.0, le=1.0)
    dead_100h_fraction: float = Field(0.12, ge=0.0, le=1.0)
    live_herbaceous_fraction: float = Field(0.60, ge=0.0, le=5.0)
    live_woody_fraction: Optional[float] = Field(None, ge=0.0, le=5.0, description="Fracción de humedad de combustible leñoso vivo")
    status: str = "experimental"

class FuelMoistureConfig(BaseModel):
    mode: str = Field("constant", description="'constant', 'hourly_dynamic' o 'external'")
    physics_model: str = Field("simple_timelag", description="'simple_timelag' o 'nelson'")
    runtime_mode: Optional[str] = Field(None, description="Modo de lookup subhorario para Nelson ('hourly_anchored_exact_sampled_5min', 'nfdrs4_sequential_5min', etc.)")
    initialization_mode: str = Field("explicit", description="'explicit' o 'conditioned'")
    seed_mode: str = Field("equilibrium", description="'equilibrium' o 'explicit' para modo conditioned")
    conditioning_hours: Optional[int] = Field(None, description="Horas de acondicionamiento meteorológico previo")
    conditioning_start_datetime_utc: Optional[str] = Field(None, description="Timestamp inicial UTC del período de conditioning")
    dead_1h: Optional[float] = Field(None, ge=0.0, le=1.0, description="Fracción de humedad 1-h")
    dead_10h: Optional[float] = Field(None, ge=0.0, le=1.0, description="Fracción de humedad 10-h")
    dead_100h: Optional[float] = Field(None, ge=0.0, le=1.0, description="Fracción de humedad 100-h")
    live_herbaceous: Optional[float] = Field(None, ge=0.0, le=5.0, description="Fracción de humedad herbácea viva")
    live_woody: Optional[float] = Field(None, ge=0.0, le=5.0, description="Fracción de humedad leñosa viva")
    dead_1h_fraction: Optional[float] = Field(None, ge=0.0, le=1.0)
    dead_10h_fraction: Optional[float] = Field(None, ge=0.0, le=1.0)
    dead_100h_fraction: Optional[float] = Field(None, ge=0.0, le=1.0)
    live_herbaceous_fraction: Optional[float] = Field(None, ge=0.0, le=5.0)
    live_woody_fraction: Optional[float] = Field(None, ge=0.0, le=5.0)

    @model_validator(mode="after")
    def validate_nelson_configuration(self) -> "FuelMoistureConfig":
        p_model = (self.physics_model or "").lower().strip()
        if p_model == "nelson":
            if self.mode != "hourly_dynamic":
                raise ValueError(
                    f"physics_model='nelson' requiere mode='hourly_dynamic', pero se recibió mode='{self.mode}'."
                )
            if self.runtime_mode is not None:
                rm = self.runtime_mode.lower().strip()
                if rm not in VALID_NELSON_RUNTIME_MODES:
                    raise ValueError(
                        f"Unknown runtime_mode '{self.runtime_mode}' para Nelson. "
                        f"Modos válidos permitidos: {sorted(list(VALID_NELSON_RUNTIME_MODES))}"
                    )
        return self

class PropagationConfig(BaseModel):
    neighborhood_type: str = Field("stencil_16", description="'stencil_16', 'moore', etc.")
    wind_limit_policy: str = Field("rothermel_limit", description="'rothermel_limit' o 'none'")
    apply_partial_barrier_reduction: bool = Field(False, description="Modificador empírico de barreras parciales")
    rothermel_max_ros: Optional[float] = Field(None, description="Cap numérico externo de ROS")

class ScenarioConfig(BaseModel):
    id: str = "scenario_001"
    status: str = "experimental"
    spread_model: str = Field("experimental_qnyx", description="'experimental_qnyx' o 'rothermel'")
    fuel_model_set: str = Field("experimental_maule", description="'experimental_maule' u 'ORIGINAL_13'")
    pilot_area: Optional[str] = "boundaries/pilot_area.geojson"
    duration_minutes: int = Field(360, gt=0, description="Duración total de la simulación en minutos")
    output_interval_minutes: int = Field(15, gt=0, description="Intervalo entre salidas en minutos")
    ignition: IgnitionPoint
    weather: WeatherSeries
    initial_fuel_moisture: Optional[InitialFuelMoisture] = None
    fuel_moisture: Optional[FuelMoistureConfig] = None
    propagation: Optional[PropagationConfig] = Field(default_factory=PropagationConfig)
    rasters: Dict[str, str] = Field(default_factory=dict)
    warning: str = "No apto para decisiones operacionales; parámetros sin calibración local."


class SimulationRequest(BaseModel):
    scenario_id: Optional[str] = "scenario_001"
    scenario: Optional[ScenarioConfig] = None
    ignition: Optional[IgnitionPoint] = None
    duration_minutes: Optional[int] = Field(None, gt=0)
    output_interval_minutes: Optional[int] = Field(None, gt=0)
    weather: Optional[WeatherSeries] = None
    random_seed: Optional[int] = 42
    spread_model: Optional[str] = Field(None, description="'experimental_qnyx' o 'rothermel'")
    fuel_model_set: Optional[str] = Field(None, description="'experimental_maule' u 'ORIGINAL_13'")
    propagation_method: Optional[str] = Field(None, description="'min_arrival_time' o 'cumulative'")
    neighborhood_type: Optional[str] = Field(None, description="'moore', 'stencil_16', etc.")
    propagation: Optional[PropagationConfig] = None
    authorized_params: Optional[Dict[str, Any]] = None

class TimelineEntry(BaseModel):
    elapsed_minutes: float
    timestamp_iso: str
    burned_area_hectares: float
    active_cells_count: int
    cumulative_burned_cells: int
    spatial_extent_utm: List[float]
    perimeter_ref: str

class SimulationSummary(BaseModel):
    simulation_id: str
    status: str  # queued, running, completed, failed, cancelled
    created_at: str
    scenario_id: str
    duration_minutes: int
    burned_area_ha: float
    execution_time_seconds: float
    output_dir: str
    disclaimer: str

class WebExportMetadata(BaseModel):
    landscape_id: str
    crs: str
    extent_epsg32719: List[float]
    extent_epsg4326: List[float]
    resolution_m: float
    rows: int
    columns: int
    simulation_id: str
    duration_minutes: int
    output_interval_minutes: int
    available_timesteps: List[float]
    total_burned_area_ha: float
    disclaimer: str
    rasters: Dict[str, str]

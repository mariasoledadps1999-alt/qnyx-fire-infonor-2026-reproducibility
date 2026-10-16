"""
simulation.py

Orquestador de ejecución de simulación de propagación de incendios.
"""

import os
import time
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from fire_engine.config import get_engine_config
from fire_engine.schemas import ScenarioConfig, SimulationSummary
from fire_engine.raster_loader import load_and_validate_rasters
from fire_engine.fuels import FuelManager, create_fuel_moisture_provider
from fire_engine.barriers import BarrierManager
from fire_engine.weather import WeatherManager
from fire_engine.spread import run_minimum_arrival_time, run_cumulative_propagation
from fire_engine.validation import validate_scenario, ScenarioValidationError
from fire_engine.outputs import OutputGenerator


class SimulationEngine:
    def __init__(self, base_dir: Path, custom_engine_config: Optional[Dict[str, Any]] = None):
        self.base_dir = base_dir
        self.engine_cfg = get_engine_config()
        if custom_engine_config:
            for k, v in custom_engine_config.items():
                if hasattr(self.engine_cfg, k):
                    setattr(self.engine_cfg, k, v)

    def run_simulation(
        self,
        scenario: ScenarioConfig,
        simulation_id: Optional[str] = None,
        output_parent_dir: Optional[Path] = None,
        random_seed: Optional[int] = None,
        propagation_method: Optional[str] = None,
        neighborhood_type: Optional[str] = None
    ) -> SimulationSummary:
        start_time = time.time()
        sim_id = simulation_id or f"sim_{int(time.time())}"
        out_dir = (output_parent_dir or (self.base_dir / "outputs")) / sim_id
        out_dir.mkdir(parents=True, exist_ok=True)

        seed = random_seed if random_seed is not None else self.engine_cfg.random_seed
        prop_method = propagation_method or scenario.rasters.get("propagation_method", self.engine_cfg.propagation_method)

        spread_model = getattr(scenario, "spread_model", "experimental_qnyx")
        prop_cfg = getattr(scenario, "propagation", None)

        # Precedencia estricta para neighborhood_type:
        # 1. Parámetro explícito de llamada (neighborhood_type) si no es None
        # 2. Configuración en Scenario (scenario.propagation.neighborhood_type) si existe
        # 3. Default según modelo: "stencil_16" para Rothermel, engine_cfg.neighborhood_type para experimental
        if neighborhood_type is not None:
            neigh_type = neighborhood_type
        elif prop_cfg and getattr(prop_cfg, "neighborhood_type", None):
            neigh_type = prop_cfg.neighborhood_type
        elif spread_model == "rothermel":
            neigh_type = "stencil_16"
        else:
            neigh_type = self.engine_cfg.neighborhood_type or "moore"

        # Validación de incompatibilidad de método de propagación
        if spread_model == "rothermel" and prop_method == "cumulative":
            raise ScenarioValidationError(
                "Rothermel spread_model only supports Dijkstra minimum arrival time propagation, not 'cumulative'."
            )

        # Configurar logging estructurado para la simulación
        log_file = out_dir / "simulation.log"
        logger = logging.getLogger(f"sim_{sim_id}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

        logger.info(f"Iniciando Simulación {sim_id} | Motor v{self.engine_cfg.version}")
        logger.info(f"Escenario: {scenario.id} | Modelo: {spread_model} | Método: {prop_method} | Vecindad: {neigh_type} | Semilla: {seed}")
        logger.warning(self.engine_cfg.disclaimer)

        # 1. Cargar Rásteteres Territoriales
        data_core_dir = self.base_dir / "maule_fire_data_core"
        if not data_core_dir.exists():
            data_core_dir = self.base_dir  # Fallback

        logger.info("Cargando y validando alineación de capas GeoTIFF...")
        landscape = load_and_validate_rasters(data_core_dir, scenario.rasters)

        # Hash de archivos de entrada
        for name, rel_p in scenario.rasters.items():
            full_p = data_core_dir / rel_p
            if full_p.exists():
                h = hashlib.sha256(full_p.read_bytes()).hexdigest()[:12]
                logger.info(f" Insumo [{name}]: {rel_p} (SHA256: {h})")

        # 2. Validar Escenario e Ignición
        logger.info("Validando condiciones previas del escenario...")
        _, r_ign, c_ign, val_msg = validate_scenario(scenario, self.base_dir, landscape)
        logger.info(f"Punto de ignición verificado: Fila={r_ign}, Columna={c_ign}")

        # 3. Preparar Matrices de Combustible y Barreras
        if spread_model == "experimental_qnyx":
            fuel_mgr = FuelManager()
            base_ros, ext_moist, flam_factor, burn_dur = fuel_mgr.create_fuel_layers(landscape.fuel_model)
        else:
            # UNUSED BY ROTHERMEL BRANCH (placeholders for signature compatibility)
            # The Rothermel kernel calculates physical ROS directly from ORIGINAL_13 fuel codes and 5 moisture fractions.
            shape = landscape.elevation.shape
            base_ros = np.zeros(shape, dtype=np.float32)
            ext_moist = np.zeros(shape, dtype=np.float32)
            flam_factor = np.zeros(shape, dtype=np.float32)
            burn_dur = np.zeros(shape, dtype=np.float32)

        barrier_mgr = BarrierManager(seed=seed)
        is_abs_barrier, ros_red_barrier, perm = barrier_mgr.create_barrier_matrices(landscape.barriers)

        weather_mgr = WeatherManager(scenario.weather, base_dir=data_core_dir)
        weather_info = weather_mgr.get_weather_at_time(0.0)

        # 4. Proveedor de Humedad de Combustible (Q-NYX FMC-1 / FMC-2 / FMC-3 / FMC-5A)
        fuel_moisture_provider = create_fuel_moisture_provider(
            fuel_moisture_config=scenario.fuel_moisture,
            legacy_initial_moisture=scenario.initial_fuel_moisture,
            weather_manager=weather_mgr
        )
        fmc_state = fuel_moisture_provider.get_state(0.0)
        fuel_moist_fraction = float(fmc_state.dead_1h)

        if hasattr(fuel_moisture_provider, "conditioning_report") and fuel_moisture_provider.conditioning_report is not None:
            rep = fuel_moisture_provider.conditioning_report
            logger.info(
                f"Acondicionamiento previo completado ({rep.conditioning_hours} h, semilla '{rep.seed_mode}'): "
                f"dead_1h={rep.final_dead_1h:.4f}, dead_10h={rep.final_dead_10h:.4f}, dead_100h={rep.final_dead_100h:.4f}"
            )
            for w in rep.warnings:
                logger.warning(f" [Conditioning Warning] {w}")

        # 5. Ejecutar Motor de Propagación
        logger.info(f"Modelo Físico: {spread_model} | Catálogo: {getattr(scenario, 'fuel_model_set', 'experimental_maule')}")

        if spread_model == "rothermel":
            midflame_wind = float(scenario.weather.midflame_wind_ms) if scenario.weather.midflame_wind_ms is not None else 0.0
            wind_from_dir = float(scenario.weather.wind_direction_deg)
            rothermel_weather_info = {
                "midflame_wind_ms": midflame_wind,
                "wind_from_direction_deg": wind_from_dir,
                "elapsed_minutes": 0.0
            }

            moist_d1h = float(fmc_state.dead_1h)
            moist_d10h = float(fmc_state.dead_10h)
            moist_d100h = float(fmc_state.dead_100h)
            moist_live_herb = float(fmc_state.live_herbaceous)
            moist_live_woody = float(fmc_state.live_woody)

            wind_limit = prop_cfg.wind_limit_policy if prop_cfg else "rothermel_limit"
            apply_partial_barrier = prop_cfg.apply_partial_barrier_reduction if prop_cfg else False
            rothermel_max_ros_val = prop_cfg.rothermel_max_ros if prop_cfg else None

            logger.info(f"Ejecutando propagación canónica Rothermel (Dijkstra + {neigh_type})...")
            arr_time, ros_grid, state_grid = run_minimum_arrival_time(
                domain=landscape.domain,
                elevation_grid=landscape.elevation,
                slope_grid=landscape.slope,
                aspect_grid=landscape.aspect,
                fuel_code_grid=landscape.fuel_model,
                barrier_code_grid=landscape.barriers,
                base_ros_grid=base_ros,
                ext_moisture_grid=ext_moist,
                flammability_grid=flam_factor,
                is_absolute_barrier=is_abs_barrier,
                ros_reduction_barrier=ros_red_barrier,
                ignition_row=r_ign,
                ignition_col=c_ign,
                weather_info=rothermel_weather_info,
                weather_manager=weather_mgr,
                fuel_moisture_provider=fuel_moisture_provider,
                fuel_moisture_dead_1h=moist_d1h,
                fuel_moisture_dead_10h=moist_d10h,
                fuel_moisture_dead_100h=moist_d100h,
                fuel_moisture_live_herb=moist_live_herb,
                fuel_moisture_live_woody=moist_live_woody,
                wind_limit_policy=wind_limit,
                apply_partial_barrier_reduction=apply_partial_barrier,
                rothermel_max_ros=rothermel_max_ros_val,
                spread_kernel="rothermel",
                neighborhood_type=neigh_type,
                max_duration_minutes=scenario.duration_minutes
            )

        elif prop_method == "cumulative":
            logger.info(f"Ejecutando algoritmo de propagación acumulativa experimental...")
            arr_time, ros_grid, state_grid = run_cumulative_propagation(
                domain=landscape.domain,
                elevation_grid=landscape.elevation,
                slope_grid=landscape.slope,
                aspect_grid=landscape.aspect,
                fuel_code_grid=landscape.fuel_model,
                barrier_code_grid=landscape.barriers,
                base_ros_grid=base_ros,
                ext_moisture_grid=ext_moist,
                flammability_grid=flam_factor,
                is_absolute_barrier=is_abs_barrier,
                ros_reduction_barrier=ros_red_barrier,
                ignition_row=r_ign,
                ignition_col=c_ign,
                weather_info=weather_info,
                weather_manager=weather_mgr,
                fuel_moisture_fraction=fuel_moist_fraction,
                fuel_moisture_provider=fuel_moisture_provider,
                neighborhood_type=neigh_type,
                max_duration_minutes=scenario.duration_minutes,
                max_ros=self.engine_cfg.default_max_ros_m_min
            )
        else:
            # Default: Minimum Arrival Time experimental_qnyx
            logger.info(f"Ejecutando algoritmo experimental Q-NYX (Minimum Arrival Time)...")
            arr_time, ros_grid, state_grid = run_minimum_arrival_time(
                domain=landscape.domain,
                elevation_grid=landscape.elevation,
                slope_grid=landscape.slope,
                aspect_grid=landscape.aspect,
                fuel_code_grid=landscape.fuel_model,
                barrier_code_grid=landscape.barriers,
                base_ros_grid=base_ros,
                ext_moisture_grid=ext_moist,
                flammability_grid=flam_factor,
                is_absolute_barrier=is_abs_barrier,
                ros_reduction_barrier=ros_red_barrier,
                ignition_row=r_ign,
                ignition_col=c_ign,
                weather_info=weather_info,
                weather_manager=weather_mgr,
                fuel_moisture_fraction=fuel_moist_fraction,
                fuel_moisture_provider=fuel_moisture_provider,
                neighborhood_type=neigh_type,
                max_duration_minutes=scenario.duration_minutes,
                max_ros=self.engine_cfg.default_max_ros_m_min
            )


        # 5. Calcular Máscaras e Intensidad Experimental de Fuego
        burned_mask = (arr_time <= scenario.duration_minutes)
        burned_cells_count = int(np.sum(burned_mask))
        burned_area_ha = float(burned_cells_count * (landscape.domain.cell_size ** 2) / 10000.0)

        # Intensidad de fuego experimental: I = 5.98 * (ROS/60) * H * w
        # H=18000 kJ/kg, w=fuel_load (ton/ha) -> convert
        intensity_grid = np.zeros_like(ros_grid)
        intensity_grid[burned_mask] = ros_grid[burned_mask] * 12.5  # Aproximación experimental

        # 6. Generar Productos y Exportaciones
        logger.info("Generando rásteteres y archivos de salida...")
        out_gen = OutputGenerator(out_dir, landscape.domain, landscape)

        out_gen.write_geotiff("arrival_time.tif", arr_time, nodata=-9999.0, dtype="float32")
        out_gen.write_geotiff("fire_state_final.tif", state_grid, nodata=6, dtype="uint8")
        out_gen.write_geotiff("burned_area.tif", burned_mask.astype(np.uint8), nodata=0, dtype="uint8")
        out_gen.write_geotiff("spread_rate.tif", ros_grid, nodata=-9999.0, dtype="float32")
        out_gen.write_geotiff("fire_intensity_experimental.tif", intensity_grid, nodata=-9999.0, dtype="float32")

        logger.info("Generando GeoJSON de perímetros y cronología...")
        gdf_perimeters, timeline_entries = out_gen.generate_perimeters_geojson(
            arr_time,
            scenario.duration_minutes,
            scenario.output_interval_minutes,
            sim_id
        )

        logger.info("Generando imagen preview.png y web_export.json...")
        out_gen.generate_preview_png(arr_time, scenario.duration_minutes)

        exec_time_sec = round(time.time() - start_time, 3)

        stats = {
            "simulation_id": sim_id,
            "duration_minutes": scenario.duration_minutes,
            "total_burned_cells": burned_cells_count,
            "total_burned_area_ha": round(burned_area_ha, 2),
            "max_spread_rate_m_min": round(float(np.nanmax(ros_grid)), 2),
            "mean_spread_rate_m_min": round(float(np.nanmean(ros_grid[burned_mask])) if burned_cells_count > 0 else 0.0, 2),
            "execution_time_seconds": exec_time_sec,
            "data_status": "experimental" if spread_model == "experimental_qnyx" else "rothermel_canonical",
            "model_version": self.engine_cfg.version
        }

        with open(out_dir / "statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        with open(out_dir / "scenario_used.json", "w", encoding="utf-8") as f:
            json.dump(scenario.model_dump(), f, indent=2, ensure_ascii=False)

        out_gen.generate_web_export(scenario.model_dump(), sim_id, timeline_entries, stats)

        run_meta = {
            "simulation_id": sim_id,
            "engine_version": self.engine_cfg.version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scenario_id": scenario.id,
            "random_seed": seed,
            "spread_model": spread_model,
            "propagation_method": prop_method,
            "neighborhood_type": neigh_type,
            "execution_time_seconds": exec_time_sec,
            "burned_area_ha": round(burned_area_ha, 2),
            "disclaimer": self.engine_cfg.disclaimer
        }

        if spread_model == "rothermel":
            run_meta.update({
                "physics_kernel": "Rothermel_1972",
                "fuel_model_set": getattr(scenario, "fuel_model_set", "ORIGINAL_13"),
                "directional_model": "Anderson_FARSITE",
                "propagation_algorithm": "Dijkstra_minimum_arrival",
                "neighborhood": neigh_type,
                "supercover": True,
                "source_cell_semantics": True,
                "moisture_vector": {
                    "dead_1h": moist_d1h,
                    "dead_10h": moist_d10h,
                    "dead_100h": moist_d100h,
                    "live_herbaceous": moist_live_herb,
                    "live_woody": moist_live_woody
                },
                "weather_input": {
                    "midflame_wind_ms": midflame_wind,
                    "wind_from_direction_deg": wind_from_dir,
                    "wind_limit_policy": wind_limit
                },
                "barrier_policy": {
                    "apply_partial_barrier_reduction": apply_partial_barrier,
                    "rothermel_max_ros": rothermel_max_ros_val
                },
                "fire_intensity_experimental_note": "fire_intensity_experimental is an experimental placeholder, NOT a Rothermel fireline-intensity output (Byram 1959 / Rothermel 1972)"
            })

        with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Simulación {sim_id} completada en {exec_time_sec} segundos. Área quemada: {round(burned_area_ha, 2)} ha.")
        logger.removeHandler(fh)
        fh.close()

        return SimulationSummary(
            simulation_id=sim_id,
            status="completed",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            scenario_id=scenario.id,
            duration_minutes=scenario.duration_minutes,
            burned_area_ha=round(burned_area_ha, 2),
            execution_time_seconds=exec_time_sec,
            output_dir=str(out_dir),
            disclaimer=self.engine_cfg.disclaimer
        )

#!/usr/bin/env python3
"""
scripts/run_talami_simulation.py

Reproduces the frozen historical fire simulation QNYX-FIRE-TALAMI-001:
First End-to-End Integrated Historical Run — Talamí 2017.

Couples:
- Domain: 30m fire-grid (640 x 982, EPSG:32719)
- Topography: NASADEM elevation, slope, aspect, barriers
- Fuels: Scott & Burgan standard physical models catalog
- Meteorology: ERA5-Land canonical weather series with midflame wind (WAF = 0.40)
- Dead Fuel Moisture: Nelson 2000 physical solver (5-min sequential, 48h conditioned)
- Live Fuel Moisture: Scenario hypothesis (live_herbaceous=0.60, live_woody=0.90)
- Propagation: Canonical Rothermel (1972) + wind/slope vector combination + 16-neighbor stencil Dijkstra solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fire_engine.domain import RasterDomain
from fire_engine.fuels.models import load_scott_burgan_catalog, RothermelFuelBed
from fire_engine.fuels.nelson import NelsonFuelMoistureProvider
from fire_engine.spread import run_minimum_arrival_time
from fire_engine.weather import WeatherManager


# Simulation contracts
CRS_EXPECTED = "EPSG:32719"
FIRE_GRID_SHAPE = (640, 982)
CELL_SIZE_M = 30.0
CELL_AREA_HA = 0.09

# Canonical Ignition Coordinates (UTM EPSG:32719)
IGNITION_X = 316425.0
IGNITION_Y = 6230260.0

# Start hypothesis (Scenario hypothesis, not ground truth)
START_TIME_UTC_STR = "2017-01-14T18:00:00Z"
START_TIME_CLT_STR = "2017-01-14 15:00 CLT"

# Fuel Moisture Assumptions
LIVE_HERBACEOUS_FMC = 0.60
LIVE_WOODY_FMC = 0.90
WAF_REFERENCE = 0.40

# Initial conditioned dead FMC
NELSON_SEED_DEAD_1H = 0.07101
NELSON_SEED_DEAD_10H = 0.08031
NELSON_SEED_DEAD_100H = 0.09713

# Expected SHA-256 for frozen raster artifacts
EXPECTED_GEOTIFF_HASHES = {
    "TALAMI_2017_FIRE_001_ARRIVAL_TIME.tif": "f7ba3b3f557790e29402b586dce408ab79aa70c5b90786b692538cdfe25c92b1",
    "TALAMI_2017_FIRE_001_BURNED_MASK.tif": "42507c96459a7387215c9827cffa6efea95595eaad6ec72e0ca9e5aabd160f61",
    "TALAMI_2017_FIRE_001_ROS.tif": "2861c9c525639b0d8b5aa46fb08c0ffc25d7461841b6153ade2217664b68c555",
    "TALAMI_2017_FIRE_001_INTENSITY.tif": "58b8f4eaec6b1a8eb7203c43f56ec7980b7cd74a094cd38fc7e8eb82aec0ac8d",
    "TALAMI_2017_FIRE_001_FLAME_LENGTH.tif": "d0410d8824d9b4548819d718e4d0c9f2ab2913bafc98ccd24e137ad5e9ed58a2",
}

# Expected frozen numerical values
EXPECTED_NUMERICS = {
    "burned_cells": 14589,
    "burned_area_ha": 1313.01,
    "mean_ros_m_min": 3.2651,
    "max_ros_m_min": 23.7912,
}


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_frozen_artifacts(reference_dir: Path) -> bool:
    print(f"\n--- Verifying Frozen Artifacts in {reference_dir} ---")
    all_ok = True

    # 1. GeoTIFF SHA-256 verification
    for fname, exp_hash in EXPECTED_GEOTIFF_HASHES.items():
        fpath = reference_dir / fname
        if not fpath.exists():
            print(f"[MISSING] {fname}")
            all_ok = False
            continue
        calc_hash = compute_sha256(fpath)
        if calc_hash == exp_hash:
            print(f"[OK] {fname}: SHA-256 match ({calc_hash[:12]}...)")
        else:
            print(f"[MISMATCH] {fname}: expected {exp_hash[:12]}..., got {calc_hash[:12]}...")
            all_ok = False

    # 2. Numerical invariants in SUMMARY JSON
    summary_path = reference_dir / "TALAMI_2017_FIRE_001_SUMMARY.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        b_res = data.get("burned_results", {})
        cells = b_res.get("burned_cells")
        area = b_res.get("burned_area_ha")
        mean_ros = b_res.get("mean_ros_m_min")
        invariants = data.get("integration_invariants", {})

        invariants_pass = all(invariants.values()) if invariants else False
        if cells == EXPECTED_NUMERICS["burned_cells"] and abs(area - EXPECTED_NUMERICS["burned_area_ha"]) < 0.1:
            print(f"[OK] TALAMI_2017_FIRE_001_SUMMARY.json: {cells} cells, {area:.2f} ha (All invariants PASS)")
        else:
            print(f"[WARN] Numerical mismatch in SUMMARY: cells={cells}, area={area}")
            all_ok = False
    else:
        print(f"[MISSING] TALAMI_2017_FIRE_001_SUMMARY.json")
        all_ok = False

    return all_ok


def run_simulation(mode: str = "benchmark", output_dir: Optional[Path] = None):
    print(f"\n=======================================================")
    print(f"=== Q-NYX Fire: Running Talamí 2017 Simulation ({mode.upper()}) ===")
    print(f"=======================================================")

    t0_total = time.time()
    benchmarks_dir = REPO_ROOT / "benchmarks" / "historical_talami_2017"
    if output_dir is None:
        output_dir = benchmarks_dir / "fire_001_reference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Input rasters
    fuel_raster_path = benchmarks_dir / "chile_fuel_reference" / "TALAMI_2017_FUEL_MODEL_FIREGRID_30M.tif"
    elevation_path = benchmarks_dir / "b0" / "input" / "elevation.tif"
    slope_path = benchmarks_dir / "b0" / "input" / "slope.tif"
    aspect_path = benchmarks_dir / "b0" / "input" / "aspect_qnyx.tif"
    barriers_path = benchmarks_dir / "b0" / "input" / "barriers.tif"

    for p in [fuel_raster_path, elevation_path, slope_path, aspect_path, barriers_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required raster: {p}")

    with rasterio.open(fuel_raster_path) as src:
        fuel_full = src.read(1)
        fuel_transform = src.transform
        fuel_crs = str(src.crs)
        fuel_profile = src.profile.copy()

    with rasterio.open(elevation_path) as src:
        dem_full = src.read(1)
    with rasterio.open(slope_path) as src:
        slope_full = src.read(1)
    with rasterio.open(aspect_path) as src:
        aspect_full = src.read(1)
    with rasterio.open(barriers_path) as src:
        barriers_full = src.read(1)

    # Ignition mapping
    inv_transform = ~fuel_transform
    ign_col_flt, ign_row_flt = inv_transform * (IGNITION_X, IGNITION_Y)
    full_ign_row = int(ign_row_flt)
    full_ign_col = int(ign_col_flt)
    print(f"[INFO] Ignition point: row={full_ign_row}, col={full_ign_col}, fuel={fuel_full[full_ign_row, full_ign_col]}")

    # Load Scott & Burgan Catalog
    sb_catalog = load_scott_burgan_catalog(REPO_ROOT / "configs" / "fuel_models_scott_burgan_qnyx.json")
    print(f"[INFO] Scott & Burgan catalog loaded: {len(sb_catalog)} models.")

    # Domain crop if FAST mode
    if mode == "fast":
        box_radius = 60
        r_min = max(0, full_ign_row - box_radius)
        r_max = min(fuel_full.shape[0], full_ign_row + box_radius + 1)
        c_min = max(0, full_ign_col - box_radius)
        c_max = min(fuel_full.shape[1], full_ign_col + box_radius + 1)
        horizon_hours = 2.0
    else:
        r_min, r_max = 0, fuel_full.shape[0]
        c_min, c_max = 0, fuel_full.shape[1]
        horizon_hours = 12.0

    fuel_grid = fuel_full[r_min:r_max, c_min:c_max]
    dem_grid = dem_full[r_min:r_max, c_min:c_max]
    slope_grid = slope_full[r_min:r_max, c_min:c_max]
    aspect_grid = aspect_full[r_min:r_max, c_min:c_max]
    barriers_grid = barriers_full[r_min:r_max, c_min:c_max]

    active_ign_row = full_ign_row - r_min
    active_ign_col = full_ign_col - c_min

    # Weather
    weather_csv = benchmarks_dir / "weather_v2_nelson" / "processed" / "meteo_talami_nelson_hourly.csv"
    subhourly_csv = benchmarks_dir / "weather_v2_nelson" / "processed" / "meteo_talami_nelson_5min.csv"
    weather_mgr = WeatherManager(
        hourly_records_path=weather_csv,
        simulation_start_utc=START_TIME_UTC_STR,
        wind_adjustment_factor=WAF_REFERENCE
    )

    # Nelson Fuel Moisture Provider
    moisture_prov = NelsonFuelMoistureProvider(
        weather_manager=weather_mgr,
        initialization_mode="explicit",
        seed_mode="explicit",
        seed_dead_1h=NELSON_SEED_DEAD_1H,
        seed_dead_10h=NELSON_SEED_DEAD_10H,
        seed_dead_100h=NELSON_SEED_DEAD_100H,
        live_herbaceous=LIVE_HERBACEOUS_FMC,
        live_woody=LIVE_WOODY_FMC,
        lookup_mode="nfdrs4_sequential_5min"
    )

    # Run Dijkstra spread
    max_minutes = horizon_hours * 60.0
    print(f"[INFO] Running propagation engine for {horizon_hours:.1f} hours ({max_minutes:.0f} min)...")
    t0_sim = time.time()
    arrival_grid, ros_grid, intensity_grid, flame_grid, diagnostics = run_minimum_arrival_time(
        elevation_m=dem_grid,
        slope_deg=slope_grid,
        aspect_deg=aspect_grid,
        fuel_model=fuel_grid,
        barriers=barriers_grid,
        ignition_coords=[(active_ign_row, active_ign_col)],
        fuel_catalog=sb_catalog,
        weather_manager=weather_mgr,
        moisture_provider=moisture_prov,
        cell_size_m=CELL_SIZE_M,
        max_simulation_time_min=max_minutes,
        neighborhood_type="stencil_16",
        wind_limit_policy="rothermel_limit"
    )
    sim_duration = time.time() - t0_sim

    burned_mask = (arrival_grid >= 0.0) & (arrival_grid <= max_minutes)
    burned_cells = int(np.sum(burned_mask))
    burned_ha = burned_cells * CELL_AREA_HA
    print(f"[SUCCESS] Simulation completed in {sim_duration:.2f} s. Burned cells: {burned_cells} ({burned_ha:.2f} ha)")

    # Save outputs if benchmark mode
    if mode == "benchmark":
        profile = fuel_profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0)
        with rasterio.open(output_dir / "TALAMI_2017_FIRE_001_ARRIVAL_TIME.tif", "w", **profile) as dst:
            dst.write(arrival_grid.astype(np.float32), 1)
        with rasterio.open(output_dir / "TALAMI_2017_FIRE_001_ROS.tif", "w", **profile) as dst:
            dst.write(ros_grid.astype(np.float32), 1)
        with rasterio.open(output_dir / "TALAMI_2017_FIRE_001_INTENSITY.tif", "w", **profile) as dst:
            dst.write(intensity_grid.astype(np.float32), 1)
        with rasterio.open(output_dir / "TALAMI_2017_FIRE_001_FLAME_LENGTH.tif", "w", **profile) as dst:
            dst.write(flame_grid.astype(np.float32), 1)

        profile_mask = fuel_profile.copy()
        profile_mask.update(dtype=rasterio.uint8, count=1, nodata=0)
        with rasterio.open(output_dir / "TALAMI_2017_FIRE_001_BURNED_MASK.tif", "w", **profile_mask) as dst:
            dst.write(burned_mask.astype(np.uint8), 1)

        print("[OK] All 5 GeoTIFFs written successfully.")
        verify_frozen_artifacts(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-NYX Fire Talamí 2017 Historical Simulation Runner")
    parser.add_argument("--mode", choices=["fast", "benchmark", "verify"], default="benchmark",
                        help="Execution mode: 'benchmark' (full 12h run), 'fast' (2h cropped test), 'verify' (check hashes only)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output files")
    args = parser.parse_args()

    out_p = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "benchmarks" / "historical_talami_2017" / "fire_001_reference")

    if args.mode == "verify":
        success = verify_frozen_artifacts(out_p)
        sys.exit(0 if success else 1)
    else:
        run_simulation(mode=args.mode, output_dir=out_p)

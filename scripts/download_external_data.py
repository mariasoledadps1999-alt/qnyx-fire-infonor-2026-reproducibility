#!/usr/bin/env python3
"""
scripts/download_external_data.py

Reference metadata, source URLs, access protocols, and integrity verification
for all external data sources utilized in the INFONOR 2026 paper experiments:
1. NASA NASADEM Digital Elevation Model (NASA JPL / Planetary Computer)
2. Copernicus Climate Change Service (C3S) / ECMWF ERA5-Land Reanalysis
3. CONAF Chile Historical Wildfire Perimeter Database (CONAF 2017)
4. USDA Forest Service Scott & Burgan Fuel Catalog (RMRS-GTR-153)

This script verifies local artifact integrity (SHA-256) and provides automated
guidance on how to fetch or reconstruct any external datasets from official primary sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent

EXTERNAL_SOURCES: Dict[str, Dict[str, Any]] = {
    "NASADEM_TOPOGRAPHY": {
        "provider": "NASA JPL / USGS / Microsoft Planetary Computer",
        "product": "NASADEM Merged DEM Global 1 arc-second (30m)",
        "tile": "NASADEM_HGT_s34w071",
        "license": "Public Domain (NASA Open Data Policy / US Government Work)",
        "redistribution": "Permitted with attribution",
        "source_url": "https://planetarycomputer.microsoft.com/dataset/nasadem",
        "alternative_url": "https://search.earthdata.nasa.gov/search?q=NASADEM_NC_001",
        "bounding_box_wgs84": {
            "west": -71.25,
            "south": -34.00,
            "east": -70.90,
            "north": -33.80
        },
        "target_local_files": {
            "elevation": "benchmarks/historical_talami_2017/b0/input/elevation.tif",
            "slope": "benchmarks/historical_talami_2017/b0/input/slope.tif",
            "aspect": "benchmarks/historical_talami_2017/b0/input/aspect_qnyx.tif",
            "barriers": "benchmarks/historical_talami_2017/b0/input/barriers.tif"
        },
        "sha256": {
            "elevation.tif": "01879d71fa964772c49bfc44579b2dc8d5edc93e238da3ddf73a22da6d31e3f7",
            "slope.tif": "a7fc7e82a00280ef050f3d791e38f7f4141ecc753a97eec560b8189cef97ed86",
            "aspect_qnyx.tif": "2c113c5339d1bfe0231108a74de5238b45d489fad78e9129e4166282d016fc6a",
            "barriers.tif": "77ddd8ee118f9062badbdc7a400bbbcc3611b186be7227b7bbf367551c08ccc6"
        }
    },
    "ERA5_LAND_METEOROLOGY": {
        "provider": "Copernicus Climate Change Service (C3S) / ECMWF",
        "product": "ERA5-Land hourly data from 1950 to present",
        "license": "Creative Commons Attribution 4.0 International (CC-BY 4.0) / Copernicus Open Access Policy",
        "citation": "Generated using Copernicus Climate Change Service information [2017]",
        "redistribution": "Permitted with attribution",
        "source_url": "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land",
        "cds_api_dataset": "reanalysis-era5-land",
        "temporal_coverage": {
            "start": "2017-01-12T00:00:00Z",
            "end": "2017-01-16T23:00:00Z"
        },
        "variables": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "2m_dewpoint_temperature",
            "surface_solar_radiation_downwards",
            "total_precipitation"
        ],
        "target_local_files": {
            "canonical_nc": "benchmarks/historical_talami_2017/weather_v2_nelson/raw/canonical_era5_land/era5_land_talami_2017_canonical.nc",
            "hourly_csv": "benchmarks/historical_talami_2017/weather_v2_nelson/processed/meteo_talami_nelson_hourly.csv",
            "subhourly_5min_csv": "benchmarks/historical_talami_2017/weather_v2_nelson/processed/meteo_talami_nelson_5min.csv"
        },
        "sha256": {
            "era5_land_talami_2017_canonical.nc": "78fdf43b080c6505ae8561f912e07bfab2bf1c28e61368dffa19a3b5d1c680b3",
            "meteo_talami_nelson_hourly.csv": "2d7bc7cecb8c5fc0acc03e9aa07a733d5dc6eb70b140a92c0c81f9b142255922",
            "meteo_talami_nelson_5min.csv": "d21f3abb5f21493089e8867d6f6829905535de34b811c94ce8c269326b3a8ca7"
        }
    },
    "CONAF_TALAMI_PERIMETER": {
        "provider": "Corporación Nacional Forestal (CONAF), Chile",
        "product": "Base de Datos Histórica de Incendios Forestales / Perímetros Oficiales de Incendios",
        "incident": "Temporada 2016-2017, Incendio 261_TALAMI (Alhué / Melipilla / Maule Domain)",
        "license": "Datos Abiertos del Estado de Chile (Ley 20.285 / Política de Datos Abiertos)",
        "citation": "Fuente: Corporación Nacional Forestal (CONAF), Chile.",
        "redistribution": "Permitted for scientific and educational research with attribution",
        "source_url": "https://www.conaf.cl/situacion-nacional-de-incendios-forestales/",
        "target_local_files": {
            "observed_perimeter": "benchmarks/historical_talami_2017/conaf_reference/observed_perimeter_talami_2017.gpkg"
        },
        "sha256": {
            "observed_perimeter_talami_2017.gpkg": "f2409aad372e003d7371c582c5729ad9d0481be5df6dd48644655a71f55bc028"
        }
    },
    "SCOTT_BURGAN_FUEL_CATALOG": {
        "provider": "USDA Forest Service, Rocky Mountain Research Station",
        "product": "Standard Fire Behavior Fuel Models: A Comprehensive Set for Use with Rothermel's Surface Fire Spread Model (RMRS-GTR-153)",
        "authors": "Joe H. Scott and Robert E. Burgan (2005)",
        "license": "Public Domain (US Federal Government Work)",
        "redistribution": "Permitted",
        "source_url": "https://www.fs.usda.gov/research/treesearch/21808",
        "target_local_files": {
            "catalog_json": "configs/fuel_models_scott_burgan_qnyx.json"
        },
        "sha256": {
            "fuel_models_scott_burgan_qnyx.json": "a86df6efd427d8e16593c0a4168c6ad63583d8421f2f98bd55021ea43ad887b0"
        }
    }
}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_local_integrity() -> bool:
    print("=================================================================")
    print("=== Checking Integrity of External Scientific Data Invariants ===")
    print("=================================================================")
    all_ok = True

    for source_key, meta in EXTERNAL_SOURCES.items():
        print(f"\n--- {source_key} ({meta['provider']}) ---")
        for fname, exp_hash in meta["sha256"].items():
            # Find the path in target_local_files
            rel_path = None
            for p in meta["target_local_files"].values():
                if p.endswith(fname):
                    rel_path = p
                    break
            if not rel_path:
                rel_path = fname

            local_file = REPO_ROOT / rel_path
            if not local_file.exists():
                print(f"[MISSING] {rel_path}")
                all_ok = False
                continue

            calc_hash = compute_sha256(local_file)
            if calc_hash == exp_hash:
                print(f"[OK] {fname}: SHA-256 match ({calc_hash[:12]}...)")
            else:
                print(f"[MISMATCH] {fname}: expected {exp_hash[:12]}..., got {calc_hash[:12]}...")
                all_ok = False

    print("\n-----------------------------------------------------------------")
    if all_ok:
        print("[SUCCESS] All external dataset artifacts match official frozen hashes.")
    else:
        print("[WARNING] One or more external files are missing or modified.")
    return all_ok


def print_info():
    print(json.dumps(EXTERNAL_SOURCES, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Data Provenance & Integrity Auditor")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify SHA256 of local data artifacts")
    parser.add_argument("--info", action="store_true", help="Print structured metadata of external datasets")
    args = parser.parse_args()

    if args.info:
        print_info()
    else:
        ok = verify_local_integrity()
        sys.exit(0 if ok else 1)

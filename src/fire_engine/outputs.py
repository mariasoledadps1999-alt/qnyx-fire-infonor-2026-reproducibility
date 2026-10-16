"""
outputs.py

Generación y exportación de productos GeoTIFF, GeoJSON, JSON y vista previa PNG.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class OutputGenerator:
    def __init__(self, output_dir: Path, domain, landscape_data):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.landscape = landscape_data

    def write_geotiff(self, filename: str, data: np.ndarray, nodata: float = -9999.0, dtype: str = "float32"):
        """Escribe un array 2D como GeoTIFF alineado con el dominio."""
        filepath = self.output_dir / filename
        data_write = np.nan_to_num(data, nan=nodata)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=self.domain.rows,
            width=self.domain.cols,
            count=1,
            dtype=dtype,
            crs=f"EPSG:{self.domain.crs_epsg}",
            transform=self.domain.transform,
            nodata=nodata
        ) as dst:
            dst.write(data_write.astype(dtype), 1)
        return str(filename)

    def generate_perimeters_geojson(
        self,
        arrival_time_grid: np.ndarray,
        duration_minutes: int,
        interval_minutes: int,
        simulation_id: str
    ) -> Tuple[gpd.GeoDataFrame, List[Dict[str, Any]]]:
        """
        Genera polígonos perimetrales en GeoJSON para cada intervalo de salida (15, 30, ..., duration).
        Calcula área quemada en hectáreas y longitud de perímetro en metros.
        """
        features = []
        timeline_entries = []

        timesteps = list(range(interval_minutes, duration_minutes + 1, interval_minutes))
        if duration_minutes not in timesteps:
            timesteps.append(duration_minutes)

        for t in timesteps:
            mask = (arrival_time_grid <= t)
            burned_cells = int(np.sum(mask))
            burned_ha = float(burned_cells * (self.domain.cell_size ** 2) / 10000.0)

            geom_list = []
            perimeter_len_m = 0.0

            if burned_cells > 0:
                # Contorno con scikit-image
                contours = measure.find_contours(mask.astype(float), 0.5)
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    # contour es (row, col)
                    coords_utm = [self.domain.grid_to_utm(r, c) for r, c in contour]
                    poly = Polygon(coords_utm)
                    if poly.is_valid and poly.area > 0:
                        geom_list.append(poly)
                        perimeter_len_m += float(poly.length)

            if geom_list:
                if len(geom_list) == 1:
                    merged_geom = geom_list[0]
                else:
                    merged_geom = MultiPolygon(geom_list)
            else:
                merged_geom = Polygon()

            feature = {
                "type": "Feature",
                "geometry": merged_geom.__geo_interface__ if not merged_geom.is_empty else None,
                "properties": {
                    "simulation_id": simulation_id,
                    "elapsed_minutes": t,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t * 60)),
                    "burned_area_hectares": round(burned_ha, 2),
                    "perimeter_length_m": round(perimeter_len_m, 2),
                    "data_status": "experimental",
                    "model_version": "1.0.0"
                }
            }
            features.append(feature)

            # Bounds UTM
            if not merged_geom.is_empty:
                b = merged_geom.bounds
                spatial_extent = [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)]
            else:
                spatial_extent = [0.0, 0.0, 0.0, 0.0]

            timeline_entries.append({
                "elapsed_minutes": t,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t * 60)),
                "burned_area_hectares": round(burned_ha, 2),
                "active_cells_count": int(np.sum(np.isclose(arrival_time_grid, t, atol=interval_minutes/2.0))),
                "cumulative_burned_cells": burned_cells,
                "spatial_extent_utm": spatial_extent,
                "perimeter_ref": f"perimeters.geojson#t={t}"
            })

        gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{self.domain.crs_epsg}")
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        gdf_wgs84.to_file(self.output_dir / "perimeters.geojson", driver="GeoJSON")

        with open(self.output_dir / "timeline.json", "w", encoding="utf-8") as f:
            json.dump(timeline_entries, f, indent=2, ensure_ascii=False)

        return gdf, timeline_entries

    def generate_preview_png(self, arrival_time_grid: np.ndarray, duration_minutes: int):
        """Genera un archivo preview.png del mapa de tiempo de llegada."""
        plt.figure(figsize=(10, 10))
        display_grid = np.where(arrival_time_grid <= duration_minutes, arrival_time_grid, np.nan)
        
        plt.imshow(self.landscape.elevation, cmap="terrain", alpha=0.6)
        plt.imshow(display_grid, cmap="YlOrRd", vmin=0, vmax=duration_minutes, alpha=0.8)
        plt.colorbar(label="Tiempo de llegada (minutos)")
        plt.title(f"Simulación Propagación Incendio - {duration_minutes} min")
        plt.xlabel("Columna")
        plt.ylabel("Fila")
        plt.tight_layout()
        plt.savefig(self.output_dir / "preview.png", dpi=150)
        plt.close()

    def generate_web_export(
        self,
        scenario_data: Dict[str, Any],
        simulation_id: str,
        timeline_entries: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ):
        """
        Genera el archivo web_export.json optimizado para Three.js / CesiumJS.
        """
        web_data = {
            "metadata": {
                "landscape_id": "CL-MAULE-PILOT-001",
                "simulation_id": simulation_id,
                "crs": f"EPSG:{self.domain.crs_epsg}",
                "extent_epsg32719": list(self.domain.extent_utm),
                "extent_epsg4326": list(self.domain.extent_wgs84),
                "resolution_m": self.domain.cell_size,
                "rows": self.domain.rows,
                "columns": self.domain.cols,
                "duration_minutes": scenario_data.get("duration_minutes", 360),
                "output_interval_minutes": scenario_data.get("output_interval_minutes", 15),
                "disclaimer": (
                    "Prototipo experimental de simulación. No utilizar para coordinación "
                    "de emergencias, evacuaciones ni decisiones operacionales sin validación científica e institucional."
                )
            },
            "ignition": scenario_data.get("ignition", {}),
            "weather": scenario_data.get("weather", {}),
            "rasters": {
                "arrival_time": "arrival_time.tif",
                "fire_state_final": "fire_state_final.tif",
                "burned_area": "burned_area.tif",
                "spread_rate": "spread_rate.tif",
                "fire_intensity_experimental": "fire_intensity_experimental.tif",
                "preview_png": "preview.png"
            },
            "perimeters_geojson": "perimeters.geojson",
            "timeline": timeline_entries,
            "statistics": stats
        }

        with open(self.output_dir / "web_export.json", "w", encoding="utf-8") as f:
            json.dump(web_data, f, indent=2, ensure_ascii=False)

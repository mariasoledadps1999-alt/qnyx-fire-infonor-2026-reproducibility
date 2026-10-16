"""
raster_loader.py

Carga, verificación de alineación y validación de capas GeoTIFF.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import rasterio
from fire_engine.domain import RasterDomain

REQUIRED_RASTERS = ["elevation", "slope", "aspect", "fuel_model", "barriers"]
OPTIONAL_RASTERS = ["land_cover", "ndvi", "ndmi", "nbr", "hillshade"]

class RasterValidationError(ValueError):
    pass

class LandscapeData:
    def __init__(self, domain: RasterDomain, arrays: Dict[str, np.ndarray], rasters_info: Dict[str, Dict[str, Any]]):
        self.domain = domain
        self.arrays = arrays
        self.rasters_info = rasters_info

    @property
    def elevation(self) -> np.ndarray:
        return self.arrays["elevation"]

    @property
    def slope(self) -> np.ndarray:
        return self.arrays["slope"]

    @property
    def aspect(self) -> np.ndarray:
        return self.arrays["aspect"]

    @property
    def fuel_model(self) -> np.ndarray:
        return self.arrays["fuel_model"]

    @property
    def barriers(self) -> np.ndarray:
        return self.arrays["barriers"]

def load_and_validate_rasters(
    base_dir: Path,
    raster_paths: Dict[str, str],
    expected_crs: int = 32719,
    expected_res: float = 30.0,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None
) -> LandscapeData:
    """
    Carga y valida todos los rásteteres especificados en raster_paths.
    Garantiza que todos existan, estén alineados, tengan el mismo CRS y dimensiones.
    """
    arrays = {}
    rasters_info = {}
    master_transform = None
    master_crs = None
    master_rows = expected_rows
    master_cols = expected_cols

    # Verificar presencia de requeridos
    for name in REQUIRED_RASTERS:
        if name not in raster_paths:
            raise RasterValidationError(f"Falta la capa obligatoria '{name}' en la lista de rasters.")

    for name, rel_path in raster_paths.items():
        full_path = base_dir / rel_path
        if not full_path.exists():
            if name in REQUIRED_RASTERS:
                raise RasterValidationError(f"Archivo GeoTIFF obligatorio no encontrado: {full_path}")
            else:
                continue

        try:
            with rasterio.open(full_path) as src:
                data = src.read(1)
                
                # Validar CRS
                if src.crs is None or src.crs.to_epsg() != expected_crs:
                    raise RasterValidationError(
                        f"Raster '{name}' ({rel_path}) tiene CRS '{src.crs}', se esperaba EPSG:{expected_crs}"
                    )

                # Validar Dimensiones
                if master_rows is None:
                    master_rows = src.height
                    master_cols = src.width
                elif src.height != master_rows or src.width != master_cols:
                    raise RasterValidationError(
                        f"Raster '{name}' ({rel_path}) tiene dimensiones ({src.height}, {src.width}), "
                        f"no coincide con la grilla maestra ({master_rows}, {master_cols})"
                    )

                # Validar Resolución
                res_x, res_y = abs(src.res[0]), abs(src.res[1])
                if not (np.isclose(res_x, expected_res, atol=0.5) and np.isclose(res_y, expected_res, atol=0.5)):
                    raise RasterValidationError(
                        f"Raster '{name}' ({rel_path}) tiene resolución ({res_x}, {res_y}), "
                        f"se esperaba ({expected_res}, {expected_res})"
                    )

                # Validar Transform / Alineación
                if master_transform is None:
                    master_transform = src.transform
                    master_crs = src.crs
                else:
                    if not (
                        np.isclose(src.transform.c, master_transform.c, atol=1.0) and
                        np.isclose(src.transform.f, master_transform.f, atol=1.0)
                    ):
                        raise RasterValidationError(
                            f"Raster '{name}' ({rel_path}) no está alineado con la grilla maestra. "
                            f"Transform origin: ({src.transform.c}, {src.transform.f}) vs ({master_transform.c}, {master_transform.f})"
                        )

                arrays[name] = data
                rasters_info[name] = {
                    "path": str(rel_path),
                    "dtype": str(data.dtype),
                    "nodata": src.nodata,
                    "min": float(np.nanmin(data)),
                    "max": float(np.nanmax(data))
                }

        except Exception as e:
            if isinstance(e, RasterValidationError):
                raise
            raise RasterValidationError(f"Error al leer GeoTIFF '{full_path}': {e}")

    domain = RasterDomain(
        rows=master_rows,
        cols=master_cols,
        cell_size=expected_res,
        crs_epsg=expected_crs,
        transform=master_transform
    )

    return LandscapeData(domain=domain, arrays=arrays, rasters_info=rasters_info)

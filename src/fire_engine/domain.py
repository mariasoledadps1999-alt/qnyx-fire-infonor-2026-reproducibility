"""
domain.py

Definición del dominio espacial raster, estados de celda, transformación de coordenadas
y conectividad de vecindad (Moore / Von Neumann).
"""

from enum import IntEnum
from typing import List, Tuple, Optional, Dict
import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine

class CellState(IntEnum):
    NO_FUEL = 0
    UNBURNED = 1
    BURNING = 2
    BURNED = 3
    BARRIER = 4
    OUTSIDE_DOMAIN = 5
    NO_DATA = 6

class RasterDomain:
    def __init__(
        self,
        rows: int = 1000,
        cols: int = 1000,
        cell_size: float = 30.0,
        crs_epsg: int = 32719,
        transform: Optional[Affine] = None,
        extent_utm: Optional[Tuple[float, float, float, float]] = None,
        extent_wgs84: Optional[Tuple[float, float, float, float]] = None
    ):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.crs_epsg = crs_epsg
        
        if transform is None:
            # Por defecto: Maule extent (190350, 6046080, 220350, 6076080)
            self.transform = Affine(cell_size, 0.0, 190350.0, 0.0, -cell_size, 6076080.0)
        else:
            self.transform = transform
            
        self.extent_utm = extent_utm or (
            self.transform.c,
            self.transform.f + self.rows * self.transform.e,
            self.transform.c + self.cols * self.transform.a,
            self.transform.f
        )
        self.extent_wgs84 = extent_wgs84 or (
            -72.4212836, -35.6895674, -72.0799134, -35.4105892
        )
        
        # Transformadores de coordenadas
        self._transformer_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True)
        self._transformer_to_wgs84 = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)

    def wgs84_to_grid(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convierte longitud/latitud (WGS84) a índices (fila, columna) en la grilla."""
        x_utm, y_utm = self._transformer_to_utm.transform(lon, lat)
        return self.utm_to_grid(x_utm, y_utm)

    def grid_to_wgs84(self, row: int, col: int) -> Tuple[float, float]:
        """Convierte índices (fila, columna) a longitud/latitud (WGS84)."""
        x_utm, y_utm = self.grid_to_utm(row, col)
        lon, lat = self._transformer_to_wgs84.transform(x_utm, y_utm)
        return float(lon), float(lat)

    def utm_to_grid(self, x_utm: float, y_utm: float) -> Tuple[int, int]:
        """Convierte coordenadas UTM a índices (fila, columna)."""
        col = int((x_utm - self.transform.c) / self.transform.a)
        row = int((y_utm - self.transform.f) / self.transform.e)
        return row, col

    def grid_to_utm(self, row: int, col: int) -> Tuple[float, float]:
        """Convierte índices (fila, columna) a coordenadas UTM (centro de celda)."""
        x_utm = self.transform.c + (col + 0.5) * self.transform.a
        y_utm = self.transform.f + (row + 0.5) * self.transform.e
        return x_utm, y_utm

    def is_inside(self, row: int, col: int) -> bool:
        """Verifica si la celda (row, col) está dentro de los límites del dominio."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_neighbors(
        self, row: int, col: int, neighborhood_type: str = "moore"
    ) -> List[Tuple[int, int, float, float]]:
        """
        Retorna la lista de vecinos válidos para (row, col) según el stencil de conectividad.
        Retorna tuplas: (n_row, n_col, distance_m, heading_deg)
        - distance_m: distancia euclidiana exacta en metros (cell_size * sqrt(dr^2 + dc^2)).
        - heading_deg: ángulo hacia el vecino respecto al norte geográfico en [0.0, 360.0).

        Stencils soportados:
        - "von_neumann" / "stencil_4": 4 vecinos ortogonales (radio 1, d=1.0).
        - "moore" / "stencil_8": 8 vecinos ortogonales y diagonales (radio 1, d=1.0 y d=sqrt(2)). [DEFAULT]
        - "stencil_16" / "16_neighbors": 16 vecinos (8 de radio 1 + 8 de salto de caballo d=sqrt(5)).
        - "stencil_24" / "24_neighbors": 24 vecinos (caja 5x5 completa de radio 2).
        """
        neighbors = []
        nt = neighborhood_type.lower()
        
        # 1. Ortogonales de radio 1 (4 vecinos)
        von_neumann_offsets = [
            (-1, 0, 1.0, 0.0),    # Norte
            (0, 1, 1.0, 90.0),    # Este
            (1, 0, 1.0, 180.0),   # Sur
            (0, -1, 1.0, 270.0)   # Oeste
        ]
        
        # 2. Diagonales de radio 1 (4 vecinos)
        diagonal_offsets = [
            (-1, 1, 1.4142135623730951, 45.0),   # Noreste
            (1, 1, 1.4142135623730951, 135.0),  # Sureste
            (1, -1, 1.4142135623730951, 225.0),  # Suroeste
            (-1, -1, 1.4142135623730951, 315.0)  # Noroeste
        ]

        # 3. Salto de caballo / Knight offsets (8 vecinos, d=sqrt(5) ≈ 2.23606798)
        knight_offsets = [
            (-2, 1, 2.23606797749979, 26.56505117707799),   # NNE (atan2(1, 2))
            (-1, 2, 2.23606797749979, 63.43494882292201),   # ENE (atan2(2, 1))
            (1, 2, 2.23606797749979, 116.56505117707799),   # ESE (atan2(2, -1))
            (2, 1, 2.23606797749979, 153.43494882292202),   # SSE (atan2(1, -2))
            (2, -1, 2.23606797749979, 206.56505117707798),  # SSW (atan2(-1, -2))
            (1, -2, 2.23606797749979, 243.43494882292202),  # WSW (atan2(-2, -1))
            (-1, -2, 2.23606797749979, 296.565051177078),   # WNW (atan2(-2, 1))
            (-2, -1, 2.23606797749979, 333.434948822922)    # NNW (atan2(-1, 2))
        ]

        # 4. Ortogonales y diagonales de radio 2 (8 vecinos adicionales para stencil_24)
        r2_offsets = [
            (-2, 0, 2.0, 0.0),                     # Norte (radio 2)
            (0, 2, 2.0, 90.0),                     # Este (radio 2)
            (2, 0, 2.0, 180.0),                    # Sur (radio 2)
            (0, -2, 2.0, 270.0),                   # Oeste (radio 2)
            (-2, 2, 2.8284271247461903, 45.0),    # Noreste (radio 2)
            (2, 2, 2.8284271247461903, 135.0),    # Sureste (radio 2)
            (2, -2, 2.8284271247461903, 225.0),   # Suroeste (radio 2)
            (-2, -2, 2.8284271247461903, 315.0)   # Noroeste (radio 2)
        ]

        if nt in ("von_neumann", "stencil_4", "4_neighbors"):
            offsets = von_neumann_offsets
        elif nt in ("moore", "stencil_8", "8_neighbors"):
            offsets = von_neumann_offsets + diagonal_offsets
        elif nt in ("stencil_16", "16_neighbors"):
            offsets = von_neumann_offsets + diagonal_offsets + knight_offsets
        elif nt in ("stencil_24", "24_neighbors"):
            offsets = von_neumann_offsets + diagonal_offsets + knight_offsets + r2_offsets
        else:
            offsets = von_neumann_offsets + diagonal_offsets

        for dr, dc, dist_mult, heading in offsets:
            nr, nc = row + dr, col + dc
            if self.is_inside(nr, nc):
                neighbors.append((nr, nc, self.cell_size * dist_mult, heading))

        return neighbors


def cells_crossed_by_edge(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """
    Retorna la lista de coordenadas (r, c) de celdas intermedias intersecadas por el segmento
    continuo entre los centros de las celdas (r0, c0) y (r1, c1) utilizando la geometría de raster supercover.
    Excluye la celda origen (r0, c0) y la celda destino (r1, c1).

    Para vecinos directos de radio 1 (|dr| <= 1 y |dc| <= 1), retorna una lista vacía [].
    Para saltos de caballo (|dr|=2, |dc|=1 o |dr|=1, |dc|=2), retorna exactamente las 2 celdas intermedias.
    Para conexiones de radio 2, retorna las celdas intermedias intersecadas.
    """
    dr = r1 - r0
    dc = c1 - c0

    # Vecinos directos de radio 1 (ortogonales y diagonales): adyacencia directa
    if abs(dr) <= 1 and abs(dc) <= 1:
        return []

    # Salto de caballo vertical: |dr|=2, |dc|=1
    if abs(dr) == 2 and abs(dc) == 1:
        step_r = dr // 2
        return [(r0 + step_r, c0), (r0 + step_r, c0 + dc)]

    # Salto de caballo horizontal: |dr|=1, |dc|=2
    if abs(dr) == 1 and abs(dc) == 2:
        step_c = dc // 2
        return [(r0, c0 + step_c), (r0 + dr, c0 + step_c)]

    # Conexiones de radio 2 ortogonales
    if abs(dr) == 2 and dc == 0:
        return [(r0 + dr // 2, c0)]
    if dr == 0 and abs(dc) == 2:
        return [(r0, c0 + dc // 2)]

    # Conexiones de radio 2 diagonales
    if abs(dr) == 2 and abs(dc) == 2:
        return [(r0 + dr // 2, c0 + dc // 2)]

    # Caso general por muestreo paramétrico de supercover
    intermediates = []
    seen = set()
    for t in (0.25, 0.50, 0.75):
        ir = int(round(r0 + t * dr))
        ic = int(round(c0 + t * dc))
        if (ir, ic) not in ((r0, c0), (r1, c1)) and (ir, ic) not in seen:
            seen.add((ir, ic))
            intermediates.append((ir, ic))
    return intermediates

